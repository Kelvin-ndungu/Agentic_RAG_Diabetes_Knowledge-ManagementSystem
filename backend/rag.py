"""LangGraph workflow and nodes for the RAG pipeline."""
import json
import re
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer

from .schema import ChatState, ClassifierOutput, GeneratorOutput, Source
from .llm import get_llm, Timer, invoke_with_retry
from .config import (
    RETRIEVAL_TOP_K,
    RETRIEVAL_MIN_SIMILARITY,
    LLM_MAX_RETRIES,
    LOG_TIMINGS,
)
from .prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_PROMPT,
)


chroma_reader = None


def set_chroma_reader(reader) -> None:
    """Set the global chroma_reader instance."""
    global chroma_reader
    chroma_reader = reader


def classify_query_unified(state: ChatState) -> ChatState:
    """
    Single LLM call handles all classification logic.
    """
    writer = get_stream_writer()

    classifier_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFIER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("human", CLASSIFIER_USER_PROMPT),
        ]
    )

    def parse_classifier_output(text: str) -> ClassifierOutput:
        """Parse classifier output - handles both JSON and markdown formats."""
        json_match = re.search(r"\{[^{}]*\"route\"[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return ClassifierOutput(**data)
            except Exception:
                pass

        try:
            data = json.loads(text.strip())
            return ClassifierOutput(**data)
        except Exception:
            pass

        route_match = re.search(r"\*\*route\*\*:\s*(\w+)", text)
        safety_match = re.search(r"\*\*safety\*\*:\s*(\w+)", text)
        status_message_match = re.search(r"\*\*status_message\*\*:\s*(.+?)(?=\n\*\*|\Z)", text, re.DOTALL)
        intent_match = re.search(r"\*\*intent\*\*:\s*(.+?)(?=\n\*\*|\Z)", text, re.DOTALL)
        direct_response_match = re.search(
            r"\*\*direct_response\*\*:\s*(.+?)(?=\n\*\*|\Z)", text, re.DOTALL
        )

        route = route_match.group(1) if route_match else "retrieve"
        safety = safety_match.group(1) if safety_match else "safe"
        status_message = status_message_match.group(1).strip() if status_message_match else "Processing query..."
        intent = intent_match.group(1).strip() if intent_match else None
        direct_response = direct_response_match.group(1).strip() if direct_response_match else None

        return ClassifierOutput(
            route=route,
            safety=safety,
            status_message=status_message,
            intent=intent,
            direct_response=direct_response,
        )

    try:
        llm = get_llm()
        chain = classifier_prompt | llm | StrOutputParser()
        with Timer("classifier", enabled=LOG_TIMINGS):
            raw_output = invoke_with_retry(
                chain, {"messages": state.get("messages", [])}, LLM_MAX_RETRIES, "classifier"
            )
        result = parse_classifier_output(raw_output)

        # Justification: unsafe queries should still retrieve general guidance with citations.
        if result.safety == "unsafe" and result.route == "direct":
            result.route = "retrieve"
            if not result.status_message:
                result.status_message = "Evaluating query safety..."

        # Justification: ensure we have a usable intent for retrieval even if the model omits it.
        if result.route == "retrieve" and not result.intent:
            last_user = None
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    last_user = msg.content
                    break
            result.intent = last_user or "Provide general guidance from the diabetes guidelines."

        state["classifier_output"] = result

        if result.route == "direct":
            state["final_response"] = result.direct_response
            if result.direct_response:
                state["messages"] = state.get("messages", []) + [AIMessage(content=result.direct_response)]

        if writer:
            if result.route == "retrieve" and result.intent:
                status_msg = f"I am getting the relevant resources to answer: {result.intent}"
            else:
                status_msg = result.status_message
            writer({"type": "classifier_status", "message": status_msg})

        print(f"✓ Classified: route={result.route}, safety={result.safety}")
        if result.intent:
            print(f"  Intent: {result.intent[:80]}...")

    except Exception as exc:
        print(f"⚠ Classification error: {exc}")
        import traceback

        traceback.print_exc()
        result = ClassifierOutput(
            route="direct",
            safety="irrelevant",
            status_message="Processing query...",
            intent=None,
            direct_response="I encountered an error while processing your query. Please try again.",
        )
        state["classifier_output"] = result
        state["final_response"] = result.direct_response

    return state


async def retrieval_node(state: ChatState) -> ChatState:
    """
    Programmatic retrieval based on classifier intent.
    """
    classifier_output = state.get("classifier_output")
    writer = get_stream_writer()

    if not classifier_output or classifier_output.route != "retrieve":
        return state

    intent = classifier_output.intent
    if not intent:
        print("⚠ No intent available for retrieval")
        return state

    try:
        with Timer("retrieval", enabled=LOG_TIMINGS):
            chunks = await chroma_reader.search(
                query=intent,
                n_results=RETRIEVAL_TOP_K,
                min_similarity=RETRIEVAL_MIN_SIMILARITY,
            )

        state["retrieved_chunks"] = chunks

        if writer:
            if chunks:
                writer({"type": "retrieval_status", "message": f"Found {len(chunks)} relevant sources. Generating answer..."})
            else:
                writer({"type": "retrieval_status", "message": "No sources found with sufficient relevance. Responding..."})

        print(f"✓ Retrieved {len(chunks)} chunks (similarity >= {RETRIEVAL_MIN_SIMILARITY})")
        if chunks:
            print(f"  Top relevance: {chunks[0]['relevance_score']:.3f}")
    except Exception as exc:
        print(f"⚠ Retrieval error: {exc}")
        import traceback

        traceback.print_exc()
        state["retrieved_chunks"] = []
        if writer:
            writer({"type": "retrieval_error", "message": "Error during retrieval. Continuing..."})

    return state


def generator_node(state: ChatState) -> ChatState:
    """
    Single LLM call for generation with conversation history.
    """
    writer = get_stream_writer()

    try:
        chunks = state.get("retrieved_chunks", [])
        classifier_output = state.get("classifier_output")

        if not classifier_output:
            state["final_response"] = "Error: No classifier output available."
            state["messages"] = state.get("messages", []) + [AIMessage(content=state["final_response"])]
            return state

        intent = classifier_output.intent
        if not intent:
            state["final_response"] = "Error: No intent available for generation."
            state["messages"] = state.get("messages", []) + [AIMessage(content=state["final_response"])]
            return state

        if chunks:
            context_parts = []
            chunk_to_source_map = {}
            seen_urls = {}

            for i, chunk in enumerate(chunks, 1):
                metadata = chunk.get("metadata", {})
                title = metadata.get("title", "Unknown")
                url = metadata.get("url", "")
                content = chunk.get("content", "")
                relevance = chunk.get("relevance_score", 0)

                context_parts.append(
                    f"--- Source {i}: {title} (Relevance: {relevance:.2f}) ---\nURL: {url}\n\n{content}"
                )

                source = Source(title=title, url=url, chunk_id=chunk.get("chunk_id", ""))
                chunk_to_source_map[i] = source
                if url and url not in seen_urls:
                    seen_urls[url] = source

            context = "\n\n".join(context_parts)
        else:
            context = "No relevant information found in knowledge base."
            chunk_to_source_map = {}
            seen_urls = {}

        generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERATOR_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", GENERATOR_USER_PROMPT),
            ]
        )

        has_sufficient_info = len(chunks) > 0 and any(
            chunk.get("relevance_score", 0) >= RETRIEVAL_MIN_SIMILARITY for chunk in chunks
        )

        llm = get_llm()
        chain = generator_prompt | llm | StrOutputParser()

        final_response = ""
        if writer:
            writer({"type": "generator_start", "message": "Generating answer..."})

        try:
            with Timer("generation_stream", enabled=LOG_TIMINGS):
                for chunk in chain.stream(
                    {
                        "messages": state.get("messages", []),
                        "intent": intent,
                        "context": context,
                        "safety": classifier_output.safety if classifier_output else "safe",
                    }
                ):
                    if chunk:
                        final_response += chunk
                        if writer:
                            writer({"type": "token", "content": chunk})
        except Exception as exc:
            print(f"⚠ Streaming failed, falling back to invoke: {exc}")
            final_response = ""

        if not final_response:
            response = invoke_with_retry(
                chain,
                {
                    "messages": state.get("messages", []),
                    "intent": intent,
                    "context": context,
                    "safety": classifier_output.safety if classifier_output else "safe",
                },
                LLM_MAX_RETRIES,
                "generator",
            )
            final_response = response if isinstance(response, str) else response.content
            if writer and final_response:
                writer({"type": "token", "content": final_response})

        if classifier_output and classifier_output.safety == "unsafe":
            disclaimer = (
                "I can share general information from the diabetes guidelines. "
                "For personal diagnosis or medication changes, please consult a licensed clinician."
            )
            if disclaimer not in final_response:
                final_response = f"{disclaimer}\n\n{final_response}"

        if "## Sources" in final_response:
            final_response = final_response.split("## Sources")[0].strip()

        referenced_chunk_numbers = set()
        max_chunk_num = len(chunks) if chunks else 0

        citation_pattern = r"\[(\d+)\](?!\()"
        matches = re.findall(citation_pattern, final_response)
        for num_str in matches:
            try:
                chunk_num = int(num_str)
                if 1 <= chunk_num <= max_chunk_num and chunk_num in chunk_to_source_map:
                    referenced_chunk_numbers.add(chunk_num)
                else:
                    print(f"  ⚠ Invalid citation [{chunk_num}] - out of range (valid: 1-{max_chunk_num})")
            except ValueError:
                pass

        referenced_urls = set()
        url_citation_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        url_matches = re.findall(url_citation_pattern, final_response)
        for title, url in url_matches:
            referenced_urls.add(url)

        cited_sources = []
        cited_urls = set()

        for chunk_num in sorted(referenced_chunk_numbers):
            if chunk_num in chunk_to_source_map:
                source = chunk_to_source_map[chunk_num]
                if source.url not in cited_urls:
                    cited_sources.append(source)
                    cited_urls.add(source.url)

        for url in referenced_urls:
            if url not in cited_urls:
                if url in seen_urls:
                    source = seen_urls[url]
                    cited_sources.append(source)
                    cited_urls.add(url)

        if not referenced_chunk_numbers and not referenced_urls:
            print("  ⚠ WARNING: No citations found in response!")
            print(f"     Response length: {len(final_response)} chars")
            print(f"     Chunks provided: {len(chunks)}")
            cited_sources = []

        print(f"  Citations found: {sorted(referenced_chunk_numbers)}")
        print(f"  URLs cited: {list(referenced_urls)}")
        print(f"  Chunks provided: {len(chunks)}")
        print(f"  Sources returned: {len(cited_sources)}")
        if len(cited_sources) != len(referenced_chunk_numbers) + len(referenced_urls):
            print("  ⚠ Note: Some citations may reference the same source (deduplicated)")

        cited_chunk_nums = {chunk_num for chunk_num in referenced_chunk_numbers if chunk_num in chunk_to_source_map}
        cited_source_urls = {chunk_to_source_map[cn].url for cn in cited_chunk_nums}
        cited_source_urls.update(referenced_urls)

        final_cited_sources = [s for s in cited_sources if s.url in cited_source_urls]

        if len(final_cited_sources) != len(cited_sources):
            print(f"  ⚠ WARNING: Removed {len(cited_sources) - len(final_cited_sources)} uncited sources!")
            print(f"     Expected URLs: {cited_source_urls}")
            print(f"     Found URLs: {[s.url for s in cited_sources]}")
            cited_sources = final_cited_sources

        for source in cited_sources:
            if source.url not in cited_source_urls:
                print(f"  ⚠ ERROR: Source with URL {source.url} was not cited but included in results!")
                cited_sources = [s for s in cited_sources if s.url in cited_source_urls]
                break

        sources_used = [source.url for source in cited_sources]

        result = GeneratorOutput(
            response=final_response,
            has_sufficient_info=has_sufficient_info,
            sources_used=sources_used,
        )

        if not has_sufficient_info and not chunks:
            if "don't have sufficient information" not in final_response.lower():
                final_response = (
                    "I don't have sufficient information in my knowledge base to answer this question accurately. "
                    "You may want to:\n- Rephrase your question with more specific terms\n- Ask about a different aspect of diabetes management\n- Consult the full clinical guidelines directly"
                )

        result.has_sufficient_info = has_sufficient_info
        result.response = final_response

        state["generator_output"] = result
        state["sources"] = cited_sources
        state["final_response"] = final_response
        state["messages"] = state.get("messages", []) + [AIMessage(content=final_response)]

        if writer:
            writer({"type": "generator_complete", "message": f"Answer generated: {len(final_response)} chars"})

        print(f"✓ Generated response: {len(final_response)} chars")
        print(f"  Sufficient info: {result.has_sufficient_info}")
        print(f"  Retrieved chunks: {len(chunks)}")
        print(f"  Cited sources: {len(cited_sources)}")

        return state
    except Exception as exc:
        error_msg = f"Error in generator node: {str(exc)[:200]}"
        print(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        final_response = (
            f"I encountered an error while generating the response: {str(exc)[:200]}. Please try rephrasing your question."
        )
        state["final_response"] = final_response
        state["messages"] = state.get("messages", []) + [AIMessage(content=final_response)]
        if writer:
            writer({"type": "generator_error", "message": error_msg})
    return state


def route_after_classifier(state: ChatState) -> str:
    """Route based on classifier decision."""
    classifier_output = state.get("classifier_output")
    if classifier_output and classifier_output.route == "retrieve":
        return "retrieval"
    return END


def build_graph(reader) -> Any:
    """
    Build and compile the optimized LangGraph workflow.
    """
    set_chroma_reader(reader)

    workflow = StateGraph(ChatState)
    workflow.add_node("classifier", classify_query_unified)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generator", generator_node)

    workflow.set_entry_point("classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "retrieval": "retrieval",
            END: END,
        },
    )

    workflow.add_edge("retrieval", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()
