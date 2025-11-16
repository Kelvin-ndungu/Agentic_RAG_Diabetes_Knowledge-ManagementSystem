"""
Graph node functions for LangGraph workflow.
"""
import json
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from .models import (
    QuerySafetyClassification,
    Source,
    FollowUpAnalysis,
    ChatState,
    ClassifierOutput,
    GeneratorOutput
)
from .llm_setup import get_llm


# Global references (will be initialized in graph_builder)
chroma_reader = None


def set_chroma_reader(reader):
    """Set the global chroma_reader instance."""
    global chroma_reader
    chroma_reader = reader


def create_structured_chain(prompt_template: str, model_class: type[BaseModel], system_message: str = None, **template_vars):
    """
    Create a chain that produces structured output using Pydantic model.
    
    This uses LangChain's PydanticOutputParser which is the recommended approach
    for models that don't support native structured output.
    
    Args:
        prompt_template: The prompt template with placeholders like {query}, etc.
        model_class: Pydantic model class for structured output
        system_message: Optional system message
        **template_vars: Additional variables to pre-fill in the template
        
    Returns:
        Runnable chain that returns a Pydantic model instance
    """
    llm = get_llm()
    
    # Create PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=model_class)
    
    # Get format instructions and escape curly braces
    format_instructions = parser.get_format_instructions()
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    # Build the full prompt template
    full_prompt_template = prompt_template + "\n\n" + escaped_format_instructions
    
    # Build prompt messages
    if system_message:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", full_prompt_template)
        ])
        if template_vars:
            prompt = prompt.partial(**template_vars)
    else:
        prompt = ChatPromptTemplate.from_template(full_prompt_template)
        if template_vars:
            prompt = prompt.partial(**template_vars)
    
    # Create chain
    chain = prompt | llm | StrOutputParser() | parser
    
    return chain


def classify_query_unified(state: ChatState) -> ChatState:
    """
    Single LLM call handles all classification logic:
    - Greetings
    - Questions about system
    - Intent understanding/rephrasing
    - Relevance check
    - Safety check
    - Routing decision
    
    Returns:
        Updated state with classifier_output stored in state["classifier_output"]
    """
    writer = get_stream_writer()
    
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the classification system for a Diabetes Knowledge Management Assistant based on Kenya National Clinical Guidelines.

Your job is to analyze user queries and determine the appropriate response path.

## Query Types & Responses:

1. **GREETING** (e.g., "Hi", "Hello", "Hey there")
   - query_type: "greeting"
   - direct_response: "Hello! I'm a diabetes knowledge assistant based on Kenya National Clinical Guidelines for Diabetes Management. I can help healthcare providers with questions about diabetes diagnosis, treatment, management, and prevention. How can I assist you today?"
   - is_relevant: False
   - is_safe: True
   - should_generate: False
   - status_message: "Processing greeting..."

2. **ABOUT_SYSTEM** (e.g., "What can you do?", "How do you work?", "What are you?")
   - query_type: "about_system"
   - direct_response: "I'm a specialized AI assistant for healthcare providers focused on diabetes management. I provide information based on the Kenya National Clinical Guidelines for the Management of Diabetes. I can answer questions about:\\n\\n- Diabetes diagnosis and screening\\n- Treatment options and medications\\n- Management strategies\\n- Complications and prevention\\n- Clinical protocols\\n\\nI cannot provide patient-specific medical advice. How can I help with your diabetes-related question?"
   - is_relevant: False
   - is_safe: True
   - should_generate: False
   - status_message: "Explaining system capabilities..."

3. **IRRELEVANT** (Not about diabetes)
   - query_type: "irrelevant"
   - is_relevant: False
   - is_safe: True
   - direct_response: "I'm sorry, but I'm specifically designed to answer questions about diabetes management based on the Kenya National Clinical Guidelines. Your query doesn't appear to be related to diabetes. Please ask me about diabetes diagnosis, treatment, management, or prevention."
   - should_generate: False
   - status_message: "Analyzing query relevance..."

4. **UNSAFE** (Patient-specific medical advice, diagnoses, prognoses)
   - query_type: "unsafe"
   - is_relevant: True
   - is_safe: False
   - direct_response: "I cannot provide patient-specific medical advice, diagnoses, or treatment recommendations. This type of question requires a healthcare provider who can evaluate the full clinical context and provide personalized guidance.\\n\\nI can help with general questions about diabetes management guidelines and protocols. Would you like to rephrase your question in a more general way?"
   - should_generate: False
   - status_message: "Evaluating query safety..."

5. **SUBSTANTIVE** (Safe diabetes questions)
   - query_type: "substantive"
   - is_relevant: True
   - is_safe: True
   - intent: <Rephrase query with full context for retrieval>
   - direct_response: None
   - should_generate: True
   - status_message: "Understanding your query and searching knowledge base..."

## Intent Rephrasing (for SUBSTANTIVE queries only):
- If follow-up question, incorporate context from conversation history
- If standalone question, ensure it's clear and complete
- Make it suitable for semantic search - use natural medical/clinical language
- DO NOT add phrases like "guidelines", "Kenya National Clinical Guidelines", "according to guidelines" - the knowledge base contains clinical content, not meta-references
- Focus on the actual medical/clinical concepts and terms
- Example: User says "What about that?" after discussing Type 2 diabetes → intent: "What are the management approaches for Type 2 diabetes?"
- Example: User says "How is diabetes diagnosed?" → intent: "What are the diagnostic criteria and screening procedures for diabetes?"
- Example: User says "What about treatment?" → intent: "What are the treatment options for Type 2 diabetes?"

## Safety Examples:
**UNSAFE:**
- "My patient has diabetes and blood pressure, will they die?"
- "Should I stop taking my insulin?"
- "What dose of metformin should I give this patient?"

**SAFE:**
- "What are the general treatment options for Type 2 diabetes?"
- "What are the guidelines for insulin therapy initiation?"
- "What are the recommended HbA1c targets in diabetes management?"

## Important:
- status_message should be user-friendly and informative
- direct_response should be complete, helpful, and professional
- intent should be self-contained and context-aware for retrieval
- Always be polite and helpful even when declining

## Output Format:
You MUST respond with a valid JSON object with the following structure:
{{
  "query_type": "greeting|about_system|substantive|irrelevant|unsafe",
  "is_relevant": true/false,
  "is_safe": true/false,
  "should_generate": true/false,
  "status_message": "Status message for user",
  "intent": "Rephrased query (only for substantive queries, null otherwise)",
  "direct_response": "Complete response (only for non-substantive queries, null otherwise)"
}}

Return ONLY valid JSON, no markdown formatting or additional text."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Helper function to parse output (JSON or markdown)
    def parse_classifier_output(text: str) -> ClassifierOutput:
        """Parse classifier output - handles both JSON and markdown formats"""
        # First, try to extract JSON from the text
        json_match = re.search(r'\{[^{}]*"query_type"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return ClassifierOutput(**data)
            except:
                pass
        
        # Try to parse entire text as JSON
        try:
            data = json.loads(text.strip())
            return ClassifierOutput(**data)
        except:
            pass
        
        # Fallback: parse markdown-formatted output
        query_type_match = re.search(r'\*\*query_type\*\*:\s*(\w+)', text)
        is_relevant_match = re.search(r'\*\*is_relevant\*\*:\s*(True|False)', text)
        is_safe_match = re.search(r'\*\*is_safe\*\*:\s*(True|False)', text)
        should_generate_match = re.search(r'\*\*should_generate\*\*:\s*(True|False)', text)
        status_message_match = re.search(r'\*\*status_message\*\*:\s*(.+?)(?=\n\*\*|\Z)', text, re.DOTALL)
        intent_match = re.search(r'\*\*intent\*\*:\s*(.+?)(?=\n\*\*|\Z)', text, re.DOTALL)
        direct_response_match = re.search(r'\*\*direct_response\*\*:\s*(.+?)(?=\n\*\*|\Z)', text, re.DOTALL)
        
        # Parse values
        query_type = query_type_match.group(1) if query_type_match else "substantive"
        is_relevant = is_relevant_match.group(1) == "True" if is_relevant_match else True
        is_safe = is_safe_match.group(1) == "True" if is_safe_match else True
        should_generate = should_generate_match.group(1) == "True" if should_generate_match else True
        status_message = status_message_match.group(1).strip() if status_message_match else "Processing query..."
        intent = intent_match.group(1).strip() if intent_match else None
        direct_response = direct_response_match.group(1).strip() if direct_response_match else None
        
        return ClassifierOutput(
            query_type=query_type,
            is_relevant=is_relevant,
            is_safe=is_safe,
            should_generate=should_generate,
            status_message=status_message,
            intent=intent,
            direct_response=direct_response
        )
    
    try:
        # Use direct LLM call with robust parsing (Claude works well with JSON)
        llm = get_llm()
        chain = classifier_prompt | llm | StrOutputParser()
        raw_output = chain.invoke({"messages": state.get("messages", [])})
        result = parse_classifier_output(raw_output)
        
        # Store in state
        state["classifier_output"] = result
        
        # Set final response for non-substantive queries
        if not result.should_generate:
            state["final_response"] = result.direct_response
            # Add response to messages for consistency
            if result.direct_response:
                state["messages"] = state.get("messages", []) + [AIMessage(content=result.direct_response)]
        
        # Stream status update with intent for substantive queries
        if writer:
            if result.should_generate and result.intent:
                # For substantive queries, include intent in status
                status_msg = f"I am getting the relevant resources to answer: {result.intent}"
            else:
                status_msg = result.status_message
            writer({"type": "classifier_status", "message": status_msg})
        
        print(f"✓ Classified as: {result.query_type}")
        print(f"  Relevant={result.is_relevant}, Safe={result.is_safe}, Generate={result.should_generate}")
        if result.intent:
            print(f"  Intent: {result.intent[:80]}...")
            
    except Exception as e:
        print(f"⚠ Classification error: {e}")
        import traceback
        traceback.print_exc()
        # Create fallback classification
        result = ClassifierOutput(
            query_type="irrelevant",
            is_relevant=False,
            is_safe=True,
            should_generate=False,
            status_message="Processing query...",
            intent=None,
            direct_response="I encountered an error while processing your query. Please try again."
        )
        state["classifier_output"] = result
        state["final_response"] = result.direct_response
    
    return state


def retrieval_node(state: ChatState) -> ChatState:
    """
    Programmatic retrieval based on classifier intent.
    No LLM calls - pure Python logic.
    """
    classifier_output = state.get("classifier_output")
    writer = get_stream_writer()
    
    if not classifier_output or not classifier_output.should_generate:
        # Skip retrieval for non-substantive queries
        return state
    
    intent = classifier_output.intent
    
    if not intent:
        print("⚠ No intent available for retrieval")
        return state
    
    # Always do fresh retrieval with rephrased intent
    try:
        chunks = chroma_reader.search(
            query=intent,
            n_results=5,
            min_similarity=0.4
        )
        
        # Update state
        state["retrieved_chunks"] = chunks
        
        # Stream retrieval status
        if writer:
            if chunks:
                writer({"type": "retrieval_status", "message": f"Found {len(chunks)} relevant sources. Generating answer..."})
            else:
                writer({"type": "retrieval_status", "message": "No sources found with sufficient relevance. Responding..."})
        
        print(f"✓ Retrieved {len(chunks)} chunks (similarity >= 0.4)")
        if chunks:
            print(f"  Top relevance: {chunks[0]['relevance_score']:.3f}")
    except Exception as e:
        print(f"⚠ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        state["retrieved_chunks"] = []
        if writer:
            writer({"type": "retrieval_error", "message": "Error during retrieval. Continuing..."})
    
    return state


def route_classifier(state: ChatState) -> str:
    """Route based on classification results."""
    classification = state.get("classification")
    if not classification:
        return "not_relevant"
    
    # First check relevance
    if not classification.is_relevant:
        return "not_relevant"
    
    # Check if unsafe
    if classification.is_safe is False or classification.risk_level in ["medium", "high"]:
        return "unsafe"
    
    # Relevant and safe
    return "generator"


def analyze_intent(state: ChatState) -> ChatState:
    """
    Use LLM to determine if the current message is a follow-up to previous conversation.
    Analyzes the conversation history to understand context.
    """
    messages = state.get("messages", [])
    if len(messages) < 2:
        # First message, not a follow-up
        state["is_followup"] = False
        return state
    
    # Get conversation history (last few messages for context)
    recent_messages = messages[-5:]  # Last 5 messages for context
    conversation_text = ""
    for msg in recent_messages[:-1]:  # Exclude the last message (current one)
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        conversation_text += f"{role}: {content}\n"
    
    current_message = recent_messages[-1]
    current_content = current_message.content if hasattr(current_message, 'content') else str(current_message)
    
    intent_prompt = """Analyze if the current user message is a follow-up question to the previous conversation, or if it's a new, independent query.

Previous Conversation:
{conversation_history}

Current Message: "{current_message}"

A follow-up message:
- References or builds upon previous conversation
- Asks for clarification, more details, or related information
- Uses pronouns or references that relate to previous context (e.g., "What about that?", "Tell me more", "How about...")
- Is a continuation of the same topic

A new message:
- Introduces a completely new topic
- Is independent and doesn't reference previous conversation
- Can be understood without context from previous messages

Respond with structured output."""
    
    try:
        intent_chain = create_structured_chain(
            intent_prompt,
            FollowUpAnalysis,
            system_message="You are an expert at analyzing conversation flow and intent.",
            conversation_history=conversation_text if conversation_text else "No previous conversation.",
            current_message=current_content
        )
        
        intent_result: FollowUpAnalysis = intent_chain.invoke({})
        state["is_followup"] = intent_result.is_followup
        
        print(f"✓ Intent analyzed: Follow-up={intent_result.is_followup}")
        if intent_result.reasoning:
            print(f"  Reasoning: {intent_result.reasoning[:100]}...")
            
    except Exception as e:
        print(f"⚠ Intent analysis error: {e}")
        # Default to False (new message) on error
        state["is_followup"] = False
    
    return state


@tool
def search_semantic_only(
    query: str,
    n_results: int = 5,
    min_similarity: float = 0.4,
    runtime: ToolRuntime = None
) -> List[Dict[str, Any]]:
    """
    Search using semantic similarity only. Use when you need to retrieve information from the knowledge base.
    Only chunks with relevance_score >= min_similarity (0.4) are returned.
    
    Args:
        query: Search query text
        n_results: Number of results to return (default: 5)
        min_similarity: Minimum relevance score threshold (0-1, default: 0.4)
    
    Returns:
        List of chunk dictionaries with content, metadata, and relevance_score
    """
    if runtime and runtime.stream_writer:
        runtime.stream_writer({"type": "tool_progress", "message": f"Semantic search: {query[:50]}..."})
    
    try:
        if chroma_reader is None:
            raise ValueError("ChromaDB reader not initialized")
        
        chunks = chroma_reader.search(
            query=query,
            n_results=n_results,
            min_similarity=min_similarity,
            where=None  # No metadata filtering
        )
        
        if runtime and runtime.stream_writer:
            runtime.stream_writer({"type": "tool_progress", "message": f"Found {len(chunks)} chunks via semantic search"})
        
        return chunks
    except Exception as e:
        error_msg = f"Semantic search failed: {str(e)}"
        if runtime and runtime.stream_writer:
            runtime.stream_writer({"type": "tool_error", "message": error_msg})
        return [{"error": error_msg, "chunk_id": None, "content": "", "metadata": {}, "relevance_score": 0.0}]


def not_relevant_response(state: ChatState) -> ChatState:
    """Create response for non-relevant queries."""
    classification = state.get("classification")
    reasoning = classification.reasoning if classification else "The query is not related to diabetes management."
    
    response = """I'm a diabetes specialist assistant. I can only provide information about diabetes management, treatment, diagnosis, prevention, and related healthcare topics based on the Kenya National Clinical Guidelines for the Management of Diabetes.

Your query doesn't appear to be related to diabetes."""
    
    state["messages"] = state.get("messages", []) + [AIMessage(content=response)]
    return state


def unsafe_response(state: ChatState) -> ChatState:
    """Create response for unsafe/high-risk queries."""
    classification = state.get("classification")
    reasoning = classification.reasoning if classification else "The query poses a risk of harm if answered."
    risk_level = classification.risk_level if classification else "high"
    
    response = f"""I cannot answer this question as it poses a {risk_level} risk of harm.

{reasoning}

For patient-specific medical advice, please consult with a healthcare provider who can evaluate the full clinical context and provide personalized guidance."""
    
    state["messages"] = state.get("messages", []) + [AIMessage(content=response)]
    return state


def generator_node(state: ChatState) -> ChatState:
    """
    Single LLM call for generation with conversation history.
    Handles both cases: with chunks and without chunks.
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
        
        # Build context from chunks
        # IMPORTANT: Number chunks sequentially (1, 2, 3...) and track chunk-to-source mapping
        if chunks:
            context_parts = []
            chunk_to_source_map = {}  # Maps chunk index (1-based) to Source object
            seen_urls = {}  # Maps URL to first Source object with that URL
            
            for i, chunk in enumerate(chunks, 1):  # Start numbering from 1
                metadata = chunk.get("metadata", {})
                title = metadata.get("title", "Unknown")
                url = metadata.get("url", "")
                content = chunk.get("content", "")
                relevance = chunk.get("relevance_score", 0)
                
                # Format context clearly with source info
                context_parts.append(f"--- Source {i}: {title} (Relevance: {relevance:.2f}) ---\nURL: {url}\n\n{content}")
                
                # Create source for this chunk
                source = Source(
                    title=title,
                    url=url,
                    chunk_id=chunk.get("chunk_id", "")
                )
                
                # Map this chunk index to its source
                chunk_to_source_map[i] = source
                
                # Track first occurrence of each URL for deduplication
                if url and url not in seen_urls:
                    seen_urls[url] = source
            
            context = "\n\n".join(context_parts)
            has_context = True
        else:
            context = "No relevant information found in knowledge base."
            chunk_to_source_map = {}
            seen_urls = {}
            has_context = False
        
        # Build generator prompt with conversation history
        generator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a diabetes specialist assistant for healthcare providers, based on Kenya National Clinical Guidelines for Diabetes Management.

## Your Task:
Answer the user's query using the provided context from the knowledge base.

## CRITICAL GUARDRAILS:
1. **USE the provided context** - The context contains relevant clinical information. If context is provided, you MUST use it to answer the question
2. **Be factual and truthful** - Base your answer on the context provided
3. **No personalized medical advice** - Provide general clinical information only
4. **Use numbered citations** - Format: [1], [2], [3] etc. when referencing specific information from the context
5. **DO NOT add a Sources section** - The frontend will handle displaying sources automatically
6. **Be comprehensive** - Extract and use ALL relevant information from the context to provide a thorough answer

## Critical Instructions:
- **If context is provided above, you MUST answer the question** - Do NOT say "insufficient information"
- The context contains relevant clinical information that was retrieved specifically for this query
- Extract and synthesize information from ALL provided context chunks
- Only mention "insufficient information" if the context section explicitly says "No relevant information found"
- When context is provided, your job is to answer comprehensively using that information
- **Use numbered citations [1], [2], [3] etc.** - Reference sources by their number from the available sources list
- **Only cite sources you actually use** - If you mention information from Source 1, use [1]. If from Source 2, use [2], etc.
- **DO NOT include a "## Sources" section at the end** - The system will automatically extract and display only the sources you actually cited
- **DO NOT use [Title](url) format** - Use only numbered references like [1], [2], [3]

## Response Format:
- Clear, clinical language for healthcare providers
- Numbered citations throughout: [1], [2], [3] when referencing specific information
- Structured with headers if appropriate
- NO Sources section - just use numbered citations
- Be comprehensive - use all relevant information from the context"""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", """User Query: {intent}

Relevant Information from Knowledge Base:
{context}

IMPORTANT CITATION INSTRUCTIONS:
- Each chunk in the context above is labeled as "Source 1", "Source 2", "Source 3", etc.
- When you reference information from a chunk, cite it using its Source number: [1], [2], [3], etc.
- For example, if you use information from "Source 1", cite it as [1]
- If you use information from "Source 2", cite it as [2]
- Only cite sources (chunks) that you actually use in your answer
- Do NOT cite sources you don't use

CRITICAL: 
- The context above contains relevant clinical information. Use this information to provide a comprehensive answer to the user's query.
- Include specific details, recommendations, and numbered citations [1], [2], [3] etc. when referencing information from specific chunks.
- Only cite chunks you actually use in your answer.

Provide your answer following all guardrails above.""")
        ])
        
        # Use direct LLM call (Claude works well with structured prompts)
        # Determine has_sufficient_info programmatically based on chunks
        has_sufficient_info = len(chunks) > 0 and any(chunk.get('relevance_score', 0) >= 0.4 for chunk in chunks)
        
        # Build chain for streaming
        llm = get_llm()
        chain = generator_prompt | llm | StrOutputParser()
        
        # Stream the response token by token
        final_response = ""
        if writer:
            writer({"type": "generator_start", "message": "Generating answer..."})
        
        # Stream tokens as they're generated
        for chunk in chain.stream({
            "messages": state.get("messages", []),
            "intent": intent,
            "context": context
        }):
            # chunk is a string token
            if chunk:
                final_response += chunk
                # Stream token to frontend via writer
                if writer:
                    writer({"type": "token", "content": chunk})
        
        # If no streaming happened (fallback), use invoke
        if not final_response:
            response = chain.invoke({
                "messages": state.get("messages", []),
                "intent": intent,
                "context": context
            })
            final_response = response if isinstance(response, str) else response.content
            # Stream the full response if we got it all at once
            if writer and final_response:
                writer({"type": "token", "content": final_response})
        
        # Remove any "## Sources" section that the LLM might have added
        # Split by "## Sources" and take only the content before it
        if "## Sources" in final_response:
            final_response = final_response.split("## Sources")[0].strip()
        
        # Extract only sources that are actually referenced in the response
        # Citations refer to chunk numbers (1, 2, 3...) from the context
        import re
        referenced_chunk_numbers = set()  # Chunk numbers cited (1-based)
        
        # Validate chunk numbers are within valid range
        max_chunk_num = len(chunks) if chunks else 0
        
        # Pattern to match numbered citations like [1], [2], [10] but not [Title](url)
        # Match [number] where number is digits, but not followed by (
        citation_pattern = r'\[(\d+)\](?!\()'
        matches = re.findall(citation_pattern, final_response)
        for num_str in matches:
            try:
                chunk_num = int(num_str)  # This is 1-based chunk number from context
                # Validate: chunk number must be within valid range (1 to max_chunk_num)
                if 1 <= chunk_num <= max_chunk_num and chunk_num in chunk_to_source_map:
                    referenced_chunk_numbers.add(chunk_num)
                else:
                    # Log invalid citation for debugging
                    print(f"  ⚠ Invalid citation [{chunk_num}] - out of range (valid: 1-{max_chunk_num})")
            except ValueError:
                pass
        
        # Also check for [Title](url) format as fallback (in case LLM uses old format)
        referenced_urls = set()
        url_citation_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        url_matches = re.findall(url_citation_pattern, final_response)
        for title, url in url_matches:
            referenced_urls.add(url)
        
        # Extract sources ONLY from chunks that were actually cited
        # This ensures we only return sources for chunks that were used
        # NO FALLBACK - if no citations found, return empty list
        cited_sources = []
        cited_urls = set()  # Track URLs to avoid duplicates
        
        # First, get sources from cited chunk numbers (in order of citation)
        for chunk_num in sorted(referenced_chunk_numbers):
            if chunk_num in chunk_to_source_map:
                source = chunk_to_source_map[chunk_num]
                # Only add if we haven't seen this URL yet (preserve first occurrence order)
                if source.url not in cited_urls:
                    cited_sources.append(source)
                    cited_urls.add(source.url)
        
        # Also include sources referenced by URL (fallback for markdown link format)
        # But only if they were actually cited in the response
        for url in referenced_urls:
            if url not in cited_urls:
                # Find source with this URL from seen_urls
                if url in seen_urls:
                    source = seen_urls[url]
                    cited_sources.append(source)
                    cited_urls.add(url)
        
        # CRITICAL: If no citations found at all, log warning but return empty sources
        # We should NOT return all chunks if none were cited
        if not referenced_chunk_numbers and not referenced_urls:
            print(f"  ⚠ WARNING: No citations found in response!")
            print(f"     Response length: {len(final_response)} chars")
            print(f"     Chunks provided: {len(chunks)}")
            print(f"     This means the LLM used information without citing it properly")
            # Return empty sources - we can't cite what wasn't cited
            cited_sources = []
        
        # Log for debugging
        print(f"  Citations found: {sorted(referenced_chunk_numbers)}")
        print(f"  URLs cited: {list(referenced_urls)}")
        print(f"  Chunks provided: {len(chunks)}")
        print(f"  Sources returned: {len(cited_sources)}")
        if len(cited_sources) != len(referenced_chunk_numbers) + len(referenced_urls):
            print(f"  ⚠ Note: Some citations may reference the same source (deduplicated)")
        
        # CRITICAL: Validate and filter cited_sources to ensure only actually cited sources are included
        # Verify each source in cited_sources was actually cited
        cited_chunk_nums = {chunk_num for chunk_num in referenced_chunk_numbers if chunk_num in chunk_to_source_map}
        cited_source_urls = {chunk_to_source_map[cn].url for cn in cited_chunk_nums}
        cited_source_urls.update(referenced_urls)
        
        # Remove any sources that weren't actually cited
        final_cited_sources = [s for s in cited_sources if s.url in cited_source_urls]
        
        if len(final_cited_sources) != len(cited_sources):
            print(f"  ⚠ WARNING: Removed {len(cited_sources) - len(final_cited_sources)} uncited sources!")
            print(f"     Expected URLs: {cited_source_urls}")
            print(f"     Found URLs: {[s.url for s in cited_sources]}")
            cited_sources = final_cited_sources
        
        # Final validation: ensure all sources in cited_sources were actually cited
        for source in cited_sources:
            if source.url not in cited_source_urls:
                print(f"  ⚠ ERROR: Source with URL {source.url} was not cited but included in results!")
                cited_sources = [s for s in cited_sources if s.url in cited_source_urls]
                break
        
        # Extract source URLs from validated cited sources
        sources_used = [source.url for source in cited_sources]
        
        # Create GeneratorOutput object
        result = GeneratorOutput(
            response=final_response,
            has_sufficient_info=has_sufficient_info,
            sources_used=sources_used
        )
        
        # If no chunks found, add insufficient info message
        if not has_sufficient_info and not chunks:
            if "don't have sufficient information" not in final_response.lower():
                final_response = "I don't have sufficient information in my knowledge base to answer this question accurately. You may want to:\n- Rephrase your question with more specific terms\n- Ask about a different aspect of diabetes management\n- Consult the full clinical guidelines directly"
        
        # Update result with correct has_sufficient_info
        result.has_sufficient_info = has_sufficient_info
        result.response = final_response
        
        # Store citation mapping in a way frontend can use
        state["generator_output"] = result
        state["sources"] = cited_sources  # Only cited sources, not all retrieved
        state["final_response"] = final_response
        # Store citation mapping in a way frontend can use
        # Frontend will receive sources array and can map [1] to sources[0] if citation_map[1] exists
        # For now, we'll rely on the frontend to handle the mapping based on citation numbers in text
        
        # Add response to messages
        state["messages"] = state.get("messages", []) + [AIMessage(content=final_response)]
        
        if writer:
            writer({"type": "generator_complete", "message": f"Answer generated: {len(final_response)} chars"})
        
        print(f"✓ Generated response: {len(final_response)} chars")
        print(f"  Sufficient info: {result.has_sufficient_info}")
        print(f"  Retrieved chunks: {len(chunks)}")
        print(f"  Cited sources: {len(cited_sources)}")
        
        return state
        
    except Exception as e:
        error_msg = f"Error in generator node: {str(e)[:200]}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        final_response = f"I encountered an error while generating the response: {str(e)[:200]}. Please try rephrasing your question."
        state["final_response"] = final_response
        state["messages"] = state.get("messages", []) + [AIMessage(content=final_response)]
        if writer:
            writer({"type": "generator_error", "message": error_msg})
    return state

