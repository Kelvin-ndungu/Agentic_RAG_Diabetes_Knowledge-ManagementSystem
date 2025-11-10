"""
Graph node functions for LangGraph workflow.
"""
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from .models import (
    QuerySafetyClassification,
    Source,
    FollowUpAnalysis,
    ChatState
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


def classify_query(state: ChatState) -> ChatState:
    """
    Classify query in two steps:
    1. First check if relevant to diabetes
    2. Only if relevant, check if safe to answer
    
    Extracts the last user message from state["messages"].
    
    Returns:
        Updated state with classification stored in state["classification"]
    """
    # Extract last user message
    messages = state.get("messages", [])
    if not messages:
        state["classification"] = QuerySafetyClassification(
            is_relevant=False,
            is_safe=False,
            risk_level="none",
            reasoning="No messages found in state"
        )
        return state
    
    # Get the last user message
    last_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    
    if not last_message:
        state["classification"] = QuerySafetyClassification(
            is_relevant=False,
            is_safe=False,
            risk_level="none",
            reasoning="No user message found"
        )
        return state
    
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    writer = get_stream_writer()
    
    # Step 1: Check relevance first
    relevance_prompt = """Classify if this query is relevant to diabetes management, treatment, diagnosis, prevention, or related healthcare topics.

Query: "{query}"

Determine if the query is about:
- Diabetes diagnosis and symptoms
- Treatment options (medications, insulin therapy, lifestyle changes)
- Diabetes management during pregnancy
- Hypoglycemia and hyperglycemia
- Blood glucose monitoring
- Nutrition and diabetes
- Diabetes complications and prevention
- Other diabetes-related topics

If the query is NOT about diabetes (e.g., weather, general health, other diseases), it is not relevant.

Respond with structured output."""
    
    relevance_system = """You are a diabetes expert classifier. First check if the query is relevant to diabetes topics."""
    
    try:
        # Create a simple relevance model
        class RelevanceClassification(BaseModel):
            is_relevant: bool = Field(description="Whether the query is relevant to diabetes management and care")
            reasoning: str = Field(description="Brief explanation")
        
        relevance_chain = create_structured_chain(
            relevance_prompt,
            RelevanceClassification,
            relevance_system,
            query=query
        )
        
        relevance_result: RelevanceClassification = relevance_chain.invoke({})
        
        if writer:
            writer({"type": "classification_step", "message": f"Relevance check: {relevance_result.is_relevant}"})
        
        # If not relevant, return immediately (no safety check needed)
        if not relevance_result.is_relevant:
            classification_result = QuerySafetyClassification(
                is_relevant=False,
                is_safe=False,
                risk_level="none",
                reasoning=relevance_result.reasoning
            )
            state["classification"] = classification_result
            print(f"✓ Query classified: Not relevant to diabetes")
            if classification_result.reasoning:
                print(f"  Reasoning: {classification_result.reasoning[:100]}...")
            return state
        
        # Step 2: Only if relevant, check safety
        class SafetyClassification(BaseModel):
            is_safe: bool = Field(description="Whether the query is safe to answer (must be true or false)")
            risk_level: Literal["none", "low", "medium", "high"] = Field(description="Risk level: 'none' or 'low' for safe, 'medium' or 'high' for unsafe")
            reasoning: str = Field(description="Brief explanation for the safety assessment")
        
        safety_prompt = """Classify if this RELEVANT diabetes query is safe to answer without risk of harm.

Query: "{query}"

This query is already confirmed to be relevant to diabetes. Now assess safety.

Unsafe examples (high risk):
- "My patient has diabetes and blood pressure, will they die?"
- Questions requiring personalized medical diagnosis
- Questions about specific patient outcomes or prognoses
- Questions asking for medical advice for a specific patient

Safe examples (low/no risk):
- "How is type 2 diabetes diagnosed?"
- "What are the treatment options for diabetes?"
- "What are the guidelines for insulin therapy?"
- General questions about diabetes management

You MUST provide:
- is_safe: true or false (boolean, required)
- risk_level: "none" (safe), "low" (mostly safe), "medium" (risky), or "high" (unsafe)

Respond with structured output."""
        
        safety_system = """You are a diabetes expert classifier. Assess safety/risk level for relevant diabetes queries. You must provide is_safe as a boolean (true or false)."""
        
        safety_chain = create_structured_chain(
            safety_prompt,
            SafetyClassification,
            safety_system,
            query=query
        )
        
        safety_result: SafetyClassification = safety_chain.invoke({})
        
        # Convert to QuerySafetyClassification
        classification_result = QuerySafetyClassification(
            is_relevant=True,
            is_safe=safety_result.is_safe,
            risk_level=safety_result.risk_level,
            reasoning=safety_result.reasoning
        )
        
        state["classification"] = classification_result
        
        if writer:
            writer({"type": "classification_complete", "message": f"Relevant={classification_result.is_relevant}, Safe={classification_result.is_safe}, Risk={classification_result.risk_level}"})
        
        print(f"✓ Query classified: Relevant={classification_result.is_relevant}, Safe={classification_result.is_safe}, Risk={classification_result.risk_level}")
        if classification_result.reasoning:
            print(f"  Reasoning: {classification_result.reasoning[:100]}...")
            
    except Exception as e:
        print(f"⚠ Classification error: {e}")
        state["classification"] = QuerySafetyClassification(
            is_relevant=False,
            is_safe=False,
            risk_level="none",
            reasoning=f"Classification failed: {str(e)}"
        )
    
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
    Generator node that:
    1. Analyzes intent (follow-up vs new)
    2. Uses agent to decide retrieval
    3. Makes single retrieval call if needed
    4. Generates answer with inline citations and sources section
    """
    writer = get_stream_writer()
    messages = state.get("messages", [])
    
    # Get last user message
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg
            break
    
    if not last_user_msg:
        state["messages"] = state.get("messages", []) + [AIMessage(content="Error: No user message found.")]
        return state
    
    query = last_user_msg.content if hasattr(last_user_msg, 'content') else str(last_user_msg)
    
    # Step 1: Analyze intent
    state = analyze_intent(state)
    is_followup = state.get("is_followup", False)
    existing_chunks = state.get("retrieved_chunks", [])
    
    if writer:
        writer({"type": "generator_start", "message": f"Generator node: Follow-up={is_followup}, Existing chunks={len(existing_chunks)}"})
    
    # Step 2: Create agent with retrieval tool
    agent_tools = [search_semantic_only]
    
    # Build conversation context for agent
    conversation_context = ""
    if len(messages) > 1:
        for msg in messages[:-1]:  # Exclude last message
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            conversation_context += f"{role}: {content}\n"
    
    system_prompt = f"""You are a diabetes specialist assistant helping doctors make informed decisions based on the Kenya National Clinical Guidelines for the Management of Diabetes.

## Your Task

You need to decide whether to retrieve information from the knowledge base to answer the user's query.

## Available Tool

**search_semantic_only**: Search the knowledge base using semantic similarity
   - Use this tool if you need to retrieve information to answer the query
   - Only chunks with similarity > 0.4 are returned
   - You can make ONE retrieval call per query

## Decision Guidelines

- **Retrieve if**: You need information from the knowledge base to answer accurately
- **Don't retrieve if**: You can answer from existing context or it's a simple follow-up that doesn't need new information

## Important Rules

1. You can make ONLY ONE retrieval call per query
2. After retrieving (or deciding not to retrieve), generate a comprehensive answer
3. Your answer must be:
   - Factual: Based only on information from retrieved chunks or existing context
   - Truthful: Do not make up information
   - Safe: Do not provide personalized medical advice
4. Use numbered references in the text instead of inline citations
   - When referencing information, use format: [1], [2], [3], etc.
   - These numbers correspond to the sources listed at the end
   - DO NOT use inline links like [Title](url) in the body text
5. At the end, add a "## Sources" section with numbered list of all sources used
   - Format: 1. [Title](url), 2. [Title](url), etc.
   - Use the EXACT URLs from the chunks (they start with /guidelines/)

## Current Context

{'This appears to be a follow-up question.' if is_followup else 'This appears to be a new question.'}
{'You have existing retrieved chunks available.' if existing_chunks else 'No existing chunks available.'}

{conversation_context if conversation_context else ''}

Now decide: Do you need to retrieve information? If yes, use search_semantic_only. Then generate your answer with inline citations and a sources section."""
    
    try:
        llm = get_llm()
        
        # Create agent
        agent = create_agent(
            llm,
            agent_tools,
            system_prompt=system_prompt
        )
        
        # Prepare agent input
        agent_input = {
            "messages": [HumanMessage(content=f"Query: {query}\n\nDecide if you need to retrieve information, then generate a comprehensive answer with citations.")]
        }
        
        if writer:
            writer({"type": "agent_start", "message": "Starting retrieval decision agent"})
        
        # Invoke agent
        agent_result = agent.invoke(agent_input)
        agent_messages = agent_result.get("messages", [])
        
        # Extract retrieved chunks from tool messages
        retrieved_chunks = []
        final_answer = None
        
        for msg in agent_messages:
            # Extract chunks from ToolMessage
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, str):
                    try:
                        chunks_data = json.loads(content)
                        if isinstance(chunks_data, list):
                            for chunk in chunks_data:
                                if isinstance(chunk, dict) and chunk.get("relevance_score", 0.0) >= 0.4:
                                    retrieved_chunks.append(chunk)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("relevance_score", 0.0) >= 0.4:
                            retrieved_chunks.append(chunk)
            
            # Extract final answer from AIMessage (when no tool calls)
            if isinstance(msg, AIMessage) and msg.content:
                if not msg.tool_calls or len(msg.tool_calls) == 0:
                    final_answer = msg.content
        
        # If agent retrieved chunks, replace old chunks
        if retrieved_chunks:
            state["retrieved_chunks"] = retrieved_chunks
            if writer:
                writer({"type": "retrieval_complete", "message": f"Retrieved {len(retrieved_chunks)} chunks"})
        
        # If no answer from agent but we have chunks, generate answer
        if not final_answer:
            # Use existing chunks or newly retrieved chunks
            chunks_to_use = retrieved_chunks if retrieved_chunks else existing_chunks
            
            if chunks_to_use:
                # Format chunks for generation
                context_parts = []
                sources_list = []
                seen_urls = set()
                
                for chunk in chunks_to_use:
                    metadata = chunk.get("metadata", {})
                    title = metadata.get("title", "Unknown")
                    url = metadata.get("url", "")
                    content = chunk.get("content", "")
                    
                    context_parts.append(f"[{title}]({url})\n{content}")
                    
                    # Collect unique sources
                    if url and url not in seen_urls:
                        sources_list.append(Source(
                            title=title,
                            url=url,
                            chunk_id=chunk.get("chunk_id", "")
                        ))
                        seen_urls.add(url)
                
                context = "\n\n".join(context_parts)
                
                # Build sources list with numbers for reference
                sources_with_numbers = []
                for i, source in enumerate(sources_list, 1):
                    sources_with_numbers.append(f"{i}. {source.title} - {source.url}")
                
                sources_text = "\n".join(sources_with_numbers)
                
                # Generate answer
                gen_prompt = """You are a diabetes specialist helping doctors make informed decisions. Answer based on the Kenya National Clinical Guidelines.

Query: "{query}"

Context from Knowledge Base:
{context}

Available Sources (use these for numbered references):
{sources_list}

{conversation_context}

Instructions:
1. Answer using ONLY information from the provided context
2. Be FACTUAL, TRUTHFUL, and SAFE
3. Explain the information in SIMPLE, CLEAR language that is easy to understand
   - Use plain language without medical jargon when possible
   - Break down complex concepts into understandable parts
   - DO NOT dilute or oversimplify the content - maintain accuracy and completeness
   - Balance simplicity with thoroughness
4. Use numbered references in the text instead of inline citations
   - When referencing information, use format: [1], [2], [3], etc.
   - These numbers correspond to the sources listed at the end
   - Example: "According to the guidelines [1], diabetes management requires..."
   - DO NOT use inline links like [Title](url) in the body text
5. Use clear, professional language appropriate for healthcare professionals
6. Include specific details and recommendations from the guidelines
7. Format your answer with proper markdown headings, lists, and paragraphs
8. At the end, add a "## Sources" section with numbered list of all sources used
   - Format: 1. [Title](url), 2. [Title](url), etc.
   - Use the EXACT URLs from the sources list above

Generate the answer:"""
                
                gen_chain = ChatPromptTemplate.from_template(gen_prompt) | llm | StrOutputParser()
                final_answer = gen_chain.invoke({
                    "query": query,
                    "context": context,
                    "sources_list": sources_text,
                    "conversation_context": f"\n\nConversation History:\n{conversation_context}" if conversation_context else ""
                })
                
                # Ensure sources section is present with actual URLs
                if "## Sources" not in final_answer and sources_list:
                    final_answer += "\n\n## Sources\n\n"
                    for i, source in enumerate(sources_list[:10], 1):
                        # Use the actual URL from the source (should be /guidelines/...)
                        source_url = source.url if source.url else "#"
                        final_answer += f"{i}. [{source.title}]({source_url})\n"
                
                # Remove any inline citations from the body (they should only be numbered references)
                # Replace inline link patterns with just the title, but preserve numbered references [1], [2], etc.
                import re
                # Pattern to match [Title](url) in the body (but keep Sources section and numbered references)
                def replace_inline_citations(text):
                    # Split by Sources section
                    parts = text.split("## Sources")
                    if len(parts) > 1:
                        body = parts[0]
                        sources_section = "## Sources" + parts[1]
                        # Replace [Title](url) with just Title in the body
                        # But preserve numbered references like [1](url), [2](url) - these should be kept
                        # Pattern: [text](url) where text contains non-digit characters (i.e., is not just digits)
                        # This will match [Title](url) but not [1](url) or [123](url)
                        body = re.sub(r'\[([^\]]*[^\d][^\]]*)\]\([^\)]+\)', r'\1', body)
                        return body + sources_section
                    else:
                        # Same pattern for text without Sources section
                        return re.sub(r'\[([^\]]*[^\d][^\]]*)\]\([^\)]+\)', r'\1', text)
                
                final_answer = replace_inline_citations(final_answer)
                
                state["sources"] = sources_list
            else:
                final_answer = "I don't have sufficient information in my knowledge base to answer this question accurately. Please try rephrasing your question or asking about a different aspect of diabetes management."
        
        # Add answer to messages
        state["messages"] = state.get("messages", []) + [AIMessage(content=final_answer)]
        
        if writer:
            writer({"type": "generator_complete", "message": f"Answer generated: {len(final_answer)} chars"})
        
        print(f"✓ Generator complete: Answer={len(final_answer) if final_answer else 0} chars, Chunks={len(state.get('retrieved_chunks', []))}")
        
    except Exception as e:
        error_msg = f"Error during generation: {str(e)[:200]}"
        print(f"⚠ Generator error: {e}")
        state["messages"] = state.get("messages", []) + [AIMessage(content=error_msg)]
        if writer:
            writer({"type": "generator_error", "message": error_msg})
    
    return state

