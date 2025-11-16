"""
API routes for FastAPI application.
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.config import RunnableConfig

from .models import ChatRequest, ClearChatRequest, HealthResponse, ChatState, Source
from .session_manager import get_session_manager


router = APIRouter()

# Global references (will be initialized in main.py)
graph = None
chroma_reader = None


def set_graph(g):
    """Set the global graph instance."""
    global graph
    graph = g


def set_chroma_reader_instance(cr):
    """Set the global chroma_reader instance."""
    global chroma_reader
    chroma_reader = cr


# Status message mapping for user-friendly updates
STATUS_MESSAGES = {
    "classify": "Verifying relevance...",
    "generator": "Analyzing query...",
    "not_relevant": None,  # Don't show status for these
    "unsafe": None,
}


def detect_status_from_state(node_name: str, node_state: dict, previous_state: dict = None) -> str:
    """
    Detect status message from state changes.
    
    Args:
        node_name: Name of the graph node
        node_state: Current node state
        previous_state: Previous state (for comparison)
        
    Returns:
        User-friendly status message or None if no status update needed
    """
    # Classifier node - use status_message with intent for substantive queries
    if node_name == "classifier":
        classifier_output = node_state.get("classifier_output")
        if classifier_output:
            # For substantive queries, include intent
            if classifier_output.should_generate and classifier_output.intent:
                intent_display = classifier_output.intent
                if len(intent_display) > 100:
                    intent_display = intent_display[:97] + "..."
                return f"I am getting the relevant resources to answer: {intent_display}"
            elif hasattr(classifier_output, 'status_message'):
                return classifier_output.status_message
        return "Processing query..."
    
    # Retrieval node
    if node_name == "retrieval":
        chunks = node_state.get("retrieved_chunks", [])
        if chunks:
            return f"Found {len(chunks)} relevant sources. Generating answer..."
        else:
            return "No sources found with sufficient relevance. Responding..."
    
    # Generator node - check if answer is ready
    if node_name == "generator":
        # Check if we have final_response or generator_output
        final_response = node_state.get("final_response")
        generator_output = node_state.get("generator_output")
        
        # If answer is ready, no status needed
        if final_response or (generator_output and generator_output.response):
            return None
        
        # Otherwise, still generating
        return "Generating answer..."
    
    return None


async def stream_chat_response(
    message: str,
    session_id: str,
    state: ChatState
):
    """
    Stream chat response with status updates and final answer.
    
    Args:
        message: User's message
        session_id: Session ID
        state: Current chat state
        
    Yields:
        JSON strings with type, message/content, and optional sources
    """
    try:
        # Add user message to state
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=message))
        state["messages"] = messages
        
        # Track status updates and streaming content
        last_status = None
        final_answer = None
        final_sources = []
        previous_state = state.copy() if hasattr(state, 'copy') else dict(state)
        final_state = state
        accumulated_content = ""  # Track accumulated content for streaming
        streaming_started = False  # Track if we've started streaming tokens
        current_node = None  # Track which node we're currently in
        
        # Use astream_events for fine-grained token streaming
        # This gives us both tokens and state updates in one pass
        async for event in graph.astream_events(
            state,
            version="v2",
            config=RunnableConfig(configurable={"thread_id": session_id}),
        ):
            event_type = event.get("event", "")
            event_name = event.get("name", "")
            metadata = event.get("metadata", {})
            
            # Track which node we're in based on chain start events
            if event_type == "on_chain_start":
                # Check metadata for LangGraph node name
                if "langgraph_node" in metadata:
                    current_node = metadata.get("langgraph_node")
                # Also check if event name indicates a node
                elif "classifier" in event_name.lower():
                    current_node = "classifier"
                elif "generator" in event_name.lower():
                    current_node = "generator"
                elif "retrieval" in event_name.lower():
                    current_node = "retrieval"
            
            # Stream tokens from LLM in generator node ONLY
            # Filter out classifier and other node tokens
            if event_type == "on_chat_model_stream":
                # Only stream if we're in the generator node
                if current_node == "generator":
                    # Get chunk content
                    chunk_data = event.get("data", {}).get("chunk", {})
                    if hasattr(chunk_data, 'content') and chunk_data.content:
                        content = chunk_data.content
                        if not streaming_started:
                            streaming_started = True
                            yield json.dumps({
                                "type": "stream_start",
                                "content": ""
                            }) + "\n"
                        
                        accumulated_content += content
                        # Stream token to frontend
                        yield json.dumps({
                            "type": "token",
                            "content": content
                        }) + "\n"
            
            # Capture state updates and status messages
            elif event_type == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if output and isinstance(output, dict):
                    # Update final_state
                    for key, value in output.items():
                        final_state[key] = value
                    
                    # Handle classifier - send status message with intent
                    if event_name == "classifier":
                        classifier_output = output.get("classifier_output")
                        if classifier_output:
                            # For substantive queries, show intent-based status
                            if classifier_output.should_generate and classifier_output.intent:
                                # Truncate intent if too long
                                intent_display = classifier_output.intent
                                if len(intent_display) > 100:
                                    intent_display = intent_display[:97] + "..."
                                yield json.dumps({
                                    "type": "status",
                                    "message": f"I am getting the relevant resources to answer: {intent_display}"
                                }) + "\n"
                            elif classifier_output.status_message:
                                # For non-substantive queries, use status_message
                                yield json.dumps({
                                    "type": "status",
                                    "message": classifier_output.status_message
                                }) + "\n"
                            
                            # Handle direct responses (non-substantive queries)
                            if classifier_output.direct_response and not classifier_output.should_generate:
                                final_answer = classifier_output.direct_response
                                final_sources = []
                    
                    # Handle retrieval - send status
                    elif event_name == "retrieval":
                        chunks = output.get("retrieved_chunks", [])
                        if chunks:
                            yield json.dumps({
                                "type": "status",
                                "message": f"Found {len(chunks)} relevant sources. Generating answer..."
                            }) + "\n"
                        else:
                            yield json.dumps({
                                "type": "status",
                                "message": "No sources found with sufficient relevance. Responding..."
                            }) + "\n"
                    
                    # Extract sources from generator
                    elif event_name == "generator" and "sources" in output:
                        sources = output["sources"]
                        if sources:
                            final_sources = [
                                {
                                    "title": s.title if isinstance(s, Source) else s.get("title", ""),
                                    "url": s.url if isinstance(s, Source) else s.get("url", ""),
                                    "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", "")
                                }
                                for s in sources
                            ]
        
        # Fallback: if no streaming happened, use updates mode
        if not streaming_started:
            for chunk in graph.stream(
                state,
                config=RunnableConfig(configurable={"thread_id": session_id}),
                stream_mode="updates"
            ):
                # Process updates from graph nodes
                for node_name, node_state in chunk.items():
                    # Merge node state into final state
                    if isinstance(node_state, dict):
                        # Create a proper merge that preserves ChatState structure
                        for key, value in node_state.items():
                            if key == "messages":
                                # Messages are appended, not replaced (MessagesState handles this)
                                # But we need to check if these are new messages
                                if "messages" not in final_state:
                                    final_state["messages"] = []
                                # Only add new messages (check by content to avoid duplicates)
                                existing_contents = {msg.content if hasattr(msg, 'content') else str(msg) for msg in final_state["messages"]}
                                for msg in value:
                                    msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                                    if msg_content not in existing_contents:
                                        final_state["messages"].append(msg)
                                        existing_contents.add(msg_content)
                            else:
                                final_state[key] = value
                    
                    # Detect status from state changes (only if not streaming yet)
                    if not streaming_started:
                        status_msg = detect_status_from_state(node_name, node_state, previous_state)
                        if status_msg and status_msg != last_status:
                            yield json.dumps({
                                "type": "status",
                                "message": status_msg
                            }) + "\n"
                            last_status = status_msg
                    
                    # Check for final answer from classifier (non-substantive queries)
                    if node_name == "classifier":
                        classifier_output = node_state.get("classifier_output")
                        if classifier_output and not classifier_output.should_generate:
                            # Non-substantive query - use direct_response
                            if classifier_output.direct_response:
                                final_answer = classifier_output.direct_response
                                final_sources = []
                    
                    # Extract sources from generator node
                    if node_name == "generator":
                        # Extract sources if available
                        if "sources" in node_state:
                            sources = node_state["sources"]
                            if sources:
                                final_sources = [
                                    {
                                        "title": s.title if isinstance(s, Source) else s.get("title", ""),
                                        "url": s.url if isinstance(s, Source) else s.get("url", ""),
                                        "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", "")
                                    }
                                    for s in sources
                                ]
                    
                    # Update previous state for next comparison
                    if isinstance(node_state, dict):
                        previous_state = {**previous_state, **node_state}
        
        # Update session state with final state
        session_manager = get_session_manager()
        session_manager.update_session(session_id, final_state)
        
        # If we didn't get final_answer from streaming, check final_state
        if not final_answer:
            # Check final_response
            if "final_response" in final_state and final_state["final_response"]:
                final_answer = final_state["final_response"]
            # Check generator_output
            elif "generator_output" in final_state:
                gen_output = final_state["generator_output"]
                if gen_output and gen_output.response:
                    final_answer = gen_output.response
            # Check classifier_output direct_response
            elif "classifier_output" in final_state:
                classifier_output = final_state["classifier_output"]
                if classifier_output and classifier_output.direct_response:
                    final_answer = classifier_output.direct_response
            # Fallback to messages
            elif "messages" in final_state:
                messages = final_state["messages"]
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        content = msg.content
                        if isinstance(content, str):
                            final_answer = content
                        elif isinstance(content, list):
                            final_answer = "".join(str(item) for item in content)
                        else:
                            final_answer = str(content)
                        break
        
        # Extract sources from final_state if not already extracted
        if not final_sources and "sources" in final_state:
            sources = final_state["sources"]
            if sources:
                final_sources = [
                    {
                        "title": s.title if isinstance(s, Source) else s.get("title", ""),
                        "url": s.url if isinstance(s, Source) else s.get("url", ""),
                        "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", "")
                    }
                    for s in sources
                ]
        
        # Send final answer with sources (if we streamed, use accumulated content)
        # IMPORTANT: Only send final answer from generator, never classifier JSON
        if streaming_started and accumulated_content:
            # We already streamed the content token-by-token, just send sources to finalize
            yield json.dumps({
                "type": "stream_end",
                "content": accumulated_content,
                "sources": final_sources,
                "session_id": session_id
            }) + "\n"
        elif final_answer:
            # Non-streaming answer - only send if it's from generator or classifier direct_response
            # Never send classifier_output JSON
            # Check if this is a direct response (non-substantive) or generator output
            is_direct_response = False
            if "classifier_output" in final_state:
                classifier_output = final_state["classifier_output"]
                if classifier_output and classifier_output.direct_response == final_answer:
                    is_direct_response = True
            
            # Only send if it's a proper answer (direct response or generator output)
            # Never send raw classifier JSON
            if is_direct_response or "generator_output" in final_state or "final_response" in final_state:
                yield json.dumps({
                    "type": "answer",
                    "content": final_answer,
                    "sources": final_sources,
                    "session_id": session_id
                }) + "\n"
            else:
                # Fallback if no proper answer was generated
                yield json.dumps({
                    "type": "answer",
                    "content": "I apologize, but I couldn't generate a response. Please try again.",
                    "sources": [],
                    "session_id": session_id
                }) + "\n"
        else:
            # Fallback if no answer was generated
            yield json.dumps({
                "type": "answer",
                "content": "I apologize, but I couldn't generate a response. Please try again.",
                "sources": [],
                "session_id": session_id
            }) + "\n"
            
    except Exception as e:
        # Send error response - clean up any Pydantic or internal error details
        error_str = str(e)
        # Remove Pydantic validation details if present
        if "pydantic" in error_str.lower() or "validation" in error_str.lower():
            error_msg = "An error occurred while processing your request. Please try again."
        else:
            # Truncate and clean error message
            error_msg = error_str[:200] if len(error_str) > 200 else error_str
            # Remove any object representation that might leak internal details
            if "<" in error_msg and "object at" in error_msg:
                error_msg = "An error occurred while processing your request. Please try again."
        
        yield json.dumps({
            "type": "error",
            "message": error_msg,
            "session_id": session_id
        }) + "\n"


@router.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with streaming response.
    
    Accepts a message and optional session_id, returns streaming response
    with status updates and final answer.
    """
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    
    # Get or create session
    session_manager = get_session_manager()
    session_id, state = session_manager.get_session(request.session_id)
    
    # Create streaming response (async generator)
    return StreamingResponse(
        stream_chat_response(request.message, session_id, state),
        media_type="application/x-ndjson"  # Newline-delimited JSON
    )


@router.post("/api/chat/clear")
async def clear_chat_endpoint(request: ClearChatRequest):
    """
    Clear conversation history for a session.
    """
    session_manager = get_session_manager()
    cleared = session_manager.clear_session(request.session_id)
    
    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "cleared", "session_id": request.session_id}


@router.get("/api/health", response_model=HealthResponse)
async def health_endpoint():
    """
    Health check endpoint.
    """
    return HealthResponse(status="ok")

