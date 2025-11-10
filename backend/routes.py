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
    # Classification node
    if node_name == "classify":
        return "Verifying relevance..."
    
    # Generator node - check what's happening
    if node_name == "generator":
        # Check if retrieval just happened
        current_chunks = node_state.get("retrieved_chunks", [])
        prev_chunks = previous_state.get("retrieved_chunks", []) if previous_state else []
        
        # Check if we have messages (answer generated)
        has_answer = False
        if "messages" in node_state:
            messages = node_state.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    has_answer = True
                    break
        
        # If answer is ready, no status needed
        if has_answer:
            return None
        
        # If chunks were just retrieved
        if len(current_chunks) > len(prev_chunks):
            return f"Retrieved {len(current_chunks)} relevant chunks. Generating answer..."
        
        # If we have chunks but no answer yet
        if current_chunks and not has_answer:
            return "Generating answer..."
        
        # If we're starting generation (no chunks yet, no answer)
        if not current_chunks and not has_answer:
            return "Searching knowledge base..."
    
    return None


def stream_chat_response(
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
        
        # Track status updates
        last_status = None
        final_answer = None
        final_sources = []
        previous_state = state.copy() if hasattr(state, 'copy') else dict(state)
        
        # Stream graph execution with updates mode to get state changes
        final_state = state
        for chunk in graph.stream(
            state,
            config=RunnableConfig(configurable={"thread_id": session_id}),
            stream_mode="updates"  # Get state updates after each node
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
                
                # Detect status from state changes
                status_msg = detect_status_from_state(node_name, node_state, previous_state)
                if status_msg and status_msg != last_status:
                    yield json.dumps({
                        "type": "status",
                        "message": status_msg
                    }) + "\n"
                    last_status = status_msg
                
                # Note: Retrieval status is handled in detect_status_from_state
                # This ensures we don't duplicate status messages
                
                # Check for final answer
                if "messages" in node_state:
                    node_messages = node_state["messages"]
                    for msg in node_messages:
                        if isinstance(msg, AIMessage) and msg.content:
                            # This is the final answer
                            final_answer = msg.content
                            
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
        
        # Send final answer
        if final_answer:
            yield json.dumps({
                "type": "answer",
                "content": final_answer,
                "sources": final_sources,
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
        # Send error response
        error_msg = f"Error processing request: {str(e)[:200]}"
        yield json.dumps({
            "type": "error",
            "message": error_msg,
            "session_id": session_id
        }) + "\n"


@router.post("/api/chat")
def chat_endpoint(request: ChatRequest):
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
    
    # Create streaming response
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

