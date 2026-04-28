# Justification: consolidate app, routes, and session handling for a MECE backend bundle.
"""FastAPI application entry point."""
from __future__ import annotations

import json
import logging
import time
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.config import RunnableConfig

from .config import (
    validate_config,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CORS_ORIGINS_LIST,
    MAX_REQUEST_SIZE_KB,
    LANGSMITH_TRACING,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
)
from .store import ChromaDBReader
from .llm import JinaEmbeddingFunction
from .rag import build_graph
from .schema import ChatRequest, ClearChatRequest, HealthResponse, ChatState, Source

# LangSmith is optional; keep runtime resilient if not installed.
try:
    from langsmith import tracing_context, Client as LangSmithClient
except Exception:  # pragma: no cover
    tracing_context = None
    LangSmithClient = None


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions in memory.
    Sessions expire after 30 minutes of inactivity.
    """

    def __init__(self, expiration_minutes: int = 30):
        self.sessions: Dict[str, ChatState] = {}
        self.session_timestamps: Dict[str, float] = {}
        self.expiration_seconds = expiration_minutes * 60
        self.cleanup_task = None

    def get_session(self, session_id: Optional[str] = None) -> Tuple[str, ChatState]:
        self._cleanup_expired_sessions()
        if session_id is None:
            session_id = str(uuid.uuid4())

        current_time = time.time()

        if session_id not in self.sessions:
            self.sessions[session_id] = ChatState(
                messages=[],
                classifier_output=None,
                retrieved_chunks=[],
                generator_output=None,
                sources=[],
                final_response=None,
            )

        self.session_timestamps[session_id] = current_time
        return session_id, self.sessions[session_id]

    def update_session(self, session_id: str, state: ChatState) -> None:
        self.sessions[session_id] = state
        self.session_timestamps[session_id] = time.time()

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id] = ChatState(
                messages=[],
                classifier_output=None,
                retrieved_chunks=[],
                generator_output=None,
                sources=[],
                final_response=None,
            )
            return True
        return False

    def _cleanup_expired_sessions(self) -> None:
        current_time = time.time()
        expired_sessions = [
            session_id
            for session_id, last_access in self.session_timestamps.items()
            if current_time - last_access > self.expiration_seconds
        ]

        for session_id in expired_sessions:
            self.sessions.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)

        if expired_sessions:
            print(f"[OK] Cleaned up {len(expired_sessions)} expired session(s)")

    def start_cleanup_task(self) -> None:
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)
                self._cleanup_expired_sessions()

        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(cleanup_loop())
            print("[OK] Session cleanup task started (runs every 5 minutes)")

    def stop_cleanup_task(self) -> None:
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            print("[OK] Session cleanup task stopped")


_session_manager = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


_langsmith_client = None


def _get_langsmith_context():
    """Return a tracing context manager when LangSmith is configured."""
    global _langsmith_client
    if not LANGSMITH_TRACING or not LANGSMITH_API_KEY:
        return None
    if tracing_context is None or LangSmithClient is None:
        return None
    if _langsmith_client is None:
        _langsmith_client = LangSmithClient(api_key=LANGSMITH_API_KEY, api_url=LANGSMITH_ENDPOINT)
    return tracing_context(enabled=True, client=_langsmith_client, project_name=LANGSMITH_PROJECT)


# Global instances
chroma_reader = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("INITIALIZING BACKEND")
    print("=" * 60)

    try:
        validate_config()
        print("[OK] Configuration validated")
    except ValueError as exc:
        print(f"[ERROR] Configuration error: {exc}")
        raise

    try:
        jina_embedding_fn = JinaEmbeddingFunction()
        print("[OK] Jina embedding function initialized")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize Jina embedding: {exc}")
        raise

    global chroma_reader
    try:
        chroma_reader = ChromaDBReader(
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=jina_embedding_fn,
        )
        chroma_reader.initialize()
        print("[OK] ChromaDB reader initialized")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize ChromaDB reader: {exc}")
        raise

    global graph
    try:
        graph = build_graph(chroma_reader)
        print("[OK] LangGraph workflow compiled")
    except Exception as exc:
        print(f"[ERROR] Failed to build graph: {exc}")
        raise

    session_manager = get_session_manager()
    session_manager.start_cleanup_task()

    print("=" * 60)
    print("[OK] Backend initialized successfully")
    print("=" * 60)

    yield

    print("Shutting down backend...")
    session_manager = get_session_manager()
    session_manager.stop_cleanup_task()


app = FastAPI(
    title="Diabetes Knowledge Management API",
    description="RAG-based chat API for diabetes guidelines",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_REQUEST_SIZE = MAX_REQUEST_SIZE_KB * 1024


@app.middleware("http")
async def request_size_limit_middleware(request: Request, call_next):
    """Limit request body size to prevent abuse."""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > MAX_REQUEST_SIZE:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE // 1024}KB."
                        },
                    )
            except ValueError:
                pass

    response = await call_next(request)
    return response


async def stream_chat_response(message: str, session_id: str, state: ChatState):
    """
    Stream chat response with status updates and final answer.
    """
    try:
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=message))
        state["messages"] = messages

        last_status = None
        final_answer = None
        final_sources = []
        previous_state = state.copy() if hasattr(state, "copy") else dict(state)
        final_state = state
        accumulated_content = ""
        streaming_started = False
        current_node = None

        tracing_ctx = _get_langsmith_context()
        if tracing_ctx:
            with tracing_ctx:
                async for event in graph.astream_events(
                    state,
                    version="v2",
                    config=RunnableConfig(configurable={"thread_id": session_id}),
                ):
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    metadata = event.get("metadata", {})

                    if event_type == "on_chain_start":
                        if "langgraph_node" in metadata:
                            current_node = metadata.get("langgraph_node")
                        elif "classifier" in event_name.lower():
                            current_node = "classifier"
                        elif "generator" in event_name.lower():
                            current_node = "generator"
                        elif "retrieval" in event_name.lower():
                            current_node = "retrieval"

                    if event_type == "on_chat_model_stream":
                        if current_node == "generator":
                            chunk_data = event.get("data", {}).get("chunk", {})
                            if hasattr(chunk_data, "content") and chunk_data.content:
                                content = chunk_data.content
                                if not streaming_started:
                                    streaming_started = True
                                    yield json.dumps({"type": "stream_start", "content": ""}) + "\n"

                                accumulated_content += content
                                yield json.dumps({"type": "token", "content": content}) + "\n"

                    elif event_type == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if output and isinstance(output, dict):
                            for key, value in output.items():
                                final_state[key] = value

                            if event_name == "classifier":
                                classifier_output = output.get("classifier_output")
                                if classifier_output:
                                    if classifier_output.route == "retrieve" and classifier_output.intent:
                                        intent_display = classifier_output.intent
                                        if len(intent_display) > 100:
                                            intent_display = intent_display[:97] + "..."
                                        yield json.dumps(
                                            {
                                                "type": "status",
                                                "message": f"I am getting the relevant resources to answer: {intent_display}",
                                            }
                                        ) + "\n"
                                    elif classifier_output.status_message:
                                        yield json.dumps(
                                            {"type": "status", "message": classifier_output.status_message}
                                        ) + "\n"

                                    if classifier_output.direct_response and classifier_output.route == "direct":
                                        final_answer = classifier_output.direct_response
                                        final_sources = []

                            elif event_name == "retrieval":
                                chunks = output.get("retrieved_chunks", [])
                                if chunks:
                                    yield json.dumps(
                                        {
                                            "type": "status",
                                            "message": f"Found {len(chunks)} relevant sources. Generating answer...",
                                        }
                                    ) + "\n"
                                else:
                                    yield json.dumps(
                                        {"type": "status", "message": "No sources found with sufficient relevance. Responding..."}
                                    ) + "\n"

                            elif event_name == "generator" and "sources" in output:
                                sources = output["sources"]
                                if sources:
                                    final_sources = [
                                        {
                                            "title": s.title if isinstance(s, Source) else s.get("title", ""),
                                            "url": s.url if isinstance(s, Source) else s.get("url", ""),
                                            "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", ""),
                                        }
                                        for s in sources
                                    ]
        else:
            async for event in graph.astream_events(
                state,
                version="v2",
                config=RunnableConfig(configurable={"thread_id": session_id}),
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                metadata = event.get("metadata", {})

                if event_type == "on_chain_start":
                    if "langgraph_node" in metadata:
                        current_node = metadata.get("langgraph_node")
                    elif "classifier" in event_name.lower():
                        current_node = "classifier"
                    elif "generator" in event_name.lower():
                        current_node = "generator"
                    elif "retrieval" in event_name.lower():
                        current_node = "retrieval"

                if event_type == "on_chat_model_stream":
                    if current_node == "generator":
                        chunk_data = event.get("data", {}).get("chunk", {})
                        if hasattr(chunk_data, "content") and chunk_data.content:
                            content = chunk_data.content
                            if not streaming_started:
                                streaming_started = True
                                yield json.dumps({"type": "stream_start", "content": ""}) + "\n"

                            accumulated_content += content
                            yield json.dumps({"type": "token", "content": content}) + "\n"

                elif event_type == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if output and isinstance(output, dict):
                        for key, value in output.items():
                            final_state[key] = value

                        if event_name == "classifier":
                            classifier_output = output.get("classifier_output")
                            if classifier_output:
                                if classifier_output.route == "retrieve" and classifier_output.intent:
                                    intent_display = classifier_output.intent
                                    if len(intent_display) > 100:
                                        intent_display = intent_display[:97] + "..."
                                    yield json.dumps(
                                        {
                                            "type": "status",
                                            "message": f"I am getting the relevant resources to answer: {intent_display}",
                                        }
                                    ) + "\n"
                                elif classifier_output.status_message:
                                    yield json.dumps({"type": "status", "message": classifier_output.status_message}) + "\n"

                                if classifier_output.direct_response and classifier_output.route == "direct":
                                    final_answer = classifier_output.direct_response
                                    final_sources = []

                        elif event_name == "retrieval":
                            chunks = output.get("retrieved_chunks", [])
                            if chunks:
                                yield json.dumps(
                                    {
                                        "type": "status",
                                        "message": f"Found {len(chunks)} relevant sources. Generating answer...",
                                    }
                                ) + "\n"
                            else:
                                yield json.dumps(
                                    {"type": "status", "message": "No sources found with sufficient relevance. Responding..."}
                                ) + "\n"

                        elif event_name == "generator" and "sources" in output:
                            sources = output["sources"]
                            if sources:
                                final_sources = [
                                    {
                                        "title": s.title if isinstance(s, Source) else s.get("title", ""),
                                        "url": s.url if isinstance(s, Source) else s.get("url", ""),
                                        "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", ""),
                                    }
                                    for s in sources
                                ]

        if not streaming_started:
            async for chunk in graph.astream(
                state,
                config=RunnableConfig(configurable={"thread_id": session_id}),
                stream_mode="updates",
            ):
                for node_name, node_state in chunk.items():
                    if isinstance(node_state, dict):
                        for key, value in node_state.items():
                            if key == "messages":
                                if "messages" not in final_state:
                                    final_state["messages"] = []
                                existing_contents = {
                                    msg.content if hasattr(msg, "content") else str(msg) for msg in final_state["messages"]
                                }
                                for msg in value:
                                    msg_content = msg.content if hasattr(msg, "content") else str(msg)
                                    if msg_content not in existing_contents:
                                        final_state["messages"].append(msg)
                                        existing_contents.add(msg_content)
                            else:
                                final_state[key] = value

                    if not streaming_started:
                        status_msg = None
                        if node_name == "classifier":
                            classifier_output = node_state.get("classifier_output")
                            if classifier_output:
                                if classifier_output.route == "retrieve" and classifier_output.intent:
                                    intent_display = classifier_output.intent
                                    if len(intent_display) > 100:
                                        intent_display = intent_display[:97] + "..."
                                    status_msg = f"I am getting the relevant resources to answer: {intent_display}"
                                elif classifier_output.status_message:
                                    status_msg = classifier_output.status_message
                        if node_name == "retrieval":
                            chunks = node_state.get("retrieved_chunks", [])
                            if chunks:
                                status_msg = f"Found {len(chunks)} relevant sources. Generating answer..."
                            else:
                                status_msg = "No sources found with sufficient relevance. Responding..."
                        if node_name == "generator":
                            final_response = node_state.get("final_response")
                            generator_output = node_state.get("generator_output")
                            if not final_response and not (generator_output and generator_output.response):
                                status_msg = "Generating answer..."

                        if status_msg and status_msg != last_status:
                            yield json.dumps({"type": "status", "message": status_msg}) + "\n"
                            last_status = status_msg

                    if node_name == "classifier":
                        classifier_output = node_state.get("classifier_output")
                        if classifier_output and classifier_output.route == "direct":
                            if classifier_output.direct_response:
                                final_answer = classifier_output.direct_response
                                final_sources = []

                    if node_name == "generator":
                        if "sources" in node_state:
                            sources = node_state["sources"]
                            if sources:
                                final_sources = [
                                    {
                                        "title": s.title if isinstance(s, Source) else s.get("title", ""),
                                        "url": s.url if isinstance(s, Source) else s.get("url", ""),
                                        "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", ""),
                                    }
                                    for s in sources
                                ]

                    if isinstance(node_state, dict):
                        previous_state = {**previous_state, **node_state}

        session_manager = get_session_manager()
        session_manager.update_session(session_id, final_state)

        if not final_answer:
            if "final_response" in final_state and final_state["final_response"]:
                final_answer = final_state["final_response"]
            elif "generator_output" in final_state:
                gen_output = final_state["generator_output"]
                if gen_output and gen_output.response:
                    final_answer = gen_output.response
            elif "classifier_output" in final_state:
                classifier_output = final_state["classifier_output"]
                if classifier_output and classifier_output.direct_response:
                    final_answer = classifier_output.direct_response
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

        if not final_sources and "sources" in final_state:
            sources = final_state["sources"]
            if sources:
                final_sources = [
                    {
                        "title": s.title if isinstance(s, Source) else s.get("title", ""),
                        "url": s.url if isinstance(s, Source) else s.get("url", ""),
                        "chunk_id": s.chunk_id if isinstance(s, Source) else s.get("chunk_id", ""),
                    }
                    for s in sources
                ]

        if streaming_started and accumulated_content:
            yield json.dumps(
                {
                    "type": "stream_end",
                    "content": accumulated_content,
                    "sources": final_sources,
                    "session_id": session_id,
                }
            ) + "\n"
        elif final_answer:
            is_direct_response = False
            if "classifier_output" in final_state:
                classifier_output = final_state["classifier_output"]
                if classifier_output and classifier_output.direct_response == final_answer:
                    is_direct_response = True

            if is_direct_response or "generator_output" in final_state or "final_response" in final_state:
                yield json.dumps(
                    {
                        "type": "answer",
                        "content": final_answer,
                        "sources": final_sources,
                        "session_id": session_id,
                    }
                ) + "\n"
            else:
                yield json.dumps(
                    {
                        "type": "answer",
                        "content": "I apologize, but I couldn't generate a response. Please try again.",
                        "sources": [],
                        "session_id": session_id,
                    }
                ) + "\n"
        else:
            yield json.dumps(
                {
                    "type": "answer",
                    "content": "I apologize, but I couldn't generate a response. Please try again.",
                    "sources": [],
                    "session_id": session_id,
                }
            ) + "\n"

    except Exception as exc:
        logger.error(f"Error in stream_chat_response: {str(exc)}", exc_info=True)
        error_str = str(exc)
        error_msg = "An error occurred while processing your request. Please try again."

        if "pydantic" in error_str.lower() or "validation" in error_str.lower():
            error_msg = "Invalid request format. Please check your input and try again."
        elif "rate limit" in error_str.lower():
            error_msg = "Too many requests. Please wait a moment and try again."
        elif "session" in error_str.lower() and "not found" in error_str.lower():
            error_msg = "Session not found. Please start a new conversation."
        elif "timeout" in error_str.lower():
            error_msg = "Request timed out. Please try again."
        elif "connection" in error_str.lower() or "network" in error_str.lower():
            error_msg = "Network error. Please check your connection and try again."

        yield json.dumps({"type": "error", "message": error_msg, "session_id": session_id}) + "\n"


@app.post("/api/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """Chat endpoint with streaming response."""
    if graph is None:
        raise HTTPException(status_code=500, detail="Service temporarily unavailable. Please try again later.")

    session_manager = get_session_manager()
    session_id, state = session_manager.get_session(chat_request.session_id)

    return StreamingResponse(
        stream_chat_response(chat_request.message, session_id, state),
        media_type="application/x-ndjson",
    )


@app.post("/api/chat/clear")
async def clear_chat_endpoint(request: Request, clear_request: ClearChatRequest):
    """Clear conversation history for a session."""
    session_manager = get_session_manager()
    cleared = session_manager.clear_session(clear_request.session_id)

    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "cleared", "session_id": clear_request.session_id}


@app.get("/api/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Diabetes Knowledge Management API",
        "version": "1.0.0",
        "endpoints": {"chat": "/api/chat", "clear": "/api/chat/clear", "health": "/api/health"},
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path

    backend_dir = Path(__file__).parent
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
