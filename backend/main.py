"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import validate_config, CHROMA_DB_PATH, COLLECTION_NAME
from .chromadb_reader import ChromaDBReader
from .jina_embedding import JinaEmbeddingFunction
from .graph_builder import build_graph
from .routes import router, set_graph, set_chroma_reader_instance


# Global instances
chroma_reader = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("=" * 60)
    print("INITIALIZING BACKEND")
    print("=" * 60)
    
    # Validate configuration
    try:
        validate_config()
        print("✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        raise
    
    # Initialize Jina embedding function
    try:
        jina_embedding_fn = JinaEmbeddingFunction()
        print("✓ Jina embedding function initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Jina embedding: {e}")
        raise
    
    # Initialize ChromaDB reader
    global chroma_reader
    try:
        chroma_reader = ChromaDBReader(
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=jina_embedding_fn
        )
        chroma_reader.initialize()
        print("✓ ChromaDB reader initialized")
    except Exception as e:
        print(f"✗ Failed to initialize ChromaDB reader: {e}")
        raise
    
    # Build and compile graph
    global graph
    try:
        graph = build_graph(chroma_reader)
        print("✓ LangGraph workflow compiled")
    except Exception as e:
        print(f"✗ Failed to build graph: {e}")
        raise
    
    # Set global references in routes
    set_chroma_reader_instance(chroma_reader)
    set_graph(graph)
    
    print("=" * 60)
    print("✓ Backend initialized successfully")
    print("=" * 60)
    
    yield
    
    # Shutdown (if needed)
    print("Shutting down backend...")


# Create FastAPI app
app = FastAPI(
    title="Diabetes Knowledge Management API",
    description="RAG-based chat API for diabetes guidelines",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Diabetes Knowledge Management API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "clear": "/api/chat/clear",
            "health": "/api/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path
    
    # Ensure we can import backend module
    backend_dir = Path(__file__).parent
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

