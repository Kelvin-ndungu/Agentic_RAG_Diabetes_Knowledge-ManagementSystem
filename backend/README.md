# Backend - Diabetes Knowledge Management API

FastAPI backend with RAG pipeline for querying diabetes clinical guidelines using LangGraph, ChromaDB, and Claude Haiku.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with required variables
# See Environment Variables section below

# Run server
python -m backend.main
```

The API will be available at `http://localhost:8000`

## Architecture Overview

### Technology Stack
- **FastAPI** - Web framework with streaming support
- **LangGraph** - Workflow orchestration
- **ChromaDB** - Vector database for semantic search
- **Jina Embeddings** - Embedding model (v4)
- **Claude Haiku 4.5** - LLM for generation

### RAG Pipeline

The workflow follows this pattern:

1. **Classification** (`classify_query`) - Verifies query relevance and safety
2. **Routing** (`route_classifier`) - Routes to appropriate handler
3. **Retrieval** - Semantic search in ChromaDB using Jina embeddings
4. **Generation** (`generator_node`) - Claude generates answer with citations
5. **Streaming** - Status updates and final answer streamed to client

### Key Modules

- **`main.py`** - FastAPI app initialization
  - Validates configuration
  - Initializes ChromaDB reader and Jina embeddings
  - Builds and compiles LangGraph workflow
  - Sets up CORS for frontend

- **`routes.py`** - API endpoints
  - `POST /api/chat` - Streaming chat endpoint
  - `POST /api/chat/clear` - Clear session history
  - `GET /api/health` - Health check

- **`graph_builder.py`** - LangGraph workflow construction
  - Defines node connections and routing logic
  - Compiles graph for execution

- **`graph_nodes.py`** - Workflow node implementations
  - Query classification
  - Semantic retrieval from ChromaDB
  - LLM generation with context

- **`session_manager.py`** - Conversation state management
  - Maintains chat history per session
  - Thread-based state persistence

## API Endpoints

### POST `/api/chat`
Streaming chat endpoint that processes queries through the RAG pipeline.

**Request:**
```json
{
  "message": "How is type 2 diabetes diagnosed?",
  "session_id": "optional-session-id"
}
```

**Response:** Newline-delimited JSON stream:
```json
{"type": "status", "message": "Verifying relevance..."}
{"type": "status", "message": "Fetching relevant information..."}
{"type": "answer", "content": "...", "sources": [...], "session_id": "..."}
```

### POST `/api/chat/clear`
Clear conversation history for a session.

### GET `/api/health`
Health check endpoint.

## Environment Variables

Create a `.env` file in the project root:

```env
CLAUDE_API_KEY=your_claude_api_key_here
JINA_API_KEY=your_jina_api_key_here
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=diabetes_guidelines_v1
```

**Required:**
- `CLAUDE_API_KEY` - Anthropic API key for Claude
- `JINA_API_KEY` - Jina AI API key for embeddings

**Optional (with defaults):**
- `CHROMA_DB_PATH` - Path to ChromaDB directory (default: `./chroma_db`)
- `COLLECTION_NAME` - ChromaDB collection name (default: `diabetes_guidelines_v1`)

## Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── routes.py            # API endpoints
├── graph_builder.py     # LangGraph workflow
├── graph_nodes.py       # Workflow node logic
├── config.py            # Configuration and validation
├── chromadb_reader.py   # ChromaDB integration
├── jina_embedding.py    # Jina embedding function
├── session_manager.py   # Session state management
├── models.py            # Pydantic models
└── requirements.txt     # Python dependencies
```

## How It Works

1. **Startup**: Server initializes ChromaDB connection and compiles LangGraph workflow
2. **Query Processing**: User message → classification → retrieval → generation
3. **Streaming**: Status updates and final answer streamed as newline-delimited JSON
4. **Session Management**: Conversation history maintained per session ID
5. **Citations**: Retrieved chunks included as sources in response

The backend provides a RAG-powered API that enables semantic search over diabetes guidelines and generates contextual answers with source citations.
