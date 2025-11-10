# Diabetes Knowledge Management System

A production-ready RAG (Retrieval-Augmented Generation) application that transforms clinical guidelines into an intelligent, searchable knowledge base with verifiable source citations.

## The Challenge

Medical literature presents unique challenges for AI systems: information must be complete, traceable, and accurate. Silent failures—where content is truncated, context is lost, or sources are fabricated—are not merely technical issues but potential risks to clinical decision-making.

This system addresses these challenges through a design philosophy that prioritizes **visibility over automation**, **completeness over convenience**, and **verification over assumption**.

## System Architecture

### Development Philosophy: Test-Driven Pipeline Development

Rather than building a black-box pipeline, this project adopts a **notebook-first development approach** that ensures each transformation stage is validated before moving to production:

```
PDF → Markdown → Cleaned Structure → Hierarchical Chunks → Vector Store → RAG Pipeline → Production API
 ↓        ↓              ↓                    ↓                ↓              ↓              ↓
01_     02_            02_                  03_             04_           05_ + 06_      backend/
test    validate       verify               inspect         search        prototype      deploy
```

Each notebook serves as both **implementation** and **validation artifact**, providing granular control and traceability throughout the development lifecycle.

### Technical Stack

**Data Processing Pipeline:**
- **PDF Extraction**: Docling + PyMuPDF for precise image extraction with coordinate-based positioning
- **Structure Parsing**: Custom hierarchical parser with TOC validation and fuzzy matching
- **Chunking**: Document-structure-aware chunking that preserves hierarchical context
- **Embeddings**: Jina Embeddings v4 (semantic + keyword capture without hybrid retrieval)
- **Vector Store**: ChromaDB with persistent storage

**RAG Application:**
- **Workflow Orchestration**: LangGraph with typed state machines and conditional routing
- **Query Classification**: Multi-stage safety and relevance filtering
- **LLM**: Claude Haiku 4.5 for production / Ollama for local development
- **Backend**: FastAPI with Server-Sent Events for streaming responses
- **Frontend**: React 19 with hierarchical navigation and resizable chat interface

## Key Design Decisions

### 1. Custom Chunking Over Generic Splitters

The project implements a **custom hierarchical chunking parser** instead of using LangChain's `RecursiveCharacterTextSplitter` or `MarkdownHeaderTextSplitter`.

**Why?** Generic splitters optimize for size-based splitting, which can:
- Break logical boundaries mid-section
- Lose "orphan content" (text between parent heading and first child)
- Discard hierarchical metadata critical for precise citations

**Our approach:**
- Chunks align with document structure (H1/H2/H3/H4 boundaries)
- Orphan sections explicitly preserved as `introContent`
- Full breadcrumb trails maintain hierarchical context
- Section numbers extracted for URL generation: `#/guidelines/chapter-1/section-1-1`

This enables **precise source citations** where each retrieved chunk links to its exact location in the source document—critical for clinical guidelines where doctors need to verify recommendations.

### 2. Semantic-Only Retrieval (No Hybrid Search)

Through systematic testing in the `05_rag_pipeline_v1.ipynb` Gradio interface, we discovered that **Jina Embeddings v4 successfully embeds keywords within semantic representations**.

Queries containing specific medical terminology (`HbA1c`, `DKA`, `insulin resistance`) retrieved relevant sections even when wording differed from source text. This eliminated the need for hybrid retrieval strategies (semantic + BM25 keyword search), simplifying the architecture without sacrificing accuracy.

**Testing methodology:**
- Gradio interface for rapid query iteration
- Similarity score analysis (0.4 threshold for relevance)
- Keyword-focused vs semantic query comparison
- Cross-validation with known section locations

### 3. Safety-First Query Classification

The system implements **multi-stage query classification** before retrieval:

```
User Query
    ↓
Is it relevant to diabetes? ────→ No → Polite refusal
    ↓ Yes
Is it safe to answer? ─────────→ No → Medical advice disclaimer
    ↓ Yes
Retrieve + Generate with citations
```

**Safety categories:**
- **Not relevant**: Queries outside diabetes management scope
- **High risk**: Questions requiring personalized medical diagnosis
- **Medium risk**: Patient-specific outcome predictions
- **Safe**: General guideline questions answerable from source material

This prevents the system from providing potentially harmful advice while maintaining utility for clinical guideline lookup.

### 4. Verifiable Source Citations

Every generated answer includes:
1. **Inline citations**: `[Section Title](url)` format embedded in response text
2. **Sources section**: Numbered list with clickable links to exact document locations
3. **Chunk metadata**: Title, breadcrumb, section number, URL, token count

Example output:
```markdown
Type 2 diabetes is diagnosed using fasting plasma glucose ≥7.0 mmol/L or HbA1c ≥6.5% 
[1.3. Diagnosis of diabetes](/guidelines/chapter-1/section-1-3).

## Sources
1. [1.3. Diagnosis of diabetes](/guidelines/chapter-1/section-1-3)
```

This design ensures **every claim is traceable** to source material, preventing LLM hallucination and enabling medical professionals to verify recommendations against original guidelines.

## Development Workflow

### Notebook-Based Prototyping with Gradio

Each pipeline stage was developed in Jupyter notebooks with **Gradio interfaces for interactive testing**:

- **`01_data_extraction.ipynb`**: Validates PDF → Markdown conversion with image preservation
- **`02_data_cleaning_v1.ipynb`**: Verifies heading hierarchy correction and TOC alignment
- **`03_chunking_v1.ipynb`**: Inspects chunk boundaries and orphan section handling
- **`04_vector_store_v1.ipynb`**: Tests embedding and ChromaDB insertion
- **`05_rag_pipeline_v1.ipynb`**: Gradio search interface for retrieval quality testing
- **`06_generation_v3.ipynb`**: Complete RAG pipeline with LangGraph orchestration

**Why Gradio?** Immediate visual feedback during development:
- Test retrieval quality with various query types
- Validate chunk relevance scores
- Verify citation accuracy
- Iterate on prompt engineering with real-time results

### Transition to Production

After notebook validation, components are refactored into the production backend (`backend/`) maintaining the same:
- LangGraph workflow structure
- ChromaDB integration
- Jina embedding function
- Query classification logic

The notebooks serve as **living documentation** showing exactly how each component was developed and tested.

## Quality Assurances

### Completeness: No Silent Failures

**Token counting at every stage:**
- Extract → 120,679 tokens total
- Clean → Validate token preservation
- Chunk → Sum of chunk tokens equals source tokens
- Orphan sections explicitly tracked: 8 sections preserved as `introContent`

**Validation logs:**
```
✓ Found 68 numbered entries in TOC
✓ Matched by number: 69
✓ Orphan sections preserved: 8
✓ All orphan sections successfully preserved!
```

### Accuracy: Semantic Search Validation

**Minimum similarity threshold**: 0.4 (cosine similarity)
- Chunks below threshold excluded from results
- Relevance scores displayed for transparency
- Manual verification against known section locations during development

### Safety: Multi-Level Classification

**Query routing with explicit reasoning:**
```python
QuerySafetyClassification(
    is_relevant: bool,
    is_safe: bool,
    risk_level: "none" | "low" | "medium" | "high",
    reasoning: str  # Logged for audit
)
```

Classification reasoning logged for each query, enabling audit trails and continuous improvement of safety filters.

## Repository Structure

```
├── 01_data_extraction.ipynb          # PDF → Markdown with images
├── 02_data_cleaning_v1.ipynb         # Heading hierarchy and TOC validation
├── 03_chunking_v1.ipynb              # Hierarchical document chunking
├── 04_vector_store_v1.ipynb          # Embedding and ChromaDB setup
├── 05_rag_pipeline_v1.ipynb          # Retrieval testing with Gradio
├── 06_generation_v3.ipynb            # Complete RAG pipeline prototype
├── backend/                          # Production FastAPI application
│   ├── main.py                       # Server initialization
│   ├── routes.py                     # Streaming chat endpoint
│   ├── graph_builder.py              # LangGraph workflow
│   ├── graph_nodes.py                # Classification, retrieval, generation
│   ├── session_manager.py            # Conversation history
│   └── README.md                     # Backend documentation
├── frontend/                         # React application
│   ├── src/components/               # UI components
│   ├── src/data/                     # document_structure.json
│   └── README.md                     # Frontend documentation
├── chroma_db/                        # Vector database (persistent)
└── output/                           # Processed markdown and JSON
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- API Keys: Claude (Anthropic) + Jina AI

### Setup

**1. Backend**
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cat > .env << EOF
CLAUDE_API_KEY=your_key_here
JINA_API_KEY=your_key_here
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=diabetes_guidelines_v1
EOF

# Run server
python -m backend.main
```

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```

**3. Access Application**
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Development Notebooks

To explore the development pipeline:
```bash
# Install notebook dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open notebooks in sequence (`01_` through `06_`) to see each pipeline stage with validation outputs.

## Technical Highlights

### Hierarchical Context Preservation

Each chunk maintains full hierarchical context:
```json
{
  "chunk_id": "section-2-1-1",
  "content": "...",
  "metadata": {
    "title": "2.1.1. Insulin treatment",
    "level": "h3",
    "breadcrumb": ["CHAPTER TWO: MANAGEMENT OF DIABETES", "2.1. Management of Type 1 Diabetes", "2.1.1. Insulin treatment"],
    "url": "/guidelines/chapter-2/section-2-1/subsection-2-1-1",
    "parent_title": "2.1. Management of Type 1 Diabetes",
    "section_number": "2.1.1",
    "token_count": 3766
  }
}
```

### Streaming Response Architecture

FastAPI endpoint streams responses as newline-delimited JSON:
```json
{"type": "status", "message": "Verifying relevance..."}
{"type": "status", "message": "Fetching relevant information..."}
{"type": "status", "message": "Generating response..."}
{"type": "answer", "content": "...", "sources": [...], "session_id": "..."}
```

React frontend processes stream in real-time, providing progressive feedback during the 3-stage pipeline (classify → retrieve → generate).

### LangGraph Workflow Orchestration

State machine with typed state dictionary:
```python
class ChatState(MessagesState):
    classification: Optional[QuerySafetyClassification]
    retrieved_chunks: List[Dict]
    sources: List[Source]
    is_followup: bool
```

Conditional routing based on classification:
```
START → classify → route_classifier
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
  not_relevant      unsafe        generator
        ↓               ↓               ↓
       END             END             END
```

## Performance Characteristics

- **Vector Store Size**: 78 chunks from 120,679 total tokens
- **Average Chunk Size**: 1,547 tokens (preserves semantic completeness)
- **Retrieval Latency**: ~500ms (5 chunks @ 0.4 similarity threshold)
- **Generation Latency**: ~3-5s (Claude Haiku streaming)
- **Frontend Load Time**: <1s (static JSON structure, no API calls on load)

## Future Enhancements

### Version 2: Self-Improving Agents

Current implementation (`v3`) validates single-iteration RAG performance. Planned enhancements:

- **Reflection Agents**: Self-critique and answer refinement
- **ReAct Pattern**: Iterative retrieval until sufficient information gathered
- **Multi-Step Reasoning**: Sub-query decomposition for complex questions
- **Source Quality Scoring**: Weighted citations based on relevance and completeness

### Additional Capabilities

- **Multi-Document Support**: Extend to other clinical guidelines
- **Comparative Analysis**: Cross-reference recommendations between guidelines
- **Update Notifications**: Track guideline revisions and flag outdated information
- **Export Capabilities**: Generate PDF reports with citations

## License

This project demonstrates a comprehensive approach to building production-ready RAG systems for medical literature. The implementation prioritizes verifiability, safety, and completeness—essential characteristics for clinical decision support tools.

---

**Development Notes**: Each notebook includes detailed cell-level comments explaining design decisions, validation steps, and testing methodologies. The notebooks serve as both implementation and documentation, providing full transparency into the development process.

