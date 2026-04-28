# Agentic RAG for Medical Knowledge Management
### Diabetes Knowledge Management System — Portfolio Document

**Kelvin Ndungu Kinyanjui**  
Analyst · AI Practitioner · Consultant · Nairobi, Kenya  
knduchu@gmail.com | [GitHub: Kelvin-ndungu](https://github.com/Kelvin-ndungu/Agentic_RAG_Diabetes_Knowledge-ManagementSystem)

---

## The Short Version

This project takes a 157-page clinical PDF — the Kenya National Clinical Guidelines for the Management of Diabetes Mellitus, 2nd Edition 2018 — and transforms it into a production-grade, verifiable AI knowledge system. A healthcare provider can ask it a question in plain language. It classifies the query, retrieves the relevant sections, generates a cited answer, and streams it token-by-token to the user. Every factual statement is numbered and linked directly to its source in the original document. The user can click the citation and read the original text.

The system is built on LangGraph, LangChain, ChromaDB, Jina Embeddings v4, Claude Haiku, FastAPI, and React. It runs two LLM calls maximum per query. It was built entirely through self-directed learning, without a client, without a team, and without prior experience in any of these frameworks. It is a demonstration that I can identify where AI can reduce friction in knowledge-heavy business processes, architect an end-to-end solution, make deliberate engineering trade-offs, and deliver something that works.

---

## Why This Problem Exists

The starting point was not diabetes. It was a question about where generative AI creates real value in business processes.

One of the clearest answers is in knowledge management. Organisations of all kinds — health systems, law firms, financial institutions, policy bodies — sit on large volumes of structured documents that contain expert knowledge. This knowledge is locked inside those documents. To access it, a person must know which document to look in, navigate to the right section, read carefully, and synthesise across multiple sections. That process is slow, error-prone, and dependent on the individual's prior familiarity with the material.

What generative AI enables — specifically, Retrieval-Augmented Generation — is a natural language interface on top of that knowledge. You ask a question the way you would ask a colleague. The system retrieves the relevant sections. The model synthesises an answer and cites its sources. The human can verify. This is not replacing the expert. It is removing the friction between the expert and the knowledge they need.

The Kenya National Clinical Guidelines for Diabetes Management was chosen because it is a real, high-stakes document. It is not a toy dataset. It has 157 pages, a complex heading hierarchy, multiple chapters with deeply nested subsections, tables, figures, cross-references, and 63 embedded images. It covers Type 1 and Type 2 diabetes, complications, comorbidities, management in pregnancy, surgery, HIV, TB, and older adults. It is exactly the kind of document where a well-functioning knowledge system has tangible value: a clinician in a busy Kenyan public hospital does not have time to flip through 157 pages to confirm a dosing threshold or a diagnostic criterion. They need to ask a question and get an answer they can trust and verify.

Building on this document was also a deliberate strategic choice. Medical information is unforgiving. Hallucination is not an inconvenience here — it is dangerous. Building a system that handles this domain correctly demonstrates a much higher standard of reliability than building it on a forgiving subject.

---

## This is a Portfolio Project — What That Means

This system was built entirely as a self-directed investment in capability. There was no client, no budget, no team, no deadline. What drove it was a conviction: that understanding how to integrate AI into business processes — not just knowing what tools exist, but being able to build and configure them — is one of the most consequential professional skills available right now.

The project demonstrates three distinct things that belong on a professional profile:

**First**, it demonstrates AI engineering capability — the ability to go from a raw PDF to a deployed API with a working frontend, making deliberate architectural decisions at each step and being able to explain why.

**Second**, it demonstrates the analytical mindset that underpins good data and BI work. The notebooks are not just scripts — they are a record of structured exploration. Before anything was built, the data was understood. Token distributions were calculated. Document structure was visualised. Failure modes were identified and addressed explicitly. The way a good analyst works through a dataset before building a dashboard is exactly the way this project worked through the document before building the pipeline.

**Third**, it demonstrates the habit of thinking about cost, efficiency, and business fitness. The system runs two LLM calls per query because three or four would be slower and more expensive. Retrieval is programmatic, not model-driven, because LLM calls are a cost centre. Streaming is implemented because perceived performance matters to users. These are not academic concerns — they are the concerns of someone who understands that AI systems exist inside budget constraints and operational realities.

---

## The Source Material: What Makes This Hard

The Kenya National Clinical Guidelines PDF is published by the Ministry of Health, Kenya. It runs to 157 pages across 8 chapters, plus front matter, references, and appendices. The challenges it presents are representative of real-world enterprise document processing:

**Heading hierarchy complexity.** The document uses H1 through H3 headings, but the hierarchy is not always consistent. Some sections have introductory paragraphs between a parent heading and its first child — content that sits between the H1 and the first H2, which generic chunkers will typically discard or misassign. This "orphan content" is often conceptually significant.

**Token volume.** The document contains approximately 120,679 tokens at the section level. Sections range from 54 tokens (a brief heading with one sentence) to 7,011 tokens (a comprehensive management chapter). Any chunking strategy must account for this variance.

**Embedded images.** The PDF contains 63 clinical figures — dosing charts, decision trees, anatomical diagrams — that are referenced in the text. These needed to be extracted and correctly linked.

**Medical terminology density.** Acronyms like DKA (Diabetic Ketoacidosis), HHS (Hyperosmolar Hyperglycaemic State), HbA1c, SGLT2, GLP-1 must be retrievable both by acronym and by their expanded form. A user who types "DKA management" and a user who types "diabetic ketoacidosis treatment" should get the same results.

**High-stakes content.** Every section deals with clinical decisions: diagnostic thresholds, medication doses, emergency protocols. The system must not hallucinate, must cite its sources, and must refuse to provide patient-specific advice while still being genuinely useful.

---

## The Development Chronology: Notebook-First

The project was built in seven notebooks before a single line of production code was written. This was not inefficiency — it was method. Each notebook was a stage of structured exploration and validation. The production backend is a direct refactor of the notebook code, inheriting the same architecture and the same Pydantic schemas.

The notebooks exist in the repository as a transparent record of how the system was built. A reader who wants to understand the thinking behind the production code can follow the notebooks in sequence and see every decision being made and tested.

### Notebook 01: Data Extraction

The PDF was converted to Markdown using a structured extraction pipeline. The output is a Markdown file with all text content, preserved heading hierarchy, and image references mapped to extracted PNG files. This produced the raw material for everything that followed.

### Notebook 02: Data Cleaning

The raw extracted Markdown had heading hierarchy issues — sections that should have been H2 were marked as H1, creating a flat structure that would break any hierarchy-aware chunking. The cleaning notebook corrected this, tracing through the document structure and fixing the heading levels to match the logical hierarchy of the guidelines.

This step is the kind of work that gets skipped in tutorial projects and breaks real ones. Garbage-in-garbage-out applies to document structure as much as to tabular data. A mis-hierarchied document produces mis-hierarchied chunks, which produces citation errors, which breaks the entire point of the system.

### Notebook 03: Hierarchical Chunking

This is where the project diverges most significantly from standard practice, and where the analytical thinking is most visible.

**The problem with generic chunkers.** LangChain's `RecursiveCharacterTextSplitter` and `MarkdownHeaderTextSplitter` are designed for general use. They split by token count or by heading markers. For this document, that approach would cause two specific failures:

- **Orphan content loss.** The document has 8 sections where meaningful text appears between a parent heading and its first child. A heading-based splitter that simply splits at headings discards this text. In a medical guidelines document, this bridging content often contains the clinical context that makes the child sections interpretable.

- **Citation imprecision.** Generic size-based splitting creates chunk boundaries in the middle of sections, making it impossible to cite cleanly. If a user asks about HbA1c targets and the answer spans two chunks that sit across an arbitrary boundary, the citation will point to half-information.

**The custom solution.** A hierarchical document parser was written from scratch. It builds a tree structure of the document — root → H1 chapters → H2 sections → H3 subsections — and handles orphan content explicitly by converting it to `introContent` nodes that are preserved with their parent section. The parser produces 78 chunks from the document. Each chunk carries full metadata: section number, title, URL path, breadcrumb trail, token count, and chunk ID. Average chunk size is 1,547 tokens — large enough to preserve clinical context, small enough to be meaningful for retrieval.

The notebook output shows the document tree structure printed in full, section by section. You can see every chunk, its token count, and whether orphan content was preserved. The validation step confirms that all 8 orphan sections from the source document appear as `introContent` in the chunk output. Nothing was lost.

**Why this matters analytically.** The decision to build a custom chunker instead of using an off-the-shelf tool is the same kind of decision a good data analyst makes when they realise that a standard transformation is silently dropping rows. You notice the problem, you understand why the tool produces it, and you build around it. The tool is not wrong for its intended purpose — it is wrong for this purpose.

The chunking notebook also calculates token distributions. The 120,679-token document compresses to 78 meaningful retrieval units. Token counts are verified before embedding to ensure no chunk exceeds the Jina v4 model's context window. This is the kind of upstream validation that prevents silent failures downstream.

### Notebook 04: Vector Store

ChromaDB was configured with HNSW indexing and cosine distance. Three specific decisions here are worth explaining in detail.

**Why HNSW, not brute-force.** HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbour algorithm. For 78 chunks, brute-force search would be fast enough. But HNSW was chosen anyway because the architecture should be production-ready and scale-ready. HNSW is configured with M=16 connections per node, ef_construction=200 for build accuracy, and ef_search=100 for query speed. This achieves greater than 95% recall versus brute-force at approximately 50 milliseconds per query.

**Why cosine distance, not L2.** ChromaDB's default distance metric is L2 (Euclidean distance). L2 measures the absolute distance between two vectors in high-dimensional space. For embedding vectors, this is the wrong metric because it is sensitive to vector magnitude. A longer document section produces a higher-magnitude embedding. L2 distance will systematically rank longer sections as more similar to any query, regardless of semantic relevance.

Cosine distance measures angular similarity — the angle between two vectors, normalised to remove magnitude. A 54-token definition section and a 3,000-token treatment protocol can be compared fairly. The question cosine distance answers is: *what does this mean?* The question L2 distance answers is: *how big is this?* For semantic search, you want the first question.

**Jina Embeddings v4.** The embedding model is Jina AI's v4 model, which produces 8,192-dimensional vectors. The high dimensionality matters because it allows the model to capture both semantic meaning and keyword presence simultaneously. In practice, this means a query for "DKA management" retrieves sections about "diabetic ketoacidosis treatment" (semantic match) and sections that contain the exact acronym "DKA" (keyword match). In a standard lower-dimensional embedding model, you would need a hybrid BM25/dense retrieval setup to achieve this. Jina v4's dimensionality eliminates that requirement and simplifies the architecture.

### Notebook 05: Retrieval Validation

The retrieval pipeline was validated with a Gradio interface built into the notebook. Queries were tested at three similarity thresholds: 0.3, 0.4, and 0.5.

At 0.3, off-topic results appeared — chunks that contained some overlapping medical vocabulary but were not genuinely relevant to the query. At 0.5, genuinely relevant paraphrases were excluded — a query about "blood sugar monitoring" would miss a highly relevant section that used the phrasing "glycaemic control targets". The 0.4 threshold was selected as the empirically validated operating point for this domain.

This threshold validation is the kind of work that distinguishes a system built to work from a system built to be demonstrated. The threshold is not a hyperparameter chosen from documentation — it was found by running real queries and reading the results.

### Notebook 06: LangGraph Orchestration

The agentic workflow was prototyped here. The initial development used local Ollama models (`kimi-k2-thinking:cloud`) because local models allow rapid iteration without per-call API costs. The architecture was designed from the beginning to be model-agnostic — LangChain's abstraction layer means switching the LLM is a one-line change. In production, this became Claude Haiku 4.5. The same workflow, the same prompts, the same state machine — different model at the invocation point.

The workflow is a LangGraph `StateGraph` with three nodes:

```
START → classify_query → route_by_classification
                              ↓
            ┌─────────────────┼
            ↓                 ↓
        direct response   retrieve_chunks → generate_answer → END
            ↓
           END
```

The classifier runs first on every query. It produces a structured `ClassifierOutput` — not a free-form string, but a Pydantic model with typed fields: `route` (either `"direct"` or `"retrieve"`), `safety` (`"safe"`, `"unsafe"`, or `"irrelevant"`), `intent` (the query rephrased for retrieval), `direct_response` (for non-substantive queries), and `status_message` (streamed to the user during processing).

If the route is `"direct"`, the workflow ends. If it is `"retrieve"`, retrieval runs programmatically — no LLM call — using the rephrased intent from the classifier. Then generation runs, producing a cited answer.

This means:
- A greeting costs one LLM call.
- An irrelevant query costs one LLM call.
- An unsafe medical query (asking for patient-specific advice) costs one LLM call and returns a disclaimer plus cited general guidance.
- A substantive clinical query costs two LLM calls.

The system never costs more than two LLM calls per query. This is not an accident — it is the result of a deliberate design choice to consolidate all classification logic (intent understanding, safety check, relevance check, routing decision) into a single structured output from a single model invocation.

### Notebook 07: Production Refinement

The final notebook refined the workflow — debugging conditional routing, fixing edge cases in citation extraction, validating the state machine against a range of query types. The Gradio interface was used for live testing throughout.

---

## The Production System

The production backend is a FastAPI application that refactors the notebook code into a clean module structure. It is not a rewrite — the same LangGraph graph, the same Pydantic schemas, the same ChromaDB reader, the same prompt templates. The backend adds production concerns: session management, streaming via NDJSON, CORS configuration, request size limiting, retry logic with exponential backoff, LangSmith observability integration, and a health endpoint.

### Session Management

Conversation history is maintained in memory per session. Sessions are identified by UUID and expire after 30 minutes of inactivity. A background cleanup task runs every 5 minutes to evict expired sessions. This is stateful — the LangGraph workflow receives the full message history on every invocation, which means follow-up questions work correctly. "What about in pregnancy?" after a conversation about Type 2 management will retrieve sections on gestational diabetes, not generic results.

### Streaming

The API streams responses as newline-delimited JSON. The frontend receives events in sequence:

1. `{"type": "status", "message": "I am getting the relevant resources to answer: [intent]"}` — from the classifier, as soon as routing is decided
2. `{"type": "status", "message": "Found 5 relevant sources. Generating answer..."}` — from retrieval
3. `{"type": "stream_start"}` — generation begins
4. `{"type": "token", "content": "..."}` — one per LLM output token, streamed live
5. `{"type": "stream_end", "sources": [...], "session_id": "..."}` — final event with sources

The user sees the status messages while the system is working. When generation begins, they see words appearing one at a time. This transforms the experience from "waiting for a black box" to "watching a clinician think out loud."

The backend implements a streaming fallback: if token-by-token streaming fails, it falls back to a single `invoke` call and returns the full response as a block. This makes the system resilient to streaming errors without degrading to a failure state.

### Citation Extraction

The citation system is built around a specific extraction pattern. The generator is instructed to cite sources using numbered references — `[1]`, `[2]`, `[3]` — corresponding to the numbered sources in its context. After generation, a regex pattern (`\[(\d+)\](?!\()`) extracts every citation number from the response text. The system validates each number against the range of available chunks and maps it to the corresponding `Source` object (title, URL, chunk ID). Only sources that appear in the response are returned to the frontend.

This prevents citation bloat — the problem where a system returns all retrieved sources regardless of whether they were used. If the generator cited sources 1 and 3 from a pool of 5 retrieved chunks, only sources 1 and 3 appear in the response. The user is not shown irrelevant sources that happened to be retrieved.

The frontend converts numbered citations into clickable links that navigate to the corresponding section in the document viewer. When a user clicks `[2]` in an answer, they are taken directly to the section of the guidelines that the model drew from. They can read the original text, verify the interpretation, and check the surrounding context.

### The Frontend

The React frontend provides four view modes: split view (document alongside chat), chat-only, document-only, and focused chat. The sidebar renders the full document hierarchy — all 8 chapters, their sections, and their subsections — generated from the `document_structure.json` produced in the chunking notebook. Navigation is hierarchical and expandable.

The chat panel supports real-time streaming, auto-scroll with user override (the user can scroll up to read while generation continues), and a "clear conversation" function that resets both the session and the conversation history. The search bar is disabled when the chat panel is open, directing users toward the conversational interface.

The frontend is production-ready, not a prototype. It handles streaming edge cases, touch events on mobile, resizable chat panels via drag, and graceful error states.

---

## Key Engineering Decisions — The Trade-offs

Every significant decision in this project had a reason. These are not implementation details — they are the choices that separate a system that works from a system that merely runs.

**Two LLM calls, not four.** Early multi-step RAG architectures make separate LLM calls for relevance classification, safety classification, intent rephrasing, and generation. Four calls per substantive query means four sources of latency, four sources of cost, and four things that can fail. This system consolidates the first three into a single structured output. The trade-off is prompt complexity — the classifier prompt is longer and more carefully constructed than any of the three individual prompts would be. That trade-off is worth it.

**Programmatic retrieval.** After classification, retrieval is pure Python — a ChromaDB query using the rephrased intent. There is no LLM call in the retrieval step. Some architectures use a model to decide which retrieval strategy to use (keyword, semantic, hybrid). For a single-collection, single-domain knowledge base, that overhead is unnecessary. The classifier's rephrased intent is sufficient.

**Pydantic for all LLM outputs.** Raw LLM output is a string. A string can be anything. Pydantic models make the expected structure explicit, enforce it at runtime, and produce serialisable dictionaries for the API response. Every field in `ClassifierOutput` and `GeneratorOutput` has a type, a description, and validation logic. When something goes wrong, the error message tells you exactly which field failed and why.

**Cosine over L2.** This is a decision that most practitioners using ChromaDB would miss because L2 is the default and it works well enough for many use cases. For a document with sections ranging from 54 to 7,011 tokens, it does not work well enough. The switch to cosine was made because the retrieval results were examined and the default produced length-biased results. That kind of empirical validation before committing to a configuration is the mark of someone who tests rather than assumes.

**Unsafe queries still retrieve.** A query that asks for patient-specific advice (unsafe) still routes to retrieval rather than a blanket refusal. The system provides cited general guidance from the guidelines and appends a disclaimer. This is a better user experience — the user gets useful information about the clinical area they are asking about, plus a clear signal that they need clinical judgment for their specific situation. A blanket refusal serves no one.

**LangChain as the abstraction layer.** The entire system is built on LangChain's expression language, which means the LLM can be replaced in a single configuration change. The development history demonstrates this — the same workflow that ran locally on an Ollama model in notebook 06 runs on Claude Haiku in production. If Anthropic changes its pricing, or if a new open-weight model outperforms Haiku on this domain, the migration is one line in `config.py`.

---

## What This Project Reveals About How I Work

The notebooks are not just code — they are a thinking record. Reading them in sequence, you can trace the analytical method:

**Understand before building.** The document structure was mapped fully before the first chunk was produced. Token distributions were calculated. The hierarchy was printed as a tree. The failure modes of alternative approaches were identified and documented. This is how a good analyst treats any unfamiliar dataset — you do not start cleaning before you understand what you have.

**Validate at every stage.** The chunking notebook confirms that all 8 orphan sections were preserved. The vector store notebook compares L2 and cosine results on the same queries. The retrieval notebook runs real clinical queries at three threshold values and examines the results. No stage was marked complete based on the code running — it was marked complete based on the output making sense.

**Cost and efficiency are design inputs, not afterthoughts.** The two-LLM-call ceiling was established as a constraint before the architecture was finalised. Streaming was implemented because response time perception matters. The retrieval threshold was chosen to minimise noise without excluding signal. These are not performance optimisations added later — they are structural choices made early.

**The BI mindset in an AI context.** Business intelligence is about turning data into decisions. The question at every stage of this project was: does this produce the right information for the right person at the right moment? That question drives the document chunking (does this chunk contain complete, citable information?), the retrieval threshold (does this result set contain relevant material without noise?), the citation system (can this answer be verified?), and the streaming interface (does the user know what is happening while they wait?). These are not software engineering questions — they are information design questions.

---

## Technical Specification

| Component | Technology | Notes |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Typed state, conditional routing |
| LLM | Claude Haiku 4.5 (Anthropic) | Via LangChain abstraction |
| Embeddings | Jina Embeddings v4 | 8,192-dimensional, async httpx with connection pooling |
| Vector Store | ChromaDB (persistent) | HNSW, cosine distance, M=16, ef=100 |
| API | FastAPI | Streaming NDJSON, CORS, session management |
| Frontend | React + Vite | React Router, ReactMarkdown, remark-gfm |
| Observability | LangSmith (optional) | Toggle via env var |
| Validation | Pydantic v2 | All LLM outputs and API inputs |
| Source document | Kenya National Diabetes Guidelines 2018 | 157 pages, 78 chunks, 120,679 tokens |
| Chunk count | 78 | Average 1,547 tokens/chunk |
| Retrieval | Top-5, cosine ≥ 0.4 | ~50ms per query |
| LLM calls | 1–2 per query | 1 for non-substantive, 2 for clinical queries |
| Streaming | Token-by-token | Fallback to full-response invoke |
| Session TTL | 30 minutes | Background cleanup every 5 min |

---

## Skills Demonstrated

**AI Engineering**
- LangGraph stateful workflow design and conditional routing
- Pydantic v2 structured output schemas
- Prompt engineering for classification and generation
- Retrieval-Augmented Generation architecture
- Vector database configuration (ChromaDB, HNSW, cosine distance)
- Embedding model selection and trade-off analysis
- Streaming API implementation (FastAPI + NDJSON)
- LangChain abstraction for model portability
- LangSmith observability integration
- Async programming (asyncio, httpx connection pooling)

**Data and Analytical Thinking**
- Structured document parsing and hierarchy extraction
- Custom text chunking with completeness validation
- Token distribution analysis and constraint-driven design
- Empirical threshold selection through output examination
- Failure mode identification and mitigation
- Iterative notebook-based exploration methodology

**Software Engineering**
- FastAPI production application (lifespan, middleware, routing)
- Session state management with expiry and cleanup
- Retry logic with exponential backoff
- React frontend with streaming state management
- Environment-based configuration management
- Separation of production and development dependencies

**Domain Understanding**
- Medical information safety requirements
- Clinical citation standards and verifiability
- Knowledge management for structured regulatory documents

---

## What This Demonstrates for Employers and Clients

**For organisations evaluating AI capability:** This project is evidence of end-to-end AI engineering competency — not consuming AI APIs, but understanding the architecture behind them and making deliberate choices at each layer. The system handles edge cases (orphan content, unsafe queries, retrieval fallback, streaming errors), not just the happy path.

**For organisations evaluating analytical capability:** The decision-making visible in the notebooks — understanding data structure before processing it, validating outputs empirically, choosing tools based on documented reasoning rather than convention — is the same mindset that produces reliable dashboards, trustworthy models, and defensible recommendations. The domain changes. The method does not.

**For organisations asking whether AI can be integrated into their processes:** This project is a direct answer. Take a complex expert document that most employees find difficult to navigate. Add a natural language interface. Add citations that allow verification. Add streaming so the system feels responsive. The result is not a replacement for expertise — it is a reduction in the friction between expertise and the people who need it. That principle applies to policy documents, product catalogues, legal contracts, financial reports, compliance manuals, and internal knowledge bases. The template is repeatable.

---

## What Comes Next

The system is built for a single document and a single collection. The architecture is ready to scale:

- **Multi-document support**: Adding a second collection (a different set of clinical guidelines, a different regulatory domain) requires adding a collection to ChromaDB and a routing decision in the classifier.
- **Model evaluation**: The LangSmith integration is already wired. A systematic evaluation harness would allow A/B testing of prompts, threshold values, and models against a ground-truth query set.
- **Production deployment**: The system is containerisable. A Docker build, environment injection, and a cloud host are the remaining steps to public availability.
- **Broader domain application**: The same pipeline — document extraction, custom chunking, vector retrieval, agentic generation, citation UI — applies to any structured expert document. The domain-specific work is in the prompts and the chunking logic. Both are replaceable.

---

*This document was produced from a code-level analysis of the full project repository, including all 7 development notebooks, the production backend, the React frontend, the test harness, and the git history.*

*Source document: Kenya National Clinical Guidelines for the Management of Diabetes Mellitus, 2nd Edition (2018). Produced by the National Diabetes Prevention and Control Program, Division of Non-Communicable Diseases, Ministry of Health, Kenya. Used for educational and demonstrative purposes.*
