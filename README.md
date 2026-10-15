# Customer Support FAQ RAG System

A full-stack, grounded Retrieval-Augmented Generation (RAG) assistant for customer support. The system combines an algorithmic RAG core, a **FastAPI** backend service, a responsive **React** chat interface, and an empirical **evaluation and threshold calibration** suite.

Built with Hugging Face's `MakTek/Customer_support_faqs_dataset`, **ChromaDB** persistent vector storage, **BAAI/bge-small-en-v1.5** embeddings, and **Groq** LLMs (`openai/gpt-oss-20b`).

> [!NOTE]
> **Current Status & Scope**: This repository contains a fully functional local service including the backend API, web frontend, conversation memory, and an 83-test offline test suite. While incorporating prompt injection defenses, heuristic validation against structural hallucinations (unsupported phone numbers, emails, links, and placeholders), and bounded in-memory session limits, running it in production requires standard enterprise infrastructure (authentication gateway, rate limiting, and TLS termination) detailed in the [Production Roadmap](#8-production-roadmap--deployment-prerequisites).

---

## Table of Contents
1. [Quick Start](#1-quick-start)
   - [Prerequisites](#prerequisites)
   - [Backend Setup & Running](#backend-setup--running)
   - [Frontend Setup & Running](#frontend-setup--running)
   - [CLI Commands](#cli-commands)
2. [Automated Test Suite](#2-automated-test-suite)
3. [Project Architecture & Structure](#3-project-architecture--structure)
4. [Core Capabilities & Design Decisions](#4-core-capabilities--design-decisions)
   - [Data Cleaning & Audit Accounting](#data-cleaning--mathematical-audit-accounting-scriptsclean_datapy)
   - [Index Lifecycle & Fingerprinting](#index-lifecycle--idempotent-fingerprinting-appingestpy)
   - [Retrieval Pipeline & Ranking Semantics](#retrieval-pipeline--ranking-semantics-appretrievalpy)
   - [Short-Term Bounded Conversation Memory](#short-term-bounded-conversation-memory-appmemorypy)
   - [Prompt Hardening & Answer Validation](#prompt-hardening--answer-validation-appgenerationpy)
5. [FastAPI Service & API Reference](#5-fastapi-service--api-reference)
   - [`GET /health`](#1-get-health)
   - [`POST /chat`](#2-post-chat)
6. [Threshold Calibration & Evaluation Workflow](#6-threshold-calibration--evaluation-workflow)
7. [Frontend Web Application](#7-frontend-web-application)
8. [Production Roadmap & Deployment Prerequisites](#8-production-roadmap--deployment-prerequisites)

---

## 1. Quick Start

### Prerequisites
- **Python**: 3.10+ (tested on Python 3.12)
- **Node.js**: 18+ (tested on Node 20) with `npm`
- **Groq API Key**: A free API key from [console.groq.com](https://console.groq.com)

---

### Backend Setup & Running

1. **Clone the repository and create a virtual environment**:
   ```powershell
   git clone https://github.com/SouliSun/Fastapi-rag-chatbot.git
   cd Fastapi-rag-chatbot
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # On macOS/Linux: source .venv/bin/activate
   ```

2. **Install Python dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```powershell
   Copy-Item .env.example .env    # On macOS/Linux: cp .env.example .env
   ```
   Open `.env` and set your Groq API key:
   ```ini
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=openai/gpt-oss-20b
   SIMILARITY_THRESHOLD=0.75
   DEFAULT_TOP_K=3
   ```

4. **Initialize data and index (Optional)**:
   The database auto-indexes on application startup or during your first query. To explicitly pre-build the vector index from the cleaned dataset, run:
   ```powershell
   python ingest.py --reindex
   ```
   To verify that ChromaDB is initialized, responsive, and reports healthy status without starting the web server:
   ```powershell
   python ingest.py --health
   ```

5. **Start the FastAPI backend service**:
   ```powershell
   uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
   ```
   - **Service Endpoint**: `http://127.0.0.1:8000`
   - **Interactive API Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`
   - **Health Probe**: `http://127.0.0.1:8000/health`

---

### Frontend Setup & Running

In a separate terminal window:

1. **Navigate to the frontend directory and install dependencies**:
   ```powershell
   cd frontend
   npm install
   ```

2. **Start the Vite development server**:
   ```powershell
   npm run dev
   ```
   The UI will open at `http://localhost:5173`. It connects to `http://127.0.0.1:8000` by default.

---

### CLI Commands

You can interact with the RAG core directly from the command line without running the web service:

* **Interactive Support Chat**:
  ```powershell
  python ingest.py
  ```
* **Single Query Evaluation**:
  ```powershell
  python ingest.py --query "How do I return an item?"
  ```
* **Force Vector Index Rebuild**:
  ```powershell
  python ingest.py --reindex
  ```
* **Run Threshold Calibration**:
  ```powershell
  python scripts/calibrate_threshold.py
  ```

---

## 2. Automated Test Suite

The project includes an offline test suite executed with **pytest**. Embedding models and Groq LLM network calls are mocked with deterministic fixtures, allowing the entire suite to run offline in seconds.

### Running the Tests
```powershell
pytest -v tests
```

### Coverage (83 tests passed across 8 suites):
- **`tests/test_api.py`** (19 tests): FastAPI `/health` (healthy and 503 degraded states), `/chat` single-turn and multi-turn success, low-similarity fallbacks, query length bounds (HTTP 400), blank query validation (HTTP 422), CORS local origin validation, error boundaries, secret leakage prevention, conversation ID auto-generation and preservation, conversation isolation, and lifespan initialization.
- **`tests/test_memory.py`** (8 tests): Conversation ID regex and length validation, ID generator uniqueness, turn-level FIFO eviction, inactivity TTL expiration, bounded LRU conversation capacity eviction, expired session pruning on access, and multithreaded concurrency safety.
- **`tests/test_calibration_and_evaluation.py`** (11 tests): Deterministic dataset partitioning, split ratio enforcement, single-item category diagnostics, multi-item balance, decoupled acceptance vs. retrieval, Hit@k and MRR calculation against source FAQs, zero-category divide-by-zero protection, and `None` vs `0.0` metric semantics for accepted queries.
- **`tests/test_clean_data.py`** (11 tests): Missing data rejection, whitespace normalization, conflict resolution policies (`keep_first`, `keep_last`, `drop`, `error`), deterministic stable IDs, simultaneous exact-duplicate & conflict accounting.
- **`tests/test_indexing.py`** (7 tests): First-time indexing, manifest fingerprinting, idempotence, answer modifications, row additions, row deletions, row order invariance, and embedding model changes.
- **`tests/test_retrieval.py`** (7 tests): Empty queries, non-strings, query length bounds, invalid $k$, $k$-clamping, empty collections, cosine distance conversion, threshold acceptance/rejection, and malformed Chroma results.
- **`tests/test_generation.py`** (16 tests): Empty retrieval fallback, structured JSON context formatting, prompt tag sanitization, output validation (refusals, placeholders, fabricated phone/email/URL), validator limitations documentation, authentication error fast-fail, transient retry recovery, retry exhaustion, and query privacy logging.
- **`tests/test_regression_and_security.py`** (4 tests): API key masking in `repr` and logs, backward compatibility of `search()` and `generate_answer()`, health check status reporting, and lazy proxy import isolation.

---

## 3. Project Architecture & Structure

The repository is organized into a clean, modular structure:

```
ChatBot/
├── app/                           # Core application package
│   ├── __init__.py                # Package exports
│   ├── api.py                     # FastAPI application with /health and /chat endpoints
│   ├── config.py                  # Centralized typed AppConfig with secret masking and validation
│   ├── retrieval.py               # Input validation, ChromaDB querying, distance conversion & diagnostics
│   ├── generation.py              # LLM generation with JSON payloads, output checks, and bounded retries
│   ├── memory.py                  # Bounded in-memory conversation store with LRU eviction and TTL
│   └── ingest.py                  # Idempotent index lifecycle, fingerprint manifest & CLI orchestration
├── data/                          # Dataset files and evaluation ground-truth benchmarks
│   ├── cleaned_faqs.csv           # Cleaned and deduplicated FAQ records
│   ├── eval_dev_tuning_set.json   # Development/tuning query set
│   ├── eval_held_out_test_set.json# Held-out evaluation query set
│   ├── ragas_eval_set.json        # Master labeled evaluation dataset
│   └── threshold_calibration_report.csv # Historical sweep metrics across thresholds
├── scripts/                       # Offline calibration, cleaning, and evaluation scripts
│   ├── __init__.py                # Scripts package init
│   ├── calibrate_threshold.py     # Empirical threshold calibration with frozen dev/test splits
│   ├── clean_data.py              # Pure data cleaning, normalization, conflict resolution & stable IDs
│   ├── evaluate1.py               # RAGAS offline evaluation runner
│   └── experiment_qa_representation.py # Side-by-side benchmark of Question-Only vs Q+A representation
├── frontend/                      # Customer-support chat UI (React, Vite, TypeScript, Tailwind, shadcn/ui)
│   ├── src/                       # Chat interface, components, API client, and state management
│   ├── package.json               # Frontend dependencies and scripts
│   └── vite.config.ts             # Vite bundler configuration
├── tests/                         # 83-test automated test suite (pytest)
│   ├── conftest.py                # Shared fixtures with fast deterministic mock embeddings
│   ├── test_api.py                # FastAPI endpoint, CORS, validation, and lifespan tests
│   ├── test_memory.py             # In-memory store, turn bounds, LRU eviction, and TTL tests
│   ├── test_calibration_and_evaluation.py # Calibration splitting, metric decoupling tests
│   ├── test_clean_data.py         # Data cleaning, deduplication, and conflict policy tests
│   ├── test_indexing.py           # Index idempotence, manifest fingerprinting, and rebuild tests
│   ├── test_retrieval.py          # Input validation, cosine conversion, and threshold tests
│   ├── test_generation.py         # Prompt sanitization, retry backoff, and validation tests
│   └── test_regression_and_security.py # Key masking, backward compatibility, and lazy imports
├── .github/workflows/ci.yml       # GitHub Actions CI (Python test suite + Frontend build)
├── pytest.ini                     # Pytest configuration with pythonpath (., app, scripts)
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules for secrets, DBs, logs, and build artifacts
└── LICENSE                        # MIT License
```

> [!TIP]
> **Backwards Compatibility**: Thin root modules (`api.py`, `ingest.py`, `config.py`, etc.) proxy directly into `app/` and `scripts/`, allowing CLI commands (`uvicorn api:app`, `python ingest.py`) and standard package imports (`from app.api import app`) to work seamlessly.

---

## 4. Core Capabilities & Design Decisions

### Data Cleaning & Mathematical Audit Accounting (`scripts/clean_data.py`)
* **Strict Validation**: Rejects missing `question` or `answer` columns; strips nulls, blanks, and whitespace-only values.
* **Whitespace Normalization**: Collapses internal tabs, newlines, and multi-spaces prior to duplicate checking.
* **Separation of Exact Duplicates & Conflicts**: Exact duplicates (`subset=['question', 'answer']`) are deduplicated first. Conflicting questions mapping to distinct answers are handled separately according to configurable policies (`keep_first`, `keep_last`, `drop`, `error`).
* **Audit Accounting Invariant**: The cleaning report satisfies the exact mathematical identity:
  $$\text{total\_loaded} = \text{missing\_removed} + \text{exact\_dups\_removed} + \text{conflict\_rows\_pruned} + \text{total\_retained}$$
* **Deterministic Stable IDs**: Questions receive content-addressed stable hashes: `faq_{sha256(normalized_question)[:12]}`.
* **Lazy Module Imports**: `import clean_data` and `import ingest` execute in milliseconds without making external HTTP calls or opening vector database connections. Dataset access via `df` uses a transparent `_LazyDFProxy`.

---

### Index Lifecycle & Idempotent Fingerprinting (`app/ingest.py`)
* **Cryptographic Fingerprinting**: `compute_dataset_fingerprint()` hashes the normalized question-answer pairs, embedding model name, distance metric, and schema version.
* **Idempotent Ingestion**: Skips re-indexing if the stored manifest (`index_manifest.json`) matches the computed fingerprint and collection record count.
* **Safe Rebuilds**: Modifying an answer, adding/deleting rows, or changing the embedding model triggers an automatic clean rebuild, eliminating stale or orphaned records.

---

### Retrieval Pipeline & Ranking Semantics (`app/retrieval.py`)
* **Validation & Clamping**: Rejects empty queries, non-strings, and queries exceeding `MAX_QUERY_LENGTH` (default 1000 characters). Clamps requested $k$ to collection size.
* **Distance to Similarity Conversion**: Documents ChromaDB's cosine distance $d \in [0, 2]$ and converts it explicitly to similarity:
  $$\text{similarity} = 1 - d$$
* **Ranking Signal Only**: Cosine similarity is treated strictly as an uncalibrated **retrieval ranking signal**, **not** an answer probability or confidence value.
* **Unified Acceptance Policy**: `filter_relevant()` acts as the single source of truth for both retrieval diagnostics and generation fallbacks.

---

### Short-Term Bounded Conversation Memory (`app/memory.py`)
Supports natural multi-turn customer support interactions (e.g., *"How do I track my order?"* followed by *"Can I do that on mobile?"*) while maintaining factual grounding:

* **Strict Grounding Boundary**: Conversation history is provided to the generator and retriever solely to resolve pronouns, ellipsis, and context from recent turns. Retrieved FAQ entries from ChromaDB remain the **single source of truth** for factual answers. If no FAQ passes the configured similarity threshold, the system immediately returns the fallback message rather than attempting to synthesize answers from previous chat turns.
* **Dual-Query Contextual Retrieval**: For follow-up queries with existing history, `FAQRetriever.search_with_history()` evaluates both the standalone question and a contextualized query combining the immediate previous inquiry, merging candidates and scoring via the max similarity before applying the gating threshold.
* **Turn-Level FIFO Bounding**: Each session maintains at most `MAX_HISTORY_TURNS` (default: 5 turns). Older turns are automatically dropped in FIFO order.
* **Global LRU Capacity & TTL Eviction**: To prevent memory leaks, `ConversationStore` enforces a hard limit of `MAX_ACTIVE_CONVERSATIONS` (default: 1000) using a thread-safe `OrderedDict` LRU cache. Sessions inactive for longer than `CONVERSATION_TTL_SECONDS` (default: 3600s / 1 hour) are automatically evicted.
* **Privacy & Ephemerality**: Conversation state is stored strictly in memory (no disk or database persistence). Secrets are never stored, and user queries are not logged by default. Missing, unknown, or expired IDs gracefully start fresh sessions without throwing errors.

---

### Prompt Hardening & Answer Validation (`app/generation.py`)
* **Untrusted Data Isolation**: Retrieved FAQs are packaged inside a structured JSON payload (`<context_payload format="json">`) with explicit instructions that all context is untrusted reference data, not executable instructions.
* **Delimiter & Role Sanitization**: `sanitize_untrusted_text()` neutralizes attempts to inject delimiter tags (`</context_payload>`, `<user_question>`) or role prefixes (`system:`, `assistant:`, `developer:`).
* **Residual Prompt Injection Risk**:
  > [!WARNING]
  > Delimiter tags and system instructions are defense-in-depth mitigations, **not** an absolute cryptographic boundary. A sufficiently novel adversarial prompt injection may still evade LLM adherence. To manage residual risk, the system is designed to be completely stateless with **no execution tools, file access, database write privileges, or shell access**.
* **Output Validation**: `validate_generated_answer()` inspects LLM completions:
  - Replaces empty responses with the safe fallback message.
  - Recognizes LLM refusal patterns (e.g., *"not mentioned in the FAQ"*) and flags them cleanly.
  - Sanitizes unreplaced bracket placeholders (e.g., `[phone number]`, `[working hours]`).
  - Detects fabricated contact details (telephone numbers, email addresses, external URLs not present in context) and triggers a safety fallback.
* **Critical Limitation of Rule-Based Validation**:
  > [!IMPORTANT]
  > Heuristic post-checks catch gross structural hallucinations (unsupported phone numbers, emails, links, placeholders), but **cannot validate general factual grounding or semantic entailment**. Subtle factual distortions (e.g., altering a warranty from 1 year to 2 years, or hallucinating unlisted color options) pass through heuristic checks. General factual grounding requires downstream natural language inference (NLI) verification or human evaluation.
* **Privacy-Preserving Logging**: API keys are masked in `str()` and `repr()`. Raw user query text and extracted contact details are not logged by default (`LOG_RAW_QUERIES=false`).

---

## 5. FastAPI Service & API Reference

The service layer is implemented in `app/api.py` with Pydantic v2 validation models.

### Running with Uvicorn
```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

---

### Endpoints

#### 1. `GET /health`
Probes service readiness and verifies that ChromaDB persistent storage and embedding pipelines are responsive.

* **Response (HTTP 200 OK)**:
  ```json
  {
    "status": "healthy",
    "indexed_faqs": 89,
    "database_path": "./chroma_db",
    "collection_name": "customer_support_faqs",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "test_retrieval_ok": true,
    "error": null
  }
  ```
* **Response (HTTP 503 Service Unavailable)**: Returned if the vector database is unreachable or index initialization fails.

---

#### 2. `POST /chat`
Accepts a customer support inquiry, validates query constraints, resolves multi-turn conversational context, performs vector retrieval, filters relevant context, and generates a grounded response.

* **Request Body Schema (`ChatRequest`)**:
  ```json
  {
    "question": "How do I track my order?",
    "conversation_id": "conv_a1b2c3d4e5f67890",
    "k": 3,
    "threshold": 0.75
  }
  ```
  - `question` *(string, required)*: Customer inquiry. Blank or whitespace-only inputs return **HTTP 422**. Queries longer than `MAX_QUERY_LENGTH` (1000 characters by default) return **HTTP 400**.
  - `conversation_id` *(string, optional)*: Stable identifier for multi-turn conversational context (1 to 64 alphanumeric characters, underscores, or hyphens). If omitted or null, the service automatically generates a fresh conversation ID. Malformed IDs return **HTTP 400**. Unknown or expired IDs predictably start a new session.
  - `k` *(integer, optional)*: Number of candidates to retrieve ($1 \le k \le 20$). Defaults to `DEFAULT_TOP_K` (3).
  - `threshold` *(float, optional)*: Minimum similarity score ($0.0 \le \tau \le 1.0$). Defaults to `SIMILARITY_THRESHOLD` (0.75).

* **Response Body Schema (`ChatResponse`)**:
  ```json
  {
    "conversation_id": "conv_a1b2c3d4e5f67890",
    "answer": "You can track your order using the order number sent to your email confirmation.",
    "fallback": false,
    "fallback_reason": null,
    "contexts": [
      {
        "id": "faq_4f3b72a1e90c",
        "question": "How can I track the delivery status of my package?",
        "answer": "You can track your order using the order number sent to your email confirmation.",
        "score": 0.884,
        "distance": 0.116
      }
    ],
    "results": [ ... ],
    "latency_ms": 142.5,
    "prompt_tokens": 128,
    "completion_tokens": 34,
    "diagnostics": {
      "query_length": 25,
      "requested_k": 3,
      "effective_k": 3,
      "total_candidates": 3,
      "accepted_candidates": 1,
      "top_score": 0.884,
      "latency_ms": 14.8,
      "threshold_applied": 0.75,
      "is_accepted": true,
      "error": null
    }
  }
  ```

---

## 6. Threshold Calibration & Evaluation Workflow

The evaluation script [`scripts/calibrate_threshold.py`](scripts/calibrate_threshold.py) provides a rigorous framework for setting similarity thresholds without metric conflation or data leakage:

### 1. Decoupling Acceptance Gating from Retrieval Correctness
The evaluator separates two fundamentally distinct dimensions:
1. **Acceptance Gating Metrics** (evaluates the binary gate $\text{score} \ge \tau$):
   - **`answerable_coverage` (Recall)**: $\frac{\text{accepted answerable queries}}{\text{total answerable queries}}$. Measures what fraction of valid queries cleared the threshold instead of falsely refusing.
   - **`unanswerable_rejection_rate`**: $\frac{\text{rejected unanswerable/OOD queries}}{\text{total unanswerable/OOD queries}}$. Measures what fraction of unanswerable or out-of-scope queries were correctly rejected.
   - **`acceptance_precision`**: $\frac{\text{accepted answerable queries}}{\text{total accepted queries}}$. Measures what fraction of queries that cleared the threshold were actually answerable.
   - **`acceptance_f1`**: Harmonic mean of `acceptance_precision` and `answerable_coverage`.
2. **Retrieval Correctness Metrics** (evaluated on answerable queries against labeled `source_questions`):
   - **`Hit@1`**: Top-1 candidate matches a labeled source FAQ.
   - **`Hit@k`**: Any candidate within top-$k$ matches a labeled source FAQ.
   - **`MRR`**: Mean Reciprocal Rank of the first matching source FAQ among top-$k$.
3. **Retrieval Quality Among Accepted Queries**:
   - **`hit1_among_accepted`**: Fraction of accepted answerable queries where rank 1 matched ground truth.
   - **`strict_hit1_accepted_rate`**: Fraction of **all** answerable queries that both passed the threshold **and** retrieved the correct FAQ at rank 1 ($\frac{\text{accepted with Hit@1}}{\text{total answerable queries}}$).
   - *An accepted answerable query that retrieves the wrong FAQ counts toward coverage, but is strictly NOT counted as a retrieval hit.*

---

### 2. Frozen Development / Tuning vs. Held-Out Evaluation Splits
To ensure genuine split isolation, the master evaluation set (`data/ragas_eval_set.json`, 45 items) is partitioned once into explicit frozen files:
* **Development / Tuning Set (`data/eval_dev_tuning_set.json`)**: 33 items (73.3% of total).
  - 28 in-scope answerable queries (24 single-hop, 4 multi-hop).
  - 5 unanswerable / out-of-scope queries (4 in-domain unanswerable, 1 out-of-scope).
  - **Purpose**: Used strictly to sweep candidate thresholds ($\tau \in [0.30, 0.95]$) and observe false refusals vs. unanswerable leakage.
* **Held-Out Test Set (`data/eval_held_out_test_set.json`)**: 12 items (26.7% of total).
  - 11 in-scope answerable queries (10 single-hop, 1 multi-hop).
  - 1 in-domain unanswerable query.
  - 0 out-of-scope queries.
  - **Purpose**: Evaluated once at the selected operating threshold. **The held-out set is never used to select or tune the threshold.**

---

### 3. Evaluation Limitations
* **Out-of-Scope Split Limitation**: The master dataset contains only 1 out-of-scope query. Because this single item cannot be split across sets, it was allocated to the development set. The held-out test set contains **0 out-of-scope examples**; therefore, out-of-scope rejection performance **cannot be evaluated from the held-out split** and is reported as `N/A`.
* **Sample Size Notice**: Metrics on this small (45-query) manually authored dataset are **preliminary indicators** and cannot be claimed as an unbiased or representative measurement of production customer traffic.
* **Grounding Disclaimer**: Neither threshold acceptance nor retrieval accuracy proves that generated answers are factually grounded. Factual faithfulness requires independent natural language inference (NLI) evaluation.

Run the calibration workflow:
```powershell
python scripts/calibrate_threshold.py
```
*(Use `python scripts/calibrate_threshold.py --repartition` only if deliberately re-splitting from the master set).*

---

## 7. Frontend Web Application

A responsive customer-support chat interface built with **Vite**, **React 19**, **TypeScript**, **Tailwind CSS**, and **shadcn/ui** components located in [`frontend/`](frontend/).

### Features:
* **Live Service Status Indicator**: Periodically checks `GET /health` with a real-time status badge (green for healthy with indexed FAQ count, red for degraded/unavailable).
* **Conversational Feed**: User questions and assistant responses with timestamps, grounding indicators, and smooth auto-scroll.
* **Retrieved Sources & Diagnostics Accordion**: Restrained, collapsible telemetry drawer for assistant answers.
  > [!IMPORTANT]
  > Similarity is explicitly labeled as an uncalibrated **Retrieval Score** (cosine similarity up to 1.0), never as a probability or confidence value.
* **Multiline Chat Composer**: Auto-expanding textarea with keyboard shortcuts (<kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for newline), character counter (1000 character maximum matching backend limit), and inline validation feedback.
* **Error Recovery**: Handles network failures, HTTP 400 (query too long), HTTP 422 (blank query), and HTTP 500 cleanly with retry controls.
* **New Conversation Action**: Reset button to clear chat history and restart inquiries.
* **Environment Configuration**: Backend URL configured via `VITE_API_BASE_URL` with a local fallback to `http://127.0.0.1:8000`.

---

## 8. Production Roadmap & Deployment Prerequisites

> [!WARNING]
> **Production Notice**: While the FastAPI backend and React frontend provide a structured, tested implementation, the system is **not turnkey production-ready for public internet exposure** without standard production infrastructure:

1. **Authentication & Authorization**: The API endpoints currently have no auth layer. Deploy behind an API gateway or implement JWT / API-key bearer authentication before exposing publicly.
2. **Rate Limiting & Abuse Prevention**: Public exposure requires per-IP or per-tenant rate limiting (e.g., `slowapi` or an Nginx/Cloudflare reverse proxy) to prevent denial-of-service or LLM token exhaustion.
3. **TLS Termination & Reverse Proxy**: Run behind an enterprise reverse proxy (Nginx, Caddy, AWS ALB) with HTTPS/TLS termination and hardened HTTP security headers.
4. **Centralized Observability & Tracing**: Integrate OpenTelemetry or Prometheus metrics to monitor P50/P95/P99 latency, token usage, and retrieval cache hits.
5. **Secret Management**: Store `GROQ_API_KEY` in a cloud secrets manager (e.g. AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) rather than local `.env` files.
6. **Semantic Entailment Verification**: Rule-based validators catch overt hallucinations (unsupported phone numbers/links), but cannot detect subtle factual drift. Implement downstream NLI cross-encoders for mission-critical enterprise workflows.
