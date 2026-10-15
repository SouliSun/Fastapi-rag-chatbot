# Customer Support FAQ RAG System (Python Library & CLI)

A hardened, tested Retrieval-Augmented Generation (RAG) core built on Hugging Face's `MakTek/Customer_support_faqs_dataset`, ChromaDB persistent vector storage, `BAAI/bge-small-en-v1.5` embeddings, and Groq LLM integration.

> [!NOTE]
> **Status & Scope**: This project provides the hardened backend algorithmic core, data ingestion pipeline, retrieval/generation logic, and a testing CLI. It is **not** a turnkey production service yet: deploying to production requires wrapping it with an HTTP serving layer (e.g., FastAPI), API authentication, rate limiting, and centralized monitoring.

---

## 1. Architecture Overview

The codebase is organized into decoupled, tested modules:

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
│   ├── test_api.py                # Tests for FastAPI health probe, chat, CORS, memory, validation
│   ├── test_memory.py             # Tests for bounded LRU store, FIFO turn eviction, and TTL expiration
│   ├── test_calibration_and_evaluation.py # Tests for split determinism, ratios, Hit@k/MRR
│   ├── test_clean_data.py         # Tests for missing values, whitespace, conflicts, audit accounting
│   ├── test_indexing.py           # Tests for idempotence, updates, additions, deletions, reorders
│   ├── test_retrieval.py          # Tests for validation, k clamping, empty collections, conversions
│   ├── test_generation.py         # Tests for structured context, sanitization, output checks, retries
│   └── test_regression_and_security.py # Tests for secret masking, public interfaces, lazy imports
├── .github/workflows/ci.yml       # GitHub Actions CI (Python test suite + Frontend build)
├── pytest.ini                     # Pytest configuration with pythonpath (., app, scripts)
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules for secrets, DBs, logs, and build artifacts
└── LICENSE                        # MIT License
```

---

## 2. Implemented Capabilities & Design Decisions

### Lazy, Zero-Network Module Imports (`clean_data.py`, `ingest.py`)
* **No Import-Time Work**: `import ingest` and `import clean_data` execute in milliseconds without making external HTTP requests to Hugging Face or opening ChromaDB connections.
* **Lazy Proxy**: Dataset access via `df` is wrapped in a transparent `_LazyDFProxy` that delays dataset download and cleaning until an attribute or row is explicitly accessed.
* **Local Caching**: Cleaned FAQs are saved to a local CSV cache (`cleaned_faqs.csv`) so subsequent runs remain resilient to network degradation.

### Data Cleaning & Mathematical Audit Accounting (`clean_data.py`)
* **Strict Validation**: Rejects missing `question` or `answer` columns; strips nulls, blanks, and whitespace-only values.
* **Whitespace Normalization**: Collapses internal tabs, newlines, and multi-spaces prior to duplicate checking.
* **Separation of Exact Duplicates & Conflicts**: Exact duplicates (`subset=['question', 'answer']`) are deduplicated first. Conflicting questions mapping to distinct answers are handled separately according to configurable policies (`keep_first`, `keep_last`, `drop`, `error`).
* **Audit Accounting Invariant**: The cleaning report satisfies the exact mathematical identity:
  $$\text{total\_loaded} = \text{missing\_removed} + \text{exact\_dups\_removed} + \text{conflict\_rows\_pruned} + \text{total\_retained}$$
* **Deterministic Stable IDs**: Questions receive content-addressed stable hashes: `faq_{sha256(normalized_question)[:12]}`.

### Index Lifecycle & Synchronization (`ingest.py`)
* **Cryptographic Fingerprinting**: `compute_dataset_fingerprint()` hashes the normalized question-answer pairs, the embedding model name, distance metric, and schema version.
* **Idempotent Ingestion**: Skips re-indexing if the stored manifest matches the computed fingerprint and collection count.
* **Safe Rebuilds**: Modifying an answer, adding/deleting rows, or changing the embedding model triggers an automatic clean rebuild, eliminating stale or orphaned records.

### Retrieval Pipeline & Ranking Semantics (`retrieval.py`)
* **Validation & Clamping**: Rejects empty queries, non-strings, and queries exceeding `MAX_QUERY_LENGTH`. Clamps requested $k$ to collection size.
* **Distance to Similarity Conversion**: Documents ChromaDB's cosine distance $d \in [0, 2]$ and converts it explicitly to similarity:
  $$\text{similarity} = 1 - d$$
* **Ranking Signal Only**: Cosine similarity is treated strictly as an uncalibrated **retrieval ranking signal**, **not** a probability of answer correctness.
* **Unified Acceptance Policy**: `filter_relevant()` acts as the single source of truth for both retrieval diagnostics and generation fallbacks.

### Prompt Hardening & Answer Validation (`generation.py`)
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

## 3. Threshold Calibration & Evaluation Workflow

### 1. Decoupling Acceptance Gating from Retrieval Correctness
The evaluation script [`calibrate_threshold.py`](calibrate_threshold.py) separates two fundamentally distinct dimensions:
1. **Acceptance Gating Metrics** (evaluates the binary gate $\text{score} \ge \tau$):
   - **`answerable_coverage` (Recall)**: $\frac{\text{accepted answerable queries}}{\text{total answerable queries}}$. Measures what fraction of valid queries cleared the threshold instead of falsely refusing.
   - **`unanswerable_rejection_rate`**: $\frac{\text{rejected unanswerable/OOD queries}}{\text{total unanswerable/OOD queries}}$. Measures what fraction of unanswerable or out-of-scope queries were correctly rejected.
   - **`acceptance_precision`**: $\frac{\text{accepted answerable queries}}{\text{total accepted queries}}$. Measures what fraction of queries that cleared the threshold were actually answerable (as opposed to leaking unanswerable/OOD queries).
   - **`acceptance_f1`**: Harmonic mean of `acceptance_precision` and `answerable_coverage`.
2. **Retrieval Correctness Metrics** (evaluated on answerable queries against labeled `source_questions`):
   - **`Hit@1`**: Top-1 candidate matches a labeled source FAQ.
   - **`Hit@k`**: Any candidate within top-$k$ matches a labeled source FAQ.
   - **`MRR`**: Mean Reciprocal Rank of the first matching source FAQ among top-$k$.
3. **Retrieval Quality Among Accepted Queries**:
   - **`hit1_among_accepted`**: Fraction of accepted answerable queries where rank 1 matched ground truth.
   - **`strict_hit1_accepted_rate`**: Fraction of **all** answerable queries that both passed the threshold **and** retrieved the correct FAQ at rank 1 ($\frac{\text{accepted with Hit@1}}{\text{total answerable queries}}$).
   - *An accepted answerable query that retrieves the wrong FAQ counts toward coverage, but is strictly NOT counted as a retrieval hit.*

### 2. Frozen Development / Tuning vs. Held-Out Evaluation Splits
To ensure genuine split isolation, the master evaluation set (`ragas_eval_set.json`, 45 items) is partitioned once into explicit frozen files:
* **Development / Tuning Set (`eval_dev_tuning_set.json`)**: 33 items (73.3% of total).
  - 28 in-scope answerable queries (24 single-hop, 4 multi-hop).
  - 5 unanswerable / out-of-scope queries (4 in-domain unanswerable, 1 out-of-scope).
  - **Purpose**: Used strictly to sweep candidate thresholds ($\tau \in [0.30, 0.95]$) and observe false refusals vs. unanswerable leakage.
* **Held-Out Test Set (`eval_held_out_test_set.json`)**: 12 items (26.7% of total).
  - 11 in-scope answerable queries (10 single-hop, 1 multi-hop).
  - 1 in-domain unanswerable query.
  - 0 out-of-scope queries.
  - **Purpose**: Evaluated once at the selected operating threshold. **The held-out set is never used to select or tune the threshold.**

### 3. Missing Held-Out Out-of-Scope Coverage & Small Strata Limitations
* **Out-of-Scope Split Limitation**: The master dataset contains only 1 out-of-scope query. Because this single item cannot be split across sets, it was allocated to the development set. The held-out test set contains **0 out-of-scope examples**; therefore, out-of-scope rejection performance **cannot be evaluated from the held-out split** and is reported as `N/A`.
* **Sample Size Notice**: Metrics on this small (45-query) manually authored dataset are **preliminary indicators** and cannot be claimed as an unbiased or representative measurement of production customer traffic.
* **Grounding Disclaimer**: Neither threshold acceptance nor retrieval accuracy proves that generated answers are factually grounded. Factual faithfulness requires independent natural language inference (NLI) evaluation.

Run the calibration workflow:
```powershell
python calibrate_threshold.py
```
*(Use `python calibrate_threshold.py --repartition` only if deliberately re-splitting from the master set).*

---

## 4. Setup & Running

### Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment Configuration
```powershell
Copy-Item .env.example .env
```
Populate `.env` with your credentials:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
SIMILARITY_THRESHOLD=0.75
DEFAULT_TOP_K=3
```

### Running Commands

* **System Health Check (CLI)**:
  ```powershell
  python ingest.py --health
  ```
* **Interactive CLI**:
  ```powershell
  python ingest.py
  ```
* **Single Query via CLI**:
  ```powershell
  python ingest.py --query "How do I return an item?"
  ```
* **Force Reindex**:
  ```powershell
  python ingest.py --reindex
  ```
* **Calibrate Thresholds**:
  ```powershell
  python calibrate_threshold.py
  ```

---

## 5. FastAPI Service (`api.py`)

A structured, async HTTP serving layer implemented with **FastAPI** and **Pydantic v2**:

### Running the API Locally with Uvicorn

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Once running, interactive OpenAPI documentation is available at:
`http://127.0.0.1:8000/docs`

### API Endpoints

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
  - `question` *(string, required)*: Customer inquiry. Blank or whitespace-only inputs are rejected with **HTTP 422**. Queries longer than `MAX_QUERY_LENGTH` (1000 characters by default) are rejected with **HTTP 400**.
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

### Short-Term Bounded Conversation Memory (`memory.py`)

The service provides an ephemeral, in-memory conversation memory manager that supports natural multi-turn follow-up interactions (e.g., *"How do I track it?"* followed by *"Can I do that on mobile?"*) without sacrificing factual grounding:

1. **Strict Grounding Boundary**: Chat history is provided to the generator and retriever solely to resolve pronouns, ellipsis, and context from recent turns. Retrieved FAQ entries from ChromaDB remain the **single source of truth** for factual answers. If no FAQ passes the configured similarity threshold, the system immediately returns the configured fallback message rather than attempting to synthesize answers from previous turns.
2. **Dual-Query Contextual Retrieval**: For follow-up queries with existing history, `FAQRetriever.search_with_history()` evaluates both the standalone question and a contextualized query combining the immediate previous inquiry, merging candidates and scoring via the max similarity before applying the gating threshold.
3. **Turn-Level FIFO Bounding**: Each session maintains at most `MAX_HISTORY_TURNS` (default: 5 turns). Older turns are automatically dropped in FIFO order.
4. **Global LRU Capacity & TTL Eviction**: To prevent memory leaks, `ConversationStore` enforces a hard limit of `MAX_ACTIVE_CONVERSATIONS` (default: 1000) using a thread-safe `OrderedDict` LRU cache. Sessions inactive for longer than `CONVERSATION_TTL_SECONDS` (default: 3600s / 1 hour) are automatically evicted.
5. **Privacy & Ephemerality**: Conversation state is stored strictly in memory (no disk or database persistence). Secrets are never stored, and user queries are not logged by default. Missing, unknown, or expired IDs gracefully start fresh sessions without throwing errors.

### Reliability & Security Architecture

1. **Lifespan Startup Management**: ChromaDB and the default retriever/generator singletons are initialized once during the FastAPI `lifespan` startup cycle, rather than re-indexing or reloading models per HTTP request.
2. **Error Boundaries & Safe Fallbacks**: Provider timeouts, ChromaDB connection errors, and generation failures are trapped gracefully; the API returns safe fallback responses rather than crashing or exposing tracebacks.
3. **Information Disclosure Prevention**: Secrets such as `GROQ_API_KEY` are strictly masked and never returned in API payloads or serialized into error structures. Unhandled server exceptions return generic 500 error messages.
4. **Privacy-Preserving Logs**: User query strings are not logged by default (`LOG_RAW_QUERIES=false`).
5. **CORS Configured for Local Development**: Restricted strictly to local development origins (`http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`), with wildcard credentials disabled.

---

## 6. Frontend Web Application (`frontend/`)

A responsive customer-support chat interface built with **Vite**, **React 19**, **TypeScript**, **Tailwind CSS**, and **shadcn/ui** components.

### Frontend Features
* **Live Service Status Indicator**: Periodically checks `GET /health` with a real-time status badge (green for healthy with indexed FAQ count, red for degraded/unavailable).
* **Conversational Feed**: User questions and assistant responses with timestamps, grounding indicators, and smooth auto-scroll.
* **Retrieved Sources & Diagnostics Accordion**: Restrained, collapsible telemetry drawer for assistant answers.
  > [!IMPORTANT]
  > Similarity is explicitly labeled as an uncalibrated **Retrieval Score** (cosine similarity up to 1.0), never as a probability or confidence value.
* **Multiline Chat Composer**: Auto-expanding textarea with keyboard shortcuts (<kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for newline), character counter (1000 character maximum matching backend limit), and inline validation feedback.
* **Error Recovery**: Handles network failures, HTTP 400 (query too long), HTTP 422 (blank query), and HTTP 500 cleanly with retry controls.
* **New Conversation Action**: Reset button to clear chat history and restart inquiries.
* **Environment Configuration**: Backend URL configured via `VITE_API_BASE_URL` with a local fallback to `http://127.0.0.1:8000`.

### Running the Frontend Locally

```powershell
# In a separate terminal
cd frontend
npm install
npm run dev
```

The frontend will start at:
`http://127.0.0.1:5173`

---

## 7. Automated Test Suite

The project includes an offline, automated test suite executed with pytest (mocking the Groq API and embedding models for zero-network execution):

```powershell
.\.venv\Scripts\pytest.exe -v tests
```

**Result**: 83 passed in ~7.1s across 8 test suites:
- `tests/test_api.py`: FastAPI `/health` (healthy and 503 degraded states), `/chat` single-turn and multi-turn success, low-similarity fallbacks, query length bounds (HTTP 400), blank/whitespace query validation (HTTP 422), malformed schema rejection, CORS local origin validation, provider error boundaries without service crashes, secret leakage prevention, conversation ID auto-generation and preservation, conversation isolation, and lifespan singleton initialization.
- `tests/test_memory.py`: Conversation ID regex and length validation, ID generator uniqueness and formatting, turn-level FIFO eviction, inactivity TTL expiration, bounded LRU conversation capacity eviction, expired session pruning on access, and multithreaded concurrency safety.
- `tests/test_calibration_and_evaluation.py`: Deterministic partitioning, ratio enforcement, single-item category diagnostics, multi-item balance, decoupled acceptance vs. retrieval, Hit@k and MRR calculation against source FAQs, zero-category divide-by-zero protection, `None` vs `0.0` metric semantics for accepted queries, and category absence reporting.
- `tests/test_clean_data.py`: Missing data, whitespace normalization, conflict resolution policies, deterministic IDs, simultaneous exact-duplicate & conflict accounting.
- `tests/test_indexing.py`: First-time indexing, manifest fingerprinting, idempotence, answer modifications, row additions, row deletions, row order invariance, embedding model changes.
- `tests/test_retrieval.py`: Empty queries, non-strings, query length bounds, invalid $k$, $k$-clamping, empty collections, cosine distance conversion, threshold acceptance/rejection, malformed Chroma results.
- `tests/test_generation.py`: Empty retrieval fallback, structured JSON context formatting, prompt tag sanitization, output validation (refusals, placeholders, fabricated phone/email/URL), validator limitations documentation, authentication error fast-fail, transient retry recovery, retry exhaustion, query privacy logging.
- `tests/test_regression_and_security.py`: API key masking in `repr` and logs, backward compatibility of `search()` and `generate_answer()`, health check status reporting, lazy proxy import isolation.

---

## 8. Production Roadmap & Deployment Prerequisites

> [!WARNING]
> **Production Notice**: While `api.py` and the chat frontend provide a structured, tested full-stack implementation, the system is **not turnkey production-ready for public internet exposure** without standard production infrastructure:

1. **Authentication & Authorization**: The API endpoints currently have no auth layer. Deploy behind an API gateway or implement JWT / API-key bearer authentication before exposing publicly.
2. **Rate Limiting & Abuse Prevention**: Public exposure requires per-IP or per-tenant rate limiting (e.g., `slowapi` or an Nginx/Cloudflare reverse proxy) to prevent denial-of-service or LLM token exhaustion.
3. **TLS Termination & Reverse Proxy**: Run behind an enterprise reverse proxy (Nginx, Caddy, AWS ALB) with HTTPS/TLS termination and hardened HTTP security headers.
4. **Centralized Observability & Tracing**: Integrate OpenTelemetry or Prometheus metrics to monitor P50/P95/P99 latency, token usage, and retrieval cache hits.
5. **Secret Management**: Store `GROQ_API_KEY` in a cloud secrets manager (e.g. AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) rather than local `.env` files.
6. **Semantic Entailment Verification**: Rule-based validators catch overt hallucinations (unsupported phone numbers/links), but cannot detect subtle factual drift. Implement downstream NLI cross-encoders for mission-critical enterprise workflows.
