## **Proj-2: Enterprise RAG Architecture w/ Security Guardrails**

A permission-aware RAG query layer over Project 1's pgvector store — answers are filtered by user clearance/role *before* they reach an LLM, and both inputs and outputs are checked by NeMo Guardrails.

- **The Problem:** Standard AI chatbots hallucinate and leak sensitive corporate data to unauthorized users.
- **What It Builds:** A query service that embeds a question, retrieves only the document chunks a given user is permitted to see, and generates an answer through input/output guardrails — exposed as both a CLI and a FastAPI service.
- **How It Connects:** Reads directly from Project 1's (`01-DataClean-and-Chunk`) existing `document_chunks` table — same Cloud SQL instance, same Vertex AI embedding model/dimensions, no re-ingestion. Feeds Project 3's autonomous agent.
- **Tech Stack:** pgvector, Vertex AI embeddings, OpenAI or DeepSeek (pluggable, OpenAI-compatible) for generation, NeMo Guardrails, FastAPI.

## Architecture

```
User question + UserContext (user_id, clearance, roles)
    │
    ▼
QueryEmbedder (Vertex AI text-embedding-005) ── same model/dims as Project 1's ingestion
    │
    ▼
PermissionAwareVectorStore.search()
    │  SQL ACL predicate (sensitivity + allowed_roles) pushed into the query
    │  + Python defense-in-depth re-check on the returned rows
    ▼
zero chunks? ──► short-circuit, no LLM call, canned "no accessible documents" response
    │ no
    ▼
GuardrailsEngine (NeMo Guardrails)
    │  self-check input rail  (blocks prompt injection / jailbreaks)
    │  self-check output rail (blocks confidential-data leakage / fabrication)
    ▼
ChatOpenAI-compatible LLM (OpenAI or DeepSeek — one factory, base_url swap)
    │
    ▼
Answer + cited sources
```

Exposed via `main.py` (CLI) and `src/rag_guardrails/api/app.py` (FastAPI: `POST /query`, `GET /health`).

## Project Structure

```
├── main.py                          # CLI entry point (query, tag-permissions, serve, info)
├── config/rails/config.yml          # NeMo Guardrails input/output self-check rails
├── src/rag_guardrails/
│   ├── config.py                    # RAGConfig from env (reuses Project 1's DB env vars)
│   ├── auth/models.py               # ClearanceLevel, UserContext
│   ├── security/
│   │   ├── access_control.py        # SQL ACL predicate + Python defense-in-depth filter
│   │   └── guardrails.py            # GuardrailsEngine wrapping NeMo Guardrails
│   ├── retrieval/
│   │   ├── embeddings.py            # QueryEmbedder (Vertex AI)
│   │   └── vector_store.py          # PermissionAwareVectorStore (reads document_chunks)
│   ├── llm/provider.py              # get_chat_llm(): openai | deepseek factory
│   ├── pipeline.py                  # RAGPipeline orchestrator
│   └── api/                         # FastAPI app, schemas, dependencies
└── tests/                           # pytest + pytest-mock, fully mocked (no live calls)
```

## Setup

```bash
# 1. Copy and fill in environment variables — point at the SAME Cloud SQL
#    instance/table Project 1 populated, plus an LLM key (OpenAI or DeepSeek)
cp .env.example .env

# 2. Install dependencies
uv sync

# 3. Tag a real ingested document with an access-control level (demo/seeding)
uv run python main.py tag-permissions <filename.pdf> --sensitivity confidential --roles finance

# 4. Ask a question as a given user
uv run python main.py query "..." --user-id demo --clearance confidential --roles finance

# 5. Or run the HTTP service
uv run python main.py serve
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py query "<question>" --user-id U --clearance C [--roles R]` | Ask a question through the permission-filtered, guarded pipeline |
| `python main.py tag-permissions <filename> --sensitivity S --roles R` | Tag already-ingested chunks with ACL metadata |
| `python main.py serve [--host] [--port]` | Run the FastAPI query service |
| `python main.py info` | Show resolved configuration (secrets redacted) |
| `POST /query` | `{question, user_id, clearance, roles, top_k?}` → `{answer, sources}` |
| `GET /health` | Liveness check |

## Tech Stack

- **Retrieval:** Cloud SQL for PostgreSQL (pgvector, HNSW cosine index) — Project 1's existing `document_chunks` table
- **Embeddings:** Vertex AI `text-embedding-005` (query-time only, matches Project 1's ingestion model)
- **Permissions:** Clearance level (public/internal/confidential/restricted) + optional role tags, stored in the existing `metadata` JSONB column; enforced in SQL and re-checked in Python
- **Generation LLM:** Pluggable via `LLM_PROVIDER` — OpenAI or DeepSeek (OpenAI-compatible API), one `ChatOpenAI`-based factory
- **Guardrails:** NeMo Guardrails self-check input/output rails
- **API:** FastAPI + Uvicorn
- **Tests:** pytest + pytest-mock (fully mocked, no live GCP/DB/LLM calls required)

## Status

- [x] Permission model (`ClearanceLevel` + role tags) and SQL ACL predicate
- [x] Defense-in-depth Python re-check on retrieved rows (fails closed on missing/malformed metadata)
- [x] Query-time embedding matched to Project 1's ingestion model/dimensions
- [x] Permission-aware vector search over Project 1's live `document_chunks` table
- [x] Pluggable generation LLM (OpenAI / DeepSeek / any OpenAI-compatible endpoint)
- [x] NeMo Guardrails input + output self-check rails wired to the same LLM
- [x] Fail-closed short-circuit when retrieval returns zero accessible chunks (no LLM call)
- [x] CLI (`query`, `tag-permissions`, `serve`, `info`) and FastAPI service (`/query`, `/health`)
- [x] Unit test suite (33 tests, fully mocked)
- [x] Live end-to-end verification against real Cloud SQL data — confirmed the ACL boundary flips correctly in both directions and the input guardrail blocks real prompt-injection attempts

## To-Do for Production

- [ ] Replace demo-level auth (`user_id`/`clearance`/`roles` on the request body) with real SSO/JWT-derived identity
- [ ] Output groundedness/hallucination check — verify the answer is actually supported by the retrieved chunks, not just policy-compliant
- [ ] Validate guardrails against a production-grade LLM (self-check rails were only proven reliable on capable models; small/local models can false-positive or false-negative)
- [ ] Rate limiting / auth on the FastAPI service before any external exposure
- [ ] Structured audit logging of denied/blocked queries for compliance review
- [ ] Deployment target (Cloud Run) + CI
