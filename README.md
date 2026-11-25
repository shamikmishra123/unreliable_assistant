# Unreliable Assistant → enterprise grade

A multi-tenant, grounded operations assistant built from [docs/design.md](docs/design.md) (LOCKED v1.2).
Every answer is either cited to a vault chunk, abstained, or refused. Lot changes never execute without
a reviewer's approval.

## Run

```bash
python3 -m pip install -e ".[test]"
export ANTHROPIC_API_KEY=...          # production generator (claude-haiku-5-5). Or LLM_STUB=1 for tests.
export ASSISTANT_REVIEWER_TOKENS='{"tok-rev-a":{"user_id":"user_reva","tenant_id":"tenant_a","role":"reviewer"}}'
python -m assistant.seed              # demo lot records
uvicorn assistant.api:app --port 8000
```

Models are loaded from the local Hugging Face cache (`BAAI/bge-large-en-v1.5`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`). Startup fails if no LLM credential is set and `LLM_STUB` is not `1`.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/agent/query` | `X-Tenant-ID` must equal body `tenant_id`, else 403 |
| POST | `/api/v1/agent/query/stream` | SSE. `released` event per verified sentence, then `done` |
| GET | `/api/v1/approvals/{id}` | Tenant-scoped status |
| POST | `/api/v1/approvals/{id}/decide` | Reviewer bearer token. Four-eyes rule, 24 h expiry, 409 once decided |
| GET | `/api/v1/audit/verify` | Reviewer only. Recomputes the audit hash chain |
| GET | `/metrics`, `/healthz` | Counters and index info |

## Layout

- `assistant/harness.py`: validation → regex repair → breaker → RBAC → UUIDv4 idempotency (design §1.2)
- `assistant/rag/`: per-tenant BM25 ∥ FAISS HNSW → RRF(60) → MiniLM rerank of top 16 → abstain < 0.42
- `assistant/verify.py`: lexical claim check (ADR-013 default). Vetoed sentences are never released
- `assistant/execution.py`: bounded ReAct state machine (max 5 steps) with a deterministic transaction planner
- `assistant/security.py`: PII redaction and injection shield (fail closed; indirect patterns on retrieved content only)
- `assistant/state.py`: SQLite state, hash-chained audit, approvals, lots, facts, turns
- `knowledge_vault/`: seed vault (two tenants, deliberately conflicting SKU-099 entries)

## Tests

```bash
/home/labuser/venv/bin/python3 -m pytest -q
```

## Deviations from the design (what is not built or differs)

These are deliberate, and each one is listed so nothing is silently claimed.

- **Storage:** SQLite only. Postgres/pgvector, Redis Cluster and the Redis-backed cache are not built.
  The cache is in-process (ADR-012 permits this for a single worker).
- **MCP:** the three servers run in-process behind a JSON-RPC-shaped harness, not as stdio/Streamable-HTTP MCP servers (ADR-018 allows in-process).
- **Verifier:** the lexical verifier is used. The NLI cross-encoder (ADR-013/016/017) is not built.
- **Planner:** TRANSACTION_EXEC uses a deterministic planner, not an LLM planner (ADR-017 not built).
  HYBRID_RAG answers are LLM-generated.
- **Streaming:** the LLM output is generated in full, then released sentence by sentence after verification.
  It is not token-streamed, so TTFT is the full generation time.
- **Auth:** reviewers authenticate with static bearer tokens from configuration. There is no gateway role mapping.
  All query callers are TIER_2 (ADR-021).
- **Not built:** Pluggy plugin registry, incident runbooks, signed manifests (Ed25519), the Streamlit workbench,
  the eval/KPI harness (`eval/kpis.py`), k6 load tests, the injection classifier (DeBERTa), vault review queue.
- **Idempotent submit:** a retried `submit_transaction` with the same idempotency key returns the same approval.

See the test results and any measured numbers in the final report, not in this README.
