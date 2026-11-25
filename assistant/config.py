"""Runtime settings (design §1.7 ADR-004/006/012/018). All values come from env with safe defaults."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    vault_dir: Path = ROOT / "knowledge_vault"
    db_path: Path = ROOT / "var" / "state.db"
    llm_stub: bool = False
    llm_model: str = "claude-haiku-5-5"            # ADR-008
    dense_model: str = "BAAI/bge-large-en-v1.5"    # §0 / ADR-001
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ADR-006
    rerank_pool: int = 16                          # RAG_RERANK_POOL (ADR-006)
    rrf_pool: int = 40                             # ADR-001 pool of 40
    min_rerank_score: float = 0.42                 # NG-3 abstention gate, not waivable
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60
    rerank_k: int = 4
    cache_ttl_s: float = 300.0                     # ADR-003
    cache_jitter_s: float = 45.0
    request_timeout_s: float = 30.0                # ADR-014
    llm_step_timeout_s: float = 10.0
    max_react_iterations: int = 5                  # ADR-004
    approval_ttl_s: float = 24 * 3600.0            # ADR-013
    reviewer_tokens: dict = field(default_factory=dict)  # token -> principal
    mcp_mode: str = "mcp"                          # "mcp" (servers over JSON-RPC) | "inproc" (tests, evals)
    verifier: str | None = None                    # "nli" | "lexical"; None = nli in mcp mode, lexical in inproc
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    per_step_timeout_s: float = 2.0                # blueprint target 0.12 s; see ADR-023 for the CPU-hardware value
    admin_token: str | None = None                 # platform-engineer endpoints (/admin/*); unset = disabled


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("ASSISTANT_DATA_DIR", ROOT / "var"))
    tokens_raw = os.environ.get("ASSISTANT_REVIEWER_TOKENS", "{}")
    return Settings(
        vault_dir=Path(os.environ.get("ASSISTANT_VAULT_DIR", ROOT / "knowledge_vault")),
        db_path=data_dir / "state.db",
        llm_stub=_flag("LLM_STUB", False),
        llm_model=os.environ.get("LLM_MODEL", "claude-haiku-5-5"),
        dense_model=os.environ.get("DENSE_MODEL", "BAAI/bge-large-en-v1.5"),
        rerank_model=os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_pool=int(os.environ.get("RAG_RERANK_POOL", "16")),
        reviewer_tokens=json.loads(tokens_raw),
        mcp_mode=os.environ.get("MCP_MODE", "mcp"),
        verifier=os.environ.get("VERIFIER") or None,
        nli_model=os.environ.get("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall"),
        per_step_timeout_s=float(os.environ.get("PER_STEP_TIMEOUT_S", "2.0")),
        admin_token=os.environ.get("ASSISTANT_ADMIN_TOKEN") or None,
    )


def state_token() -> str:
    """Bearer token between the API host and mcp-state-store (env MCP_STATE_TOKEN, else var/state_token)."""
    import secrets

    if os.environ.get("MCP_STATE_TOKEN"):
        return os.environ["MCP_STATE_TOKEN"]
    path = ROOT / "var" / "state_token"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(24))
        path.chmod(0o600)
    return path.read_text().strip()
