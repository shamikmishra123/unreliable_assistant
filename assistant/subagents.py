"""Specialised subagents (Gate 7). Each gets a ScopedHarness: a tool outside its allow-list is refused and audited.

- Retrieval: mcp-hybrid-rag + mcp-graphify-vault, read-only, cannot reach any state-mutating tool
- Execution: mcp-state-store only (no RAG, no web); a Tier 3 call becomes a PENDING_APPROVAL request
- Verification: claim-to-chunk check (NLI when available), audit log only; independent model context, vetoes drafts
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import CallContext, ErrorCode, ToolError
from .harness import Tier, ToolExecutionHarness
from .llm import ABSTAIN_TEXT
from .verify import SentenceVerdict, check_draft, check_sentence

RETRIEVAL_TOOLS = frozenset({"hybrid_retrieve", "graph_subgraph"})
EXECUTION_TOOLS = frozenset({"lookup_lot", "submit_transaction", "get_approval", "get_session_facts", "put_session_fact"})


class ScopedHarness:
    def __init__(self, agent: str, harness: ToolExecutionHarness, allowed: frozenset[str]):
        self.agent, self._harness, self.allowed = agent, harness, allowed

    def call(self, method: str, params: dict[str, Any], ctx: CallContext, tier: Tier) -> dict[str, Any]:
        if method not in self.allowed:
            self._harness._audit(ctx.tenant_id, {"type": "security", "event": "SUBAGENT_SCOPE_REFUSED",
                                                 "agent": self.agent, "tool": method})
            raise ToolError(ErrorCode.RBAC_DENIED, f"{self.agent} subagent is not allowed to call {method}")
        return self._harness.call(method, params, ctx, tier)

    @property
    def tools(self) -> set[str]:
        return set(self.allowed)


class RetrievalSubagent:
    def __init__(self, harness: ToolExecutionHarness):
        self.h = ScopedHarness("retrieval", harness, RETRIEVAL_TOOLS)

    def retrieve(self, query: str, ctx: CallContext, *, top_k: int, min_rerank_score: float) -> dict[str, Any]:
        return self.h.call("hybrid_retrieve", {"query": query, "bm25_top_k": top_k, "dense_top_k": top_k,
                                               "min_rerank_score": min_rerank_score}, ctx, Tier.TIER_2_COMPUTE)

    def subgraph(self, anchors: list[str], ctx: CallContext, max_tokens: int = 320) -> dict[str, Any]:
        return self.h.call("graph_subgraph", {"anchor_note_ids": anchors, "depth": 2, "max_tokens": max_tokens},
                           ctx, Tier.TIER_2_COMPUTE)


class ExecutionSubagent:
    def __init__(self, harness: ToolExecutionHarness):
        self.h = ScopedHarness("execution", harness, EXECUTION_TOOLS)


class VerificationSubagent:
    """Vetoes ungrounded drafts. `entail` (NLI) is optional; without it the lexical stopgap runs (ADR-013)."""

    def __init__(self, entail: Callable[[str, str], float] | None = None, audit: Callable[[str, dict], Any] | None = None):
        self.entail = entail
        self._audit = audit

    @property
    def method(self) -> str:
        return "nli" if self.entail else "lexical"

    def verify_sentence(self, sentence: str, retrieved: dict[str, str]) -> SentenceVerdict:
        return check_sentence(sentence, retrieved, ABSTAIN_TEXT, self.entail)

    def verify_draft(self, sentences: list[str], retrieved: dict[str, str]) -> list[SentenceVerdict]:
        return check_draft(sentences, retrieved, ABSTAIN_TEXT, self.entail)

    def audit(self, tenant_id: str, event: dict[str, Any]) -> None:
        if self._audit:
            self._audit(tenant_id, event)
