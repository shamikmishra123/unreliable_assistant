"""Tool handlers, grouped by the MCP server that owns them (design §1.2).

`build_*_tools` bind handlers to local resources and are used by the MCP server processes (and by
`inproc` mode). `remote_specs` rebinds the same contracts to an MCP host client (host side).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from .contracts import (
    ErrorCode, GetApprovalIn, GetSessionFactsIn, GraphSubgraphIn, HybridRetrieveIn, LookupLotIn,
    LookupLotOut, PutSessionFactIn, SubmitTransactionIn, ToolError, validate_payload,
)
from .graph import Graph
from .harness import Tier, ToolSpec
from .rag.index import RetrievalIndex
from .state import StateStore

RAG_SERVER = "mcp-hybrid-rag"
GRAPH_SERVER = "mcp-graphify-vault"
STATE_SERVER = "mcp-state-store"


def build_rag_tools(index: RetrievalIndex) -> list[ToolSpec]:
    def hybrid_retrieve(m: HybridRetrieveIn) -> dict:
        return index.retrieve(
            m.ctx.tenant_id, m.query, bm25_top_k=m.bm25_top_k, dense_top_k=m.dense_top_k,
            rrf_k=m.rrf_k, rerank_k=m.rerank_k, min_rerank_score=m.min_rerank_score, domain=m.domain)

    return [ToolSpec("hybrid_retrieve", RAG_SERVER, Tier.TIER_1_READ, HybridRetrieveIn, hybrid_retrieve)]


def build_graph_tools(graph: Graph) -> list[ToolSpec]:
    def graph_subgraph(m: GraphSubgraphIn) -> dict:
        return graph.subgraph(m.ctx, m.anchor_note_ids, m.depth, m.max_tokens)

    return [ToolSpec("graph_subgraph", GRAPH_SERVER, Tier.TIER_1_READ, GraphSubgraphIn, graph_subgraph)]


def build_state_tools(state: StateStore, has_chunk: Callable[[str, str], bool]) -> list[ToolSpec]:
    def get_session_facts(m: GetSessionFactsIn) -> dict:
        return {"facts": state.get_facts(m.ctx.tenant_id, str(m.ctx.session_id))}

    def put_session_fact(m: PutSessionFactIn) -> dict:
        source = m.fact.source_chunk_hash
        if source is None or not has_chunk(m.ctx.tenant_id, source):
            raise ToolError(ErrorCode.INVALID_PARAMS, "only verified facts can be stored: source chunk not found")
        return {"fact_id": state.put_fact(m.ctx.tenant_id, str(m.ctx.session_id), m.fact.key, m.fact.value, source)}

    def lookup_lot(m: LookupLotIn) -> dict:
        lot = state.get_lot(m.ctx.tenant_id, m.lot_id)
        if lot is None:
            if state.lot_owned_elsewhere(m.ctx.tenant_id, m.lot_id):
                raise ToolError(ErrorCode.TENANT_SCOPE_VIOLATION, "lot is not owned by this tenant")
            return LookupLotOut(lot_id=m.lot_id, found=False).model_dump()
        return LookupLotOut(found=True, **lot).model_dump()

    def submit_transaction(m: SubmitTransactionIn) -> dict:
        try:
            payload = validate_payload(m.action, m.payload)
        except ValidationError:
            raise ToolError(ErrorCode.SCHEMA_VALIDATION_FAILED, f"payload does not match action {m.action}") from None
        lot_id = payload["lot_id"]
        if state.get_lot(m.ctx.tenant_id, lot_id) is None:
            if state.lot_owned_elsewhere(m.ctx.tenant_id, lot_id):
                raise ToolError(ErrorCode.TENANT_SCOPE_VIOLATION, "lot is not owned by this tenant")
            raise ToolError(ErrorCode.INVALID_PARAMS, f"{lot_id} does not exist in this tenant")
        approval_id = state.create_approval(
            idempotency_key=str(m.ctx.idempotency_key), tenant_id=m.ctx.tenant_id,
            session_id=str(m.ctx.session_id), requester_id=m.ctx.user_id,
            action=m.action, payload=payload, justification=m.justification)
        return {"approval_id": approval_id, "status": "PENDING_APPROVAL"}

    def get_approval(m: GetApprovalIn) -> dict:
        row = state.get_approval(m.ctx.tenant_id, m.approval_id)
        if row is None:
            raise ToolError(ErrorCode.INVALID_PARAMS, "unknown approval for this tenant")
        return {"approval_id": row["approval_id"], "status": row["status"],
                "reviewer_id": row["reviewer_id"], "decided_at": row["decided_at"]}

    return [
        ToolSpec("get_session_facts", STATE_SERVER, Tier.TIER_1_READ, GetSessionFactsIn, get_session_facts),
        ToolSpec("put_session_fact", STATE_SERVER, Tier.TIER_2_COMPUTE, PutSessionFactIn, put_session_fact),
        ToolSpec("lookup_lot", STATE_SERVER, Tier.TIER_1_READ, LookupLotIn, lookup_lot),
        ToolSpec("submit_transaction", STATE_SERVER, Tier.TIER_3_MUTATE, SubmitTransactionIn,
                 submit_transaction, hitl_gate=True),
        ToolSpec("get_approval", STATE_SERVER, Tier.TIER_1_READ, GetApprovalIn, get_approval),
    ]


def build_tools(index: RetrievalIndex, graph: Graph, state: StateStore) -> list[ToolSpec]:
    """All tools bound to local resources (`inproc` mode)."""
    return build_rag_tools(index) + build_graph_tools(graph) + build_state_tools(state, index.has_chunk)


def remote_specs(contracts: list[ToolSpec], call: Callable[[str, str, BaseModel], dict[str, Any]]) -> list[ToolSpec]:
    """Same contracts, handlers forwarded to an MCP host: call(server, tool, validated_model) -> result dict."""
    out = []
    for spec in contracts:
        def handler(model: BaseModel, _spec: ToolSpec = spec) -> dict[str, Any]:
            return call(_spec.server, _spec.name, model)
        out.append(ToolSpec(spec.name, spec.server, spec.tier, spec.input_model, handler, spec.hitl_gate, spec.internal))
    return out


def contract_specs() -> list[ToolSpec]:
    """Every contract (public and internal) with no resources bound: used to build and sign manifests."""
    from .state_internal import build_internal_tools

    return (build_rag_tools(None) + build_graph_tools(None) + build_state_tools(None, lambda *_: False)  # type: ignore[arg-type]
            + build_internal_tools(None))  # type: ignore[arg-type]
