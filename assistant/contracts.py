"""Wire contracts (design §1.2): ID formats, CallContext, tool inputs/outputs, JSON-RPC error codes."""
from __future__ import annotations

import re
from enum import IntEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

TENANT_RE = r"^tenant_[a-z0-9]{1,32}$"
USER_RE = r"^user_[a-z0-9]{1,64}$"
SKU_RE = r"^SKU-\d{3}$"
LOT_RE = r"^LOT-\d{5}$"
CHUNK_RE = r"^[a-f0-9]{64}$"
NOTE_RE = r"^[a-z0-9][a-z0-9-]{0,127}$"
APPROVAL_RE = r"^apr_[a-f0-9]{32}$"
TRACEPARENT_RE = r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$"

LOT_STATUSES = ("AVAILABLE", "QUARANTINED", "ON_HOLD", "RELEASED", "REJECTED")
ACTIONS = ("release", "quarantine", "hold", "reject", "open_deviation", "open_capa", "update_status")
Action = Literal["release", "quarantine", "hold", "reject", "open_deviation", "open_capa", "update_status"]


class ErrorCode(IntEnum):
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    SCHEMA_VALIDATION_FAILED = -32001
    AUTO_REPAIR_FAILED = -32002
    RBAC_DENIED = -32003
    IDEMPOTENCY_REPLAY = -32004
    TENANT_SCOPE_VIOLATION = -32005
    HITL_REQUIRED = -32006
    CIRCUIT_OPEN = -32007
    UPSTREAM_TIMEOUT = -32008
    LOW_CONFIDENCE_ABSTAIN = -32009
    PII_BLOCKED = -32010
    INJECTION_BLOCKED = -32011
    INTERNAL_ERROR = -32603


class ToolError(Exception):
    """A JSON-RPC style error raised by the harness or a tool handler."""

    def __init__(self, code: ErrorCode, message: str, data: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def to_jsonrpc(self, request_id: Any = None) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": int(self.code), "message": self.code.name,
                      "data": {"detail": self.message, **self.data}},
        }


class CallContext(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    tenant_id: str = Field(pattern=TENANT_RE)
    user_id: str = Field(pattern=USER_RE)
    session_id: UUID
    idempotency_key: UUID
    traceparent: str = Field(pattern=TRACEPARENT_RE)

    @field_validator("idempotency_key")
    @classmethod
    def _must_be_v4(cls, value: UUID) -> UUID:
        if value.version != 4:
            raise ValueError("idempotency_key must be a UUIDv4")
        return value


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# ---- ID repair (ADR-013 / design §1.2.1 repair examples) -------------------

_LOT_LOOSE = re.compile(r"^\s*lot[\s_-]*(\d{1,5})\s*$", re.IGNORECASE)
_SKU_LOOSE = re.compile(r"^\s*sku[\s_-]*(\d{1,3})\s*$", re.IGNORECASE)
_LOT_IN_TEXT = re.compile(r"\bLOT[\s_-]?(\d{1,5})\b", re.IGNORECASE)
_SKU_IN_TEXT = re.compile(r"\bSKU[\s_-]?(\d{1,3})\b", re.IGNORECASE)


def repair_lot_id(value: str) -> str | None:
    match = _LOT_LOOSE.match(value)
    return f"LOT-{int(match.group(1)):05d}" if match else None


def repair_sku_id(value: str) -> str | None:
    match = _SKU_LOOSE.match(value)
    return f"SKU-{int(match.group(1)):03d}" if match else None


def extract_lot_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in _LOT_IN_TEXT.finditer(text):
        lot = f"LOT-{int(match.group(1)):05d}"
        if lot not in seen:
            seen.append(lot)
    return seen


def extract_sku_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in _SKU_IN_TEXT.finditer(text):
        sku = f"SKU-{int(match.group(1)):03d}"
        if sku not in seen:
            seen.append(sku)
    return seen


# ---- Tool inputs (§1.2.2 - §1.2.4) ------------------------------------------

class HybridRetrieveIn(_Strict):
    ctx: CallContext
    query: str = Field(min_length=1, max_length=2000)
    domain: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")
    bm25_top_k: int = Field(default=50, ge=1, le=100)
    dense_top_k: int = Field(default=50, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_k: int = Field(default=4, ge=1, le=10)
    min_rerank_score: float = Field(default=0.42, ge=0.0, le=1.0)


class GraphSubgraphIn(_Strict):
    ctx: CallContext
    anchor_note_ids: list[str] = Field(min_length=1, max_length=4)
    depth: int = Field(default=2, ge=1, le=2)
    max_tokens: int = Field(default=320, ge=64, le=800)


class SessionFact(_Strict):
    key: str = Field(pattern=r"^[a-z_]{1,64}$")
    value: str = Field(max_length=512)
    source_chunk_hash: str | None = Field(default=None, pattern=CHUNK_RE)


class GetSessionFactsIn(_Strict):
    ctx: CallContext


class PutSessionFactIn(_Strict):
    ctx: CallContext
    fact: SessionFact


class LookupLotIn(_Strict):
    ctx: CallContext
    lot_id: str = Field(pattern=LOT_RE)


class SubmitTransactionIn(_Strict):
    ctx: CallContext
    action: Action
    payload: dict[str, Any]
    justification: str = Field(min_length=1, max_length=512)


class LotPayload(_Strict):
    lot_id: str = Field(pattern=LOT_RE)


class DeviationPayload(_Strict):
    lot_id: str = Field(pattern=LOT_RE)
    description: str = Field(min_length=1, max_length=1000)


class StatusPayload(_Strict):
    lot_id: str = Field(pattern=LOT_RE)
    status: Literal["AVAILABLE", "QUARANTINED", "ON_HOLD", "RELEASED", "REJECTED"]


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "release": LotPayload, "quarantine": LotPayload, "hold": LotPayload, "reject": LotPayload,
    "open_deviation": DeviationPayload, "open_capa": DeviationPayload,
    "update_status": StatusPayload,
}


def validate_payload(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Strict per-action payload check (ADR-018). Raises pydantic.ValidationError on extra or bad fields."""
    model = PAYLOAD_MODELS[action]
    return model.model_validate(payload).model_dump()


class GetApprovalIn(_Strict):
    ctx: CallContext
    approval_id: str = Field(pattern=APPROVAL_RE)


class LookupLotOut(BaseModel):
    lot_id: str
    found: bool
    status: str | None = None
    protocol_ref: str | None = None
    updated_at: str | None = None


# Tool names exposed through the harness; `append_audit` is internal and not callable.
TOOL_NAMES = ("hybrid_retrieve", "graph_subgraph", "get_session_facts", "put_session_fact",
              "lookup_lot", "submit_transaction", "get_approval")
REPAIRABLE_FIELDS = frozenset({"lot_id", "sku_id", "lot_ids"})
