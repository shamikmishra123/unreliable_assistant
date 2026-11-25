"""Triage (design §1.3 step 4): deterministic route choice. Unsure means HYBRID_RAG, never TRANSACTION_EXEC (FMEA #9)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .contracts import extract_lot_ids


class Route(str, Enum):
    DIRECT_ANSWER = "DIRECT_ANSWER"
    HYBRID_RAG = "HYBRID_RAG"
    TRANSACTION_EXEC = "TRANSACTION_EXEC"
    REJECT_OOS = "REJECT_OOS"


@dataclass(frozen=True)
class Triage:
    route: Route
    intent: str
    risk_tier: str
    confidence: float


_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you do)\b", re.I)
_OUT_OF_SCOPE = re.compile(r"\b(weather|stock price|bitcoin|crypto|lottery|poem|joke|recipe|horoscope|browse|google|latest news)\b", re.I)
_WRITE = re.compile(r"\b(release|quarantine|hold|reject|open (a |an )?(deviation|capa)|(update|change|set) (the )?status)\b", re.I)
_QUESTION = re.compile(r"^\s*(what|how|why|when|which|who|does|do|is|are|can|could|should|would)\b", re.I)
_READ = re.compile(r"\b(status|verify|verification|check|protocol|state|lookup|look up|is .{0,20}released)\b", re.I)


def classify(query: str) -> Triage:
    q = query.strip()
    if _GREETING.search(q) and len(q.split()) <= 8:
        return Triage(Route.DIRECT_ANSWER, "smalltalk", "TIER_1_READ", 0.95)
    if _OUT_OF_SCOPE.search(q):
        return Triage(Route.REJECT_OOS, "out_of_scope", "TIER_1_READ", 0.9)
    lots = extract_lot_ids(q)
    is_request = not _QUESTION.search(q)
    if _WRITE.search(q) and is_request and (lots or re.search(r"open (a |an )?(deviation|capa)", q, re.I)):
        return Triage(Route.TRANSACTION_EXEC, "transaction", "TIER_3_MUTATE", 0.9)
    if lots and _READ.search(q):
        return Triage(Route.TRANSACTION_EXEC, "lot_lookup", "TIER_1_READ", 0.9)
    return Triage(Route.HYBRID_RAG, "policy_qa", "TIER_1_READ", 0.8)
