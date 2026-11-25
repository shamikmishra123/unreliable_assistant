"""Prompt assembly (design §1.3 step 6): GoalAnchorBlock at index 0, subgraph, chunks, compressed observations, last turns."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .llm import SYSTEM_PROMPT, Prompt

OBSERVATION_TOKEN_BUDGET = 180   # ADR-004


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def compress(text: str, max_tokens: int = OBSERVATION_TOKEN_BUDGET) -> str:
    if approx_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4] + " ..."


@dataclass(frozen=True)
class GoalAnchorBlock:
    """Immutable for the whole session: original intent, normalized entities, policy constraints."""
    intent: str
    query: str
    entities: tuple[str, ...]
    constraints: tuple[str, ...] = (
        "Answer only from CONTEXT", "Cite every factual sentence", "Abstain when CONTEXT is insufficient",
    )

    def render(self) -> str:
        lines = [f"GOAL intent={self.intent}", f"QUESTION {self.query}"]
        if self.entities:
            lines.append("ENTITIES " + ", ".join(self.entities))
        lines += [f"CONSTRAINT {c}" for c in self.constraints]
        return "\n".join(lines)


def assemble(anchor: GoalAnchorBlock, subgraph: dict[str, Any] | None, chunks: list[dict[str, Any]],
             observations: list[str], turns: list[dict[str, str]]) -> Prompt:
    parts = [anchor.render()]
    if subgraph and subgraph.get("nodes"):
        parts.append("NOTE GRAPH\n" + "\n".join(f"- {n['id']} ({n['domain']}): {n['summary']}"
                                                  for n in subgraph["nodes"]))
    parts.append("CONTEXT\n" + "\n\n".join(f"[{c['uri']}]\n{c['text']}" for c in chunks))
    if observations:
        parts.append("OBSERVATIONS\n" + "\n".join(compress(o) for o in observations))
    if turns:
        parts.append("RECENT TURNS\n" + "\n".join(f"{t['role']}: {compress(t['text'], 120)}" for t in turns[-3:]))
    user = "\n\n".join(parts)
    return Prompt(system=SYSTEM_PROMPT, user=user, chunks=tuple(chunks))
