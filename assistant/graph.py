"""Vault + code graph and token-bounded subgraph (Gate 17, ADR-012, ADR-014, FMEA #7).

`graph.json` is built ahead of time by `python -m assistant.graphify build` (vault layer from frontmatter and
wikilinks, code layer from the AST: modules, imports, Pydantic schemas). Prompts use the vault layer only.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import CallContext, ErrorCode, ToolError
from .rag.corpus import Note, note_links


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def vault_node(note: Note) -> dict[str, Any]:
    first = note.body.split("\n\n", 1)[0]
    return {"id": note.note_id, "tenant": note.tenant_id, "domain": note.domain, "risk_tier": note.risk_tier,
            "version": note.version, "title": note.title, "summary": first[:200],
            "raw_tokens": approx_tokens(note.body)}


@dataclass
class Graph:
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, tuple[str, ...]]
    code: dict[str, Any] = field(default_factory=dict)
    vault_digest: str = ""

    @classmethod
    def from_notes(cls, notes: dict[str, Note], code: dict[str, Any] | None = None, vault_digest: str = "") -> "Graph":
        nodes = {nid: vault_node(n) for nid, n in notes.items()}
        edges = {nid: tuple(t for t in note_links(n) if t in notes) for nid, n in notes.items()}
        return cls(nodes=nodes, edges=edges, code=code or {}, vault_digest=vault_digest)

    @classmethod
    def from_json(cls, path: Path) -> "Graph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vault = data["vault"]
        return cls(nodes={n["id"]: n for n in vault["nodes"]},
                   edges={k: tuple(v) for k, v in vault["edges"].items()},
                   code=data.get("code", {}), vault_digest=data.get("vault_digest", ""))

    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "vault_digest": self.vault_digest,
                "vault": {"nodes": list(self.nodes.values()), "edges": {k: list(v) for k, v in self.edges.items()}},
                "code": self.code}

    def subgraph(self, ctx: CallContext, anchors: list[str], depth: int = 2, max_tokens: int = 320) -> dict:
        for anchor in anchors:
            node = self.nodes.get(anchor)
            if node is None:
                raise ToolError(ErrorCode.INVALID_PARAMS, f"unknown anchor note {anchor}")
            if node["tenant"] != ctx.tenant_id:
                raise ToolError(ErrorCode.TENANT_SCOPE_VIOLATION, "anchor note belongs to another tenant")

        order: list[str] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((a, 0) for a in anchors)
        while queue:                                  # BFS, hop order, never leaving the caller's tenant
            nid, hop = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            order.append(nid)
            if hop >= depth:
                continue
            for nxt in self.edges.get(nid, ()):
                if self.nodes[nxt]["tenant"] == ctx.tenant_id and nxt not in visited:
                    queue.append((nxt, hop + 1))

        out_nodes, used = [], 0
        raw_total = sum(self.nodes[n]["raw_tokens"] for n in order)
        kept: set[str] = set()
        for nid in order:
            node = self.nodes[nid]
            entry = {"id": nid, "domain": node["domain"], "risk_tier": node["risk_tier"],
                     "version": node["version"], "summary": node["summary"]}
            cost = approx_tokens(str(entry))
            if used + cost > max_tokens:
                break
            used += cost
            out_nodes.append(entry)
            kept.add(nid)
        edges = [(s, "links_to", d) for s in kept for d in self.edges.get(s, ()) if d in kept]
        return {"nodes": out_nodes, "edges": sorted(edges), "token_count": used, "raw_token_estimate": raw_total}
