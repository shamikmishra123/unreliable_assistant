"""Vault loading, chunking and chunk hashing (design §1.2.2, ADR-010 quarantine)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

WIKILINK = re.compile(r"\[\[([a-z0-9][a-z0-9-]*)\]\]")
MAX_CHUNK_CHARS = 4096


@dataclass(frozen=True)
class Note:
    note_id: str
    title: str
    tenant_id: str
    domain: str
    risk_tier: str
    version: int
    links: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_hash: str        # sha256 over tenant|note|text, so identical text in two tenants never collides
    uri: str
    text: str
    note_id: str
    tenant_id: str
    domain: str
    index_text: str        # title + text, what BM25 and the bi-encoder see


@dataclass
class VaultLoad:
    notes: dict[str, Note] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    quarantined: list[tuple[str, str]] = field(default_factory=list)  # (note_id, chunk_hash)


def parse_note(path: Path) -> Note:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name}: missing frontmatter")
    head, _, body = raw[4:].partition("\n---\n")
    meta: dict[str, str] = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    links_raw = meta.get("links", "[]").strip("[]")
    links = tuple(item.strip() for item in links_raw.split(",") if item.strip())
    note = Note(
        note_id=meta["id"], title=meta["title"], tenant_id=meta["tenant"],
        domain=meta.get("domain", "general"), risk_tier=meta.get("risk_tier", "TIER_1_READ"),
        version=int(meta.get("version", "1")), links=links, body=body.strip(),
    )
    if note.note_id != path.stem:
        raise ValueError(f"{path.name}: id '{note.note_id}' must match the file name")
    return note


def note_links(note: Note) -> tuple[str, ...]:
    """Outgoing links: frontmatter links plus [[wikilinks]] in the body, in order, unique."""
    found = list(note.links) + WIKILINK.findall(note.body)
    seen: list[str] = []
    for item in found:
        if item != note.note_id and item not in seen:
            seen.append(item)
    return tuple(seen)


def chunk_hash(tenant_id: str, note_id: str, text: str) -> str:
    return hashlib.sha256(f"{tenant_id}|{note_id}|{text}".encode("utf-8")).hexdigest()


def split_paragraphs(body: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        while len(para) > MAX_CHUNK_CHARS:
            out.append(para[:MAX_CHUNK_CHARS])
            para = para[MAX_CHUNK_CHARS:]
        out.append(para)
    return out


def load_vault(vault_dir: Path, is_poisoned) -> VaultLoad:
    """Load every note, chunk it, and quarantine chunks the injection scanner flags (ADR-010/015)."""
    load = VaultLoad()
    for path in sorted(vault_dir.glob("*.md")):
        note = parse_note(path)
        if note.note_id in load.notes:
            raise ValueError(f"duplicate note id {note.note_id}")
        load.notes[note.note_id] = note
        for text in split_paragraphs(note.body):
            digest = chunk_hash(note.tenant_id, note.note_id, text)
            if is_poisoned(text):
                load.quarantined.append((note.note_id, digest))
                continue
            load.chunks.append(Chunk(
                chunk_hash=digest, uri=f"rag://chunks/{digest}", text=text,
                note_id=note.note_id, tenant_id=note.tenant_id, domain=note.domain,
                index_text=f"{note.title}\n{text}",
            ))
    return load


class ChunkCatalog:
    """Chunk ownership without models: which (tenant, chunk_hash) pairs exist. Used by servers that need
    to validate a source chunk (state-store) without loading the retrieval index."""

    def __init__(self, load: VaultLoad):
        self._by_tenant: dict[str, set[str]] = {}
        for chunk in load.chunks:
            self._by_tenant.setdefault(chunk.tenant_id, set()).add(chunk.chunk_hash)

    def has_chunk(self, tenant_id: str, chunk_hash: str) -> bool:
        return chunk_hash in self._by_tenant.get(tenant_id, set())
