"""Hybrid retrieval: BM25 ∥ dense(FAISS HNSW) → RRF(k=60) → cross-encoder rerank → abstention (ADR-001, ADR-006).

Indexes are partitioned per tenant, so a query can only ever see its own tenant's chunks (NG-5).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ..contracts import ErrorCode, ToolError
from .corpus import Chunk, VaultLoad, load_vault
from ..security import is_poisoned

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Models:
    """Bi-encoder and cross-encoder, loaded once per process."""

    def __init__(self, dense_name: str, rerank_name: str):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self._torch = torch
        self.embedder = SentenceTransformer(dense_name, device="cpu")
        self.reranker = CrossEncoder(rerank_name, device="cpu", max_length=512)
        self.dense_name = dense_name
        self.rerank_name = rerank_name

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if query else ""
        vecs = self.embedder.encode([prefix + t for t in texts], normalize_embeddings=True,
                                    convert_to_numpy=True, batch_size=16, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        logits = self.reranker.predict([(query, t) for t in texts], convert_to_numpy=True,
                                       show_progress_bar=False,
                                       activation_fn=self._torch.nn.Identity())
        return [float(1.0 / (1.0 + np.exp(-x))) for x in np.ravel(logits)]


class TenantIndex:
    def __init__(self, chunks: list[Chunk], models: Models):
        self.chunks = chunks
        self.by_hash = {c.chunk_hash: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks], k1=1.25, b=0.75)
        self.vectors = models.embed([c.index_text for c in chunks])
        self.hnsw = faiss.IndexHNSWFlat(self.vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.add(self.vectors)


@dataclass
class RetrievalIndex:
    models: Models
    tenants: dict[str, TenantIndex]
    load: VaultLoad

    def has_chunk(self, tenant_id: str, chunk_hash: str) -> bool:
        index = self.tenants.get(tenant_id)
        return bool(index and chunk_hash in index.by_hash)

    def chunk(self, tenant_id: str, chunk_hash: str) -> Chunk:
        index = self.tenants.get(tenant_id)
        if index is None or chunk_hash not in index.by_hash:
            raise ToolError(ErrorCode.TENANT_SCOPE_VIOLATION, "chunk is not owned by this tenant")
        return index.by_hash[chunk_hash]

    def retrieve(self, tenant_id: str, query: str, *, bm25_top_k: int = 50, dense_top_k: int = 50,
                 rrf_k: int = 60, rerank_k: int = 4, min_rerank_score: float = 0.42,
                 pool: int = 40, rerank_pool: int = 16, domain: str | None = None) -> dict[str, Any]:
        timings: dict[str, float] = {}
        index = self.tenants.get(tenant_id)
        if index is None or not index.chunks:
            return {"chunks": [], "top_score": 0.0, "abstain": True,
                    "stage_latency_ms": {"bm25": 0.0, "dense": 0.0, "rrf": 0.0, "rerank": 0.0}}

        # Stage 1a: BM25
        t0 = time.perf_counter()
        bm_scores = index.bm25.get_scores(tokenize(query))
        bm_order = [int(i) for i in np.argsort(-bm_scores)[:bm25_top_k] if bm_scores[i] > 0]
        if domain:
            bm_order = [i for i in bm_order if index.chunks[i].domain == domain]
        timings["bm25"] = (time.perf_counter() - t0) * 1000

        # Stage 1b: dense
        t0 = time.perf_counter()
        qvec = self.models.embed([query], query=True)
        k_dense = min(dense_top_k, len(index.chunks))
        dense_scores, dense_ids = index.hnsw.search(qvec, k_dense)
        dense_pairs = [(int(i), float(s)) for i, s in zip(dense_ids[0], dense_scores[0]) if i >= 0]
        timings["dense"] = (time.perf_counter() - t0) * 1000
        dense_map = dict(dense_pairs)

        # Stage 2: RRF with equal weights (ADR-001)
        t0 = time.perf_counter()
        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm_order, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (idx, _) in enumerate(dense_pairs, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        ordered = sorted(fused, key=lambda i: -fused[i])
        if domain:
            ordered = [i for i in ordered if index.chunks[i].domain == domain]
        seen_hashes: set[str] = set()
        pool_idx: list[int] = []
        for idx in ordered:
            digest = index.chunks[idx].chunk_hash
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            pool_idx.append(idx)
            if len(pool_idx) == pool:
                break
        timings["rrf"] = (time.perf_counter() - t0) * 1000

        # Stage 3: cross-encoder over the RRF head only (ADR-006)
        t0 = time.perf_counter()
        head = pool_idx[:rerank_pool]
        scores = self.models.rerank(query, [index.chunks[i].text for i in head]) if head else []
        timings["rerank"] = (time.perf_counter() - t0) * 1000
        ranked = sorted(zip(head, scores), key=lambda pair: -pair[1])[:rerank_k]

        out_chunks = []
        for idx, score in ranked:
            chunk = index.chunks[idx]
            out_chunks.append({
                "chunk_hash": chunk.chunk_hash, "uri": chunk.uri, "text": chunk.text,
                "note_id": chunk.note_id,
                "bm25_score": float(bm_scores[idx]) if idx in bm_order else None,
                "dense_cosine": dense_map.get(idx),
                "rrf_rank": pool_idx.index(idx) + 1,
                "rerank_score": score,
            })
        top_score = ranked[0][1] if ranked else 0.0
        return {
            "chunks": out_chunks,
            "top_score": top_score,
            "abstain": top_score < min_rerank_score,   # abstention short-circuit (NG-3)
            "stage_latency_ms": timings,
        }


def build_index(vault_dir: Path, models: Models) -> RetrievalIndex:
    load = load_vault(vault_dir, is_poisoned)
    by_tenant: dict[str, list[Chunk]] = {}
    for chunk in load.chunks:
        by_tenant.setdefault(chunk.tenant_id, []).append(chunk)
    tenants = {tid: TenantIndex(chunks, models) for tid, chunks in by_tenant.items()}
    return RetrievalIndex(models=models, tenants=tenants, load=load)
