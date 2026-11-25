"""Request pipeline (design §1.3). Pre-hooks (plugin registry, fail closed) → tenant-scoped cache → triage →
route (Retrieval / Execution / Verification subagents) → faithfulness-first verification → egress tenant check →
hash-chained audit. Every stage is a named trace span (Gate 10).

`_events` is the single implementation. Non-streaming callers verify the whole draft; streaming callers verify and
release one sentence at a time (ADR-011): text already released is never retracted, the first vetoed sentence stops
the stream, and if nothing was released the whole-draft veto path applies.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .contracts import CallContext, ErrorCode, ToolError, extract_lot_ids, extract_sku_ids
from .execution import ReActRunner, TransactionPlanner
from .harness import Tier
from .llm import ABSTAIN_TEXT, GeneratorError
from .prompt import GoalAnchorBlock, assemble
from .services import Services
from .tracing import new_traceparent
from .triage import Route, classify
from .verify import cited_hashes, split_sentences, strip_citations

DIRECT_ANSWER_TEXT = ("I'm the operations assistant for your tenant. I can answer policy, SOP and catalog "
                      "questions from the approved vault with citations, look up a lot's status, and file "
                      "requests for lot changes that a reviewer must approve.")
REFUSAL_OUT_OF_SCOPE = ("That is outside what I can do here. I answer policy, SOP and catalog questions from "
                        "the approved vault, and I can look up lots or file lot requests.")
REFUSAL_INJECTION = "I can't process that request. Please ask it as a question about your operations."
TIMEOUT_TEXT = "The request took too long and was stopped. Nothing further was changed."
RETRIEVAL_DOWN_TEXT = ("I can't reach the knowledge vault right now, so I won't answer from memory. "
                       "Please try again shortly.")

_FILLER = frozenset("the a an is are was were what whats please tell me about of for to in on does do can you i my "
                    "how which who when where it its this that".split())


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower()).rstrip("?!. ")


def aggressive_norm(norm: str) -> str:
    """Latency runbook (ADR-018): match the word sequence with filler words removed. No embedding, so opposite
    meanings ("with" / "without" sign-off) can never collide the way a cosine >= 0.92 match can."""
    return " ".join(w for w in re.findall(r"[a-z0-9][a-z0-9-]*", norm) if w not in _FILLER)


class Orchestrator:
    def __init__(self, services: Services):
        self.s = services

    # ---- public API ------------------------------------------------------------------
    async def handle(self, *, tenant_id: str, user_id: str, session_id: uuid.UUID, query: str,
                     stream: bool = False, gateway_ms: float | None = None) -> dict[str, Any]:
        released: list[str] = []
        final: dict[str, Any] = {}
        async for kind, payload in self._events(tenant_id, user_id, session_id, query, stream, gateway_ms):
            if kind == "released":
                released.append(payload["text"])
            else:
                final = payload
        final["_released"] = released or final.pop("_released_fallback", [final["answer"]])
        final.pop("_released_fallback", None)
        return final

    def handle_stream(self, *, tenant_id: str, user_id: str, session_id: uuid.UUID, query: str,
                      gateway_ms: float | None = None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        return self._events(tenant_id, user_id, session_id, query, True, gateway_ms)

    # ---- pipeline --------------------------------------------------------------------------
    async def _events(self, tenant: str, user: str, session: uuid.UUID, query: str, stream: bool,
                      gateway_ms: float | None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        s, tr, m = self.s, self.s.tracer, self.s.metrics
        trace_id = uuid.uuid4().hex
        traceparent = new_traceparent(trace_id)
        rec = tr.start(trace_id)
        t0 = time.perf_counter()
        m.inc("requests")
        if gateway_ms is not None:
            tr.add_span("gateway", gateway_ms)
        tuning = s.runtime.tuning
        common = dict(tenant=tenant, user=user, session=session, trace_id=trace_id)

        # 2. pre-hooks, fail closed
        try:
            with tr.span("hook.pre"):
                pre = s.plugins.run_pre(query)
        except ToolError as err:
            m.inc("injection_blocked" if err.code == ErrorCode.INJECTION_BLOCKED else "pii_blocked")
            body = _body(Route.REJECT_OOS, REFUSAL_INJECTION, status="HALTED", abstained=True, cache="BYPASS")
            yield "released", {"index": 0, "text": body["answer"]}
            yield "done", self._finish(**common, text="[blocked]", pii={}, body=body, t0=t0, rec=rec, trace_id=trace_id)
            return
        text, pii = pre.text, pre.redaction_counts
        if pii:
            m.inc("pii_redactions", sum(pii.values()))

        # 3. tenant-scoped cache (never for requests that contained PII)
        norm = normalize(text)
        keys = [norm] + ([aggressive_norm(norm)] if aggressive_norm(norm) else [])
        cache_user = user if tuning.cache_scope == "user" else None
        if not pii:
            with tr.span("cache.lookup"):
                hit = s.cache.get(tenant, aggressive_norm(norm) if tuning.aggressive_cache and keys[-1] else norm, cache_user)
            if s.cache.mismatches:
                m.inc("egress_tenant_mismatch", s.cache.mismatches)
                s.cache.mismatches = 0
            if hit is not None:
                m.inc("cache_hits")
                hit.pop("tenant_id", None)
                body = {**hit, "route": hit["route"], "_trace": [], "_cacheable": False, "cache": "HIT"}
                for i, sentence in enumerate(split_sentences(body["answer"]) or [body["answer"]]):
                    yield "released", {"index": i, "text": sentence}
                yield "done", self._finish(**common, text=text, pii=pii, body=body, t0=t0, rec=rec, trace_id=trace_id)
                return
        cache_state = "BYPASS" if pii else "MISS"

        # 4. triage
        with tr.span("triage.classify"):
            triage = classify(text)
        m.inc(f"route_{triage.route.value.lower()}")

        # 5-8. route (the whole route, streaming included, shares one request timeout; ADR-014)
        body: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(s.settings.request_timeout_s):
                if triage.route == Route.DIRECT_ANSWER:
                    body = _body(triage.route, DIRECT_ANSWER_TEXT)
                elif triage.route == Route.REJECT_OOS:
                    body = _body(triage.route, REFUSAL_OUT_OF_SCOPE, abstained=True)
                elif triage.route == Route.TRANSACTION_EXEC:
                    body = await self._transaction(text, tenant, user, session, traceparent)
                else:
                    async for kind, payload in self._hybrid(triage.intent, text, tenant, user, session,
                                                            traceparent, stream):
                        if kind == "released":
                            yield kind, payload
                        else:
                            body = payload
        except TimeoutError:
            m.inc("timeouts")
            body = _body(triage.route, TIMEOUT_TEXT, status="HALTED", faithfulness=0.0)
        assert body is not None

        cacheable = body.pop("_cacheable", False)
        body["cache"] = cache_state
        if cacheable and not pii and body["status"] == "DONE":
            value = {k: body[k] for k in ("route", "answer", "citations", "abstained", "status", "approval_id",
                                          "faithfulness")} | {"tenant_id": tenant}
            for key in keys:
                s.cache.put(tenant, key, value, cache_user)
        if body.get("_emit_answer", True) and not body.get("_streamed"):
            for i, sentence in enumerate(split_sentences(body["answer"]) or [body["answer"]]):
                yield "released", {"index": i, "text": sentence}
        yield "done", self._finish(**common, text=text, pii=pii, body=body, t0=t0, rec=rec, trace_id=trace_id)

    # ---- routes ---------------------------------------------------------------------------
    def _ctx(self, tenant: str, user: str, session: uuid.UUID, traceparent: str) -> CallContext:
        return CallContext(tenant_id=tenant, user_id=user, session_id=session,
                           idempotency_key=uuid.uuid4(), traceparent=traceparent)

    async def _transaction(self, text: str, tenant: str, user: str, session: uuid.UUID, traceparent: str) -> dict[str, Any]:
        s = self.s
        runner = ReActRunner(s.execution.h, lambda: self._ctx(tenant, user, session, traceparent),
                             Tier.TIER_2_COMPUTE, max_iterations=s.settings.max_react_iterations,
                             budget_s=s.settings.request_timeout_s - 5, step_timeout_s=s.settings.per_step_timeout_s,
                             tracer=s.tracer)
        result = await s.pools["db_pool"].run(runner.run, TransactionPlanner(text))
        if result.state.value == "PENDING_APPROVAL":
            s.metrics.inc("approvals_requested")
        return _body(Route.TRANSACTION_EXEC, result.answer, status=result.state.value,
                     approval_id=result.approval_id, _trace=result.steps)

    async def _hybrid(self, intent: str, text: str, tenant: str, user: str, session: uuid.UUID, traceparent: str,
                      stream: bool) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        s, tr, m = self.s, self.s.tracer, self.s.metrics
        tuning = s.runtime.tuning
        turns_task = asyncio.create_task(s.pools["db_pool"].run(s.store.recent_turns, tenant, str(session), 3))
        try:
            retrieval = await s.pools["rag_pool"].run(
                s.retrieval.retrieve, text, self._ctx(tenant, user, session, traceparent),
                top_k=tuning.top_k, min_rerank_score=tuning.rerank_cutoff)
        except ToolError as err:       # FMEA #3: retrieval is down and failover failed -> abstain, never answer ungrounded
            m.inc("retrieval_failures")
            turns_task.cancel()
            yield "final", _body(Route.HYBRID_RAG, RETRIEVAL_DOWN_TEXT, abstained=True, faithfulness=1.0,
                                 degraded=err.code.name)
            return
        for stage, name in (("bm25", "mcp.rag.bm25"), ("dense", "mcp.rag.biencoder"), ("rrf", "mcp.rag.rrf"),
                            ("rerank", "mcp.rag.rerank")):
            if stage in retrieval["stage_latency_ms"]:
                tr.add_span(name, retrieval["stage_latency_ms"][stage])
        if retrieval["abstain"]:
            m.inc("abstentions")
            turns_task.cancel()
            yield "final", _body(Route.HYBRID_RAG, ABSTAIN_TEXT, abstained=True, cacheable=True)
            return

        chunks = retrieval["chunks"]
        if tuning.strict_extractive:      # hallucination runbook: no free generation at all
            m.inc("strict_extractive")
            turns_task.cancel()
            yield "final", _quote_fallback(chunks, faithfulness=1.0, cacheable=False)
            return

        subgraph = None
        if not tuning.skip_graph:
            anchors: list[str] = []
            for c in chunks:
                if c["note_id"] not in anchors and len(anchors) < 4:
                    anchors.append(c["note_id"])
            try:  # FMEA #7: graph failure means no subgraph; the answer is still grounded in the chunks
                with tr.span("mcp.graphify"):
                    subgraph = await s.pools["rag_pool"].run(
                        s.retrieval.subgraph, anchors, self._ctx(tenant, user, session, traceparent))
            except ToolError:
                m.inc("graph_failures")
        try:
            turns = await turns_task
        except Exception:  # noqa: BLE001
            turns = []
        with tr.span("agent.prompt"):
            anchor = GoalAnchorBlock(intent=intent, query=text,
                                     entities=tuple(extract_lot_ids(text) + extract_sku_ids(text)))
            prompt = assemble(anchor, subgraph, chunks, observations=[], turns=turns)
        retrieved = {c["chunk_hash"]: c["text"] for c in chunks}

        gen = s.generator
        if stream and hasattr(gen, "stream"):
            async for kind, payload in self._generate_streaming(prompt, chunks, retrieved):
                yield kind, payload
            return
        try:
            with tr.span("llm.generate"):
                draft = await s.pools["llm_pool"].run(gen.generate, prompt)
        except GeneratorError:
            m.inc("generator_failures")      # FMEA #4/#5: extractive quote, verbatim and cited; never cached
            yield "final", _quote_fallback(chunks, faithfulness=0.0)
            return
        sentences = split_sentences(draft)
        with tr.span("verify.nli", method=s.verification.method):
            verdicts = await s.pools["llm_pool"].run(s.verification.verify_draft, sentences, retrieved)
        yield "final", self._conclude(sentences, verdicts, chunks)

    def _conclude(self, sentences: list[str], verdicts: list, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not verdicts:
            self.s.metrics.inc("vetoes")
            return _quote_fallback(chunks, faithfulness=0.0)
        faithfulness = round(sum(1 for v in verdicts if v.grounded) / len(verdicts), 4)
        if all(v.grounded for v in verdicts):
            released = [strip_citations(x) for x in sentences]
            return _body(Route.HYBRID_RAG, " ".join(released), citations=_uris(sentences),
                         abstained=released == [ABSTAIN_TEXT], faithfulness=faithfulness, cacheable=True)
        self.s.metrics.inc("vetoes")       # an ungrounded sentence is never released (FMEA #13, ADR-013)
        return self._veto_body(chunks, faithfulness)

    def _veto_body(self, chunks: list[dict[str, Any]], faithfulness: float) -> dict[str, Any]:
        if self.s.verification.method == "nli":      # ADR-013: an unentailed draft means the context does not answer it
            return _body(Route.HYBRID_RAG, ABSTAIN_TEXT, abstained=True, faithfulness=faithfulness)
        return _quote_fallback(chunks, faithfulness=faithfulness)

    async def _generate_streaming(self, prompt, chunks, retrieved) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Token stream -> sentence buffer -> verify each sentence -> release (ADR-011)."""
        s, tr, m = self.s, self.s.tracer, self.s.metrics
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                for delta in s.generator.stream(prompt):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, ("delta", delta))
                loop.call_soon_threadsafe(q.put_nowait, ("end", None))
            except GeneratorError as err:
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(err)))
            except Exception:   # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("error", "stream failed"))

        started = time.perf_counter()
        producer = asyncio.create_task(s.pools["llm_pool"].run(produce))
        buffer, first_delta = "", True
        verified: list[Any] = []
        released_raw: list[str] = []
        vetoed = failed = False

        async def check(sentence: str):
            with tr.span("verify.nli", method=s.verification.method):
                return await s.pools["llm_pool"].run(s.verification.verify_sentence, sentence, retrieved)

        try:
            while True:
                kind, value = await q.get()
                if kind == "error":
                    failed = True
                    break
                if kind == "delta":
                    if first_delta:
                        first_delta = False
                        tr.add_span("ttft", (time.perf_counter() - started) * 1000)
                    buffer += value
                    parts = split_sentences(buffer)
                    if len(parts) > 1:        # all but the last are complete; the tail may still be growing
                        complete, buffer = parts[:-1], parts[-1]
                        for sentence in complete:
                            verdict = await check(sentence)
                            verified.append(verdict)
                            if not verdict.grounded:
                                vetoed = True
                                break
                            released_raw.append(sentence)
                            yield "released", {"index": len(released_raw) - 1, "text": strip_citations(sentence)}
                        if vetoed:
                            break
                    continue
                tail = buffer.strip()      # stream ended: the buffered tail is the last sentence
                if tail:
                    verdict = await check(tail)
                    verified.append(verdict)
                    if verdict.grounded:
                        released_raw.append(tail)
                        yield "released", {"index": len(released_raw) - 1, "text": strip_citations(tail)}
                    else:
                        vetoed = True
                break
        finally:
            stop.set()
            await asyncio.gather(producer, return_exceptions=True)
        tr.add_span("llm.generate", (time.perf_counter() - started) * 1000, streamed=True)

        if failed and not released_raw:
            m.inc("generator_failures")
            body = _quote_fallback(chunks, faithfulness=0.0)
        elif vetoed and not released_raw:
            m.inc("vetoes")
            body = self._veto_body(chunks, round(sum(1 for v in verified if v.grounded) / max(1, len(verified)), 4))
        else:
            if vetoed:
                m.inc("vetoes")
            faithfulness = round(sum(1 for v in verified if v.grounded) / max(1, len(verified)), 4)
            texts = [strip_citations(x) for x in released_raw]
            body = _body(Route.HYBRID_RAG, " ".join(texts), citations=_uris(released_raw),
                         abstained=texts == [ABSTAIN_TEXT], faithfulness=faithfulness,
                         cacheable=not (vetoed or failed))
        if released_raw:
            body["_streamed"] = True           # the sentences were already sent one by one
        yield "final", body

    # ---- bookkeeping ----------------------------------------------------------------------------
    def _finish(self, *, tenant: str, user: str, session: uuid.UUID, trace_id: str, text: str, pii: dict[str, int],
                body: dict[str, Any], t0: float, rec: Any) -> dict[str, Any]:
        s, tr = self.s, self.s.tracer
        trace = body.pop("_trace", [])
        body.pop("_cacheable", None)
        body.pop("_emit_answer", None)
        route = body["route"].value if isinstance(body["route"], Route) else body["route"]
        resp = {"tenant_id": tenant, "trace_id": trace_id, "route": route, "answer": body["answer"],
                "citations": body["citations"], "abstained": body["abstained"], "status": body["status"],
                "approval_id": body["approval_id"], "faithfulness": body["faithfulness"],
                "cache": body.get("cache", "MISS"), "generator": s.generator.name}
        if body.get("degraded"):
            resp["degraded"] = body["degraded"]
        with tr.span("hook.post"):
            s.store.append_audit(tenant, {
                "type": "request", "trace_id": trace_id, "route": route, "status": resp["status"],
                "abstained": resp["abstained"], "faithfulness": resp["faithfulness"], "cache": resp["cache"],
                "pii": pii, "query": text, "citations": resp["citations"], "trace": trace,
                "verifier": s.verification.method})
            if resp["status"] in ("DONE", "PENDING_APPROVAL") and route != Route.REJECT_OOS.value:
                s.store.add_turn(tenant, str(session), "user", text)
                s.store.add_turn(tenant, str(session), "assistant", resp["answer"])
            with tr.span("hook.post.ragas"):
                s.plugins.run_post(resp)
        e2e = (time.perf_counter() - t0) * 1000
        tr.add_span("e2e", e2e, route=route, cache=resp["cache"])
        tr.end(rec, route=route, status=resp["status"], tenant=tenant, e2e_ms=round(e2e, 2))
        resp["_released_fallback"] = [resp["answer"]]
        return resp


def _body(route: Route, answer: str, *, status: str = "DONE", citations: list[str] | None = None,
          abstained: bool = False, approval_id: str | None = None, faithfulness: float = 1.0,
          cacheable: bool = False, cache: str | None = None, _trace: list | None = None,
          degraded: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"route": route, "answer": answer, "citations": citations or [], "abstained": abstained,
                            "status": status, "approval_id": approval_id, "faithfulness": faithfulness,
                            "_cacheable": cacheable, "_trace": _trace or []}
    if cache:
        body["cache"] = cache
    if degraded:
        body["degraded"] = degraded
    return body


def _uris(sentences: list[str]) -> list[str]:
    seen: list[str] = []
    for sentence in sentences:
        for digest in cited_hashes(sentence):
            uri = f"rag://chunks/{digest}"
            if uri not in seen:
                seen.append(uri)
    return seen


def _quote_fallback(chunks: list[dict[str, Any]], faithfulness: float, cacheable: bool = False) -> dict[str, Any]:
    top = chunks[0]
    return _body(Route.HYBRID_RAG, top["text"], citations=[top["uri"]], faithfulness=faithfulness, cacheable=cacheable)
