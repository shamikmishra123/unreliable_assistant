"""Execution Subagent (design §1.3 TRANSACTION_EXEC, ADR-004).

A bounded ReAct state machine: INIT → TRIAGE → REASONING → TOOL_CALL → OBSERVATION → VERIFY → DONE | HALTED | HALTED_BREAKER.
Every TOOL_CALL goes through the ToolExecutionHarness. A Tier 3 mutation becomes submit_transaction, which only
files a PENDING_APPROVAL record. The planner is deterministic (no LLM): read-before-write, then submit.
"""
from __future__ import annotations

import contextvars
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import CallContext, ErrorCode, LOT_STATUSES, ToolError, extract_lot_ids
from .harness import Tier, ToolExecutionHarness, canonical_hash


class State(str, Enum):
    INIT = "INIT"
    TRIAGE = "TRIAGE"
    REASONING = "REASONING"
    TOOL_CALL = "TOOL_CALL"
    OBSERVATION = "OBSERVATION"
    VERIFY = "VERIFY"
    DONE = "DONE"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    HALTED = "HALTED"
    HALTED_BREAKER = "HALTED_BREAKER"


ALLOWED = {
    State.INIT: {State.TRIAGE},
    State.TRIAGE: {State.REASONING},
    State.REASONING: {State.TOOL_CALL, State.VERIFY},
    State.TOOL_CALL: {State.OBSERVATION},
    State.OBSERVATION: {State.REASONING},
    State.VERIFY: {State.DONE, State.PENDING_APPROVAL, State.HALTED, State.HALTED_BREAKER},
}


@dataclass
class Observation:
    tool: str
    ok: bool
    result: dict[str, Any] | None = None
    code: str | None = None
    message: str | None = None

    def digest(self) -> str:
        return canonical_hash({"tool": self.tool, "ok": self.ok, "result": self.result, "code": self.code})


@dataclass(frozen=True)
class Plan:
    thought: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    answer: str | None = None
    state: State = State.DONE
    approval_id: str | None = None


@dataclass
class RunResult:
    state: State
    answer: str
    approval_id: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)


_ACTION_VERBS = [
    ("quarantine", re.compile(r"\bquarantin", re.I)),
    ("release", re.compile(r"\brelease", re.I)),
    ("hold", re.compile(r"\bhold\b", re.I)),
    ("reject", re.compile(r"\breject", re.I)),
]


@dataclass(frozen=True)
class Intent:
    kind: str                       # "read" | "write" | "clarify"
    lot_id: str | None = None
    action: str | None = None
    payload: dict[str, Any] | None = None
    clarify: str | None = None


def parse_intent(query: str) -> Intent:
    lots = extract_lot_ids(query)
    q = query.lower()
    if not lots and not re.search(r"open (a |an )?(deviation|capa)", q):
        return Intent("clarify", clarify="Which lot ID should I act on? Use the form LOT-00000.")
    lot = lots[0] if lots else None
    if re.search(r"open (a |an )?deviation", q) or re.search(r"open (a |an )?capa", q):
        action = "open_capa" if re.search(r"capa", q) else "open_deviation"
        tail = re.split(r"(?:deviation|capa)\b", query, maxsplit=1, flags=re.I)[-1]
        tail = re.sub(r"\bLOT[\s_-]?\d{1,5}\b", "", tail, flags=re.I)
        description = re.sub(r"^\W*(on|for|against|about)?\W*", "", tail, flags=re.I).strip()
        if not lot or not description:
            return Intent("clarify", clarify="Give the lot ID and a short description of the deviation.")
        return Intent("write", lot, action, {"lot_id": lot, "description": description[:1000]})
    status = re.search(r"status (?:to|as) (on hold|available|quarantined|released|rejected)\b", q)
    if status:
        target = status.group(1).upper().replace(" ", "_")
        if not lot or target not in LOT_STATUSES:
            return Intent("clarify", clarify=f"Give a lot ID and a valid status: {', '.join(LOT_STATUSES)}.")
        return Intent("write", lot, "update_status", {"lot_id": lot, "status": target})
    found = [(m.start(), name) for name, pattern in _ACTION_VERBS if (m := pattern.search(q))]
    if found and lot:
        action = min(found)[1]
        return Intent("write", lot, action, {"lot_id": lot})
    return Intent("read", lot)


class TransactionPlanner:
    """Deterministic: lookup first (read-before-write), then submit, then answer from the observations."""

    def __init__(self, query: str):
        self.intent = parse_intent(query)
        self.justification = query[:512]

    def next(self, observations: list[Observation]) -> Plan:
        intent = self.intent
        if intent.kind == "clarify":
            return Plan("ask for the missing detail", answer=intent.clarify, state=State.DONE)
        last = observations[-1] if observations else None
        if last is not None and not last.ok:
            if last.code == ErrorCode.CIRCUIT_OPEN.name:
                return Plan("breaker tripped", answer="I stopped after repeated failures with the same request. "
                            "Please check the lot ID and try again.", state=State.HALTED_BREAKER)
            if last.code == ErrorCode.TENANT_SCOPE_VIOLATION.name:
                return Plan("scope refused", answer="That lot is not available to your tenant.", state=State.HALTED)
            if last.code == ErrorCode.RBAC_DENIED.name:
                return Plan("rbac refused", answer="Your role cannot perform that action.", state=State.HALTED)
            return Plan("tool failed", answer=f"I could not complete the lookup ({last.code}). Nothing was changed.",
                        state=State.HALTED)

        lookup = next((o for o in observations if o.tool == "lookup_lot" and o.ok), None)
        if lookup is None:
            return Plan("read the lot before anything else", tool="lookup_lot", args={"lot_id": intent.lot_id})
        if not lookup.result or not lookup.result.get("found"):
            return Plan("lot record checked", answer=f"No record exists for {intent.lot_id} in your tenant. "
                        "Nothing was filed.", state=State.DONE)
        if intent.kind == "read":
            r = lookup.result
            return Plan("answer from the lot record", answer=(
                f"{r['lot_id']} is {r['status']} under protocol {r['protocol_ref'] or 'none on record'}, "
                f"last updated {r['updated_at']}."), state=State.DONE)

        submit = next((o for o in observations if o.tool == "submit_transaction" and o.ok), None)
        if submit is None:
            return Plan("file the change as a request; a reviewer must approve it",
                        tool="submit_transaction",
                        args={"action": intent.action, "payload": intent.payload,
                              "justification": self.justification})
        approval_id = submit.result["approval_id"]
        return Plan("request filed", answer=(
            f"Request {approval_id} to {intent.action.replace('_', ' ')} {intent.lot_id} is pending reviewer "
            "approval. No change has been made yet."), state=State.PENDING_APPROVAL, approval_id=approval_id)


_STEP_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="react-step")


class ReActRunner:
    def __init__(self, harness: ToolExecutionHarness, make_ctx: Callable[[], CallContext],
                 caller_tier: Tier = Tier.TIER_2_COMPUTE, max_iterations: int = 5,
                 budget_s: float = 25.0, clock: Callable[[], float] = time.monotonic,
                 step_timeout_s: float | None = None, tracer: Any = None):
        self.step_timeout_s = step_timeout_s
        self.tracer = tracer
        self.harness = harness
        self.make_ctx = make_ctx
        self.caller_tier = caller_tier
        self.max_iterations = max_iterations
        self.budget_s = budget_s
        self.clock = clock
        self.state = State.INIT

    def _call(self, tool: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """One tool call. With `step_timeout_s` the call runs on a worker and a slow step becomes UPSTREAM_TIMEOUT
        (Gate 4 per_step_timeout); a late result is discarded, never applied."""
        ctx = self.make_ctx()
        if self.step_timeout_s is None:
            return self.harness.call(tool, args, ctx, self.caller_tier)
        snapshot = contextvars.copy_context()
        fut = _STEP_POOL.submit(snapshot.run, self.harness.call, tool, args, ctx, self.caller_tier)
        try:
            return fut.result(self.step_timeout_s)
        except FutureTimeout:
            fut.cancel()
            raise ToolError(ErrorCode.UPSTREAM_TIMEOUT, f"{tool} exceeded the {self.step_timeout_s:.2f}s step budget") from None

    def _move(self, to: State) -> None:
        if to not in ALLOWED.get(self.state, set()):
            raise RuntimeError(f"illegal transition {self.state} -> {to}")
        self.state = to

    def run(self, planner: TransactionPlanner) -> RunResult:
        self._move(State.TRIAGE)
        self._move(State.REASONING)
        started = self.clock()
        observations: list[Observation] = []
        steps: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        while True:
            plan = planner.next(observations)
            if plan.tool is None:
                self._move(State.VERIFY)
                final = plan.state
                self.state = final
                return RunResult(final, plan.answer or "", plan.approval_id, steps)
            if len(steps) >= self.max_iterations:
                self._move(State.VERIFY)
                self.state = State.HALTED
                return RunResult(State.HALTED, "I stopped after the maximum number of steps. "
                                 "Please narrow the request.", None, steps)
            if self.clock() - started > self.budget_s:
                self._move(State.VERIFY)
                self.state = State.HALTED
                return RunResult(State.HALTED, "The request ran out of time. Nothing further was changed.",
                                 None, steps)
            last_digest = observations[-1].digest() if observations else ""
            signature = (plan.tool, canonical_hash({"args": plan.args, "obs": last_digest}))
            if signature in seen:   # no state change since the same call: do not loop (NG-6)
                self._move(State.VERIFY)
                self.state = State.HALTED
                return RunResult(State.HALTED, "I stopped because the next step repeated without new information.",
                                 None, steps)
            seen.add(signature)

            self._move(State.TOOL_CALL)
            step_started = time.perf_counter()
            try:
                result = self._call(plan.tool, plan.args)
                obs = Observation(plan.tool, True, result=result)
            except ToolError as err:
                obs = Observation(plan.tool, False, code=err.code.name, message=err.message)
            if self.tracer is not None:
                self.tracer.add_span("agent.react.step", (time.perf_counter() - step_started) * 1000, tool=plan.tool)
            self._move(State.OBSERVATION)
            observations.append(obs)
            steps.append({"thought": plan.thought, "tool": plan.tool, "arguments": plan.args,
                          "outcome": obs.code or "OK"})
            self._move(State.REASONING)
