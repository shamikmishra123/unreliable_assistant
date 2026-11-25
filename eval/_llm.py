"""Shared LLM helpers for the eval scripts (Anthropic API). Not part of the product."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import anthropic

PRICES_PER_M = {"claude-haiku-5-5": (0.10, 0.50)}   # (input, output) USD per 1M tokens. Others: unknown here.


class Usage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_model: dict[str, dict[str, int]] = {}

    def add(self, model: str, inp: int, out: int) -> None:
        with self._lock:
            d = self.by_model.setdefault(model, {"input": 0, "output": 0, "calls": 0})
            d["input"] += inp
            d["output"] += out
            d["calls"] += 1


USAGE = Usage()
_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=90, max_retries=3)
    return _client


def ask(model: str, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> tuple[str, dict[str, int]]:
    """One call. Returns (text, usage). Raises on persistent failure."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = client().messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                            system=system, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            u = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
            USAGE.add(model, u["input"], u["output"])
            return text, u
        except Exception as err:   # noqa: BLE001 - retried, then re-raised
            last = err
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


_JSON = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json(text: str) -> Any | None:
    """Parse the first JSON object/array in a reply. None when unreadable (callers must count that)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text.strip(), flags=re.M).strip()
    for candidate in (text, (_JSON.search(text) or [None])[0] if _JSON.search(text) else None):
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None
