"""NLI claim-to-chunk scorer for the Verification Subagent (ADR-013, ADR-016).

Cross-encoder NLI model; `entail(premise, hypothesis)` returns P(entailment). Loaded lazily and once per process.
"""
from __future__ import annotations

import threading

import numpy as np

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class NliScorer:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self._ent = next(i for i, label in id2label.items() if label.startswith("entail"))
        self._lock = threading.Lock()

    def entail(self, premise: str, hypothesis: str) -> float:
        with self._lock:
            logits = self.model.predict([(premise, hypothesis)], convert_to_numpy=True, show_progress_bar=False)
        z = np.asarray(logits, dtype="float64").reshape(-1)
        z = np.exp(z - z.max())
        return float((z / z.sum())[self._ent])


_scorer: NliScorer | None = None
_lock = threading.Lock()


def get_scorer(model_name: str = DEFAULT_NLI_MODEL) -> NliScorer:
    global _scorer
    with _lock:
        if _scorer is None or _scorer.model_name != model_name:
            _scorer = NliScorer(model_name)
        return _scorer
