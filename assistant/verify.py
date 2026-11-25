"""Verification (design ADR-013 default): lexical claim-to-chunk check, no LLM.

A sentence is grounded when at least 80% of its content words appear in the cited
chunk (or, if uncited, in any retrieved chunk), every number/ID it contains appears
verbatim in that source, and any citation points at a chunk that was actually retrieved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[?rag://chunks/([a-f0-9]{64})\]?")
# A sentence ends at . ! ? plus any citations attached to it, when the next sentence starts (capital, digit, quote).
_BOUNDARY = re.compile(r"[.!?](?:\s*\[rag://chunks/[a-f0-9]{64}\])*(?=\s+[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z][a-z-]*|\d[\w.,/-]*", re.IGNORECASE)
STOPWORDS = frozenset("""
a an and are as at be been but by can do does for from has have he her his i if in into is it its
of on or our she so that the their them then there these they this to too under was we were what when
where which while who will with would you your not no yes also each any all may must should only
""".split())
SUPPORT_THRESHOLD = 0.8


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    grounded: bool
    reason: str


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+", " ", CITATION.sub("", sentence)).strip()


def cited_hashes(sentence: str) -> list[str]:
    return CITATION.findall(sentence)


def _content_words(text: str) -> set[str]:
    words = {w.lower() for w in _WORD.findall(text)}
    return {w for w in words if w not in STOPWORDS and not w[0].isdigit() and len(w) > 2}


def _numbers_and_ids(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z]+-\d+|\d+(?:[.,]\d+)?", text)}


ENTAIL_THRESHOLD = 0.5


def check_sentence(sentence: str, retrieved: dict[str, str], abstain_text: str,
                   entail=None, entail_threshold: float = ENTAIL_THRESHOLD) -> SentenceVerdict:
    """`retrieved` maps chunk hash to chunk text for every chunk the orchestrator showed the model.

    Without `entail`: lexical support >= 0.8 (ADR-013 stopgap). With `entail(premise, hypothesis) -> p`: the exact
    number/ID guard still applies, then the sentence must be entailed by the cited chunk(s) (or any retrieved chunk)."""
    clean = strip_citations(sentence)
    if not clean:
        return SentenceVerdict(sentence, True, "no claim")
    if clean == abstain_text:
        return SentenceVerdict(sentence, True, "abstention")
    hashes = cited_hashes(sentence)
    if any(h not in retrieved for h in hashes):
        return SentenceVerdict(sentence, False, "citation to a chunk that was not retrieved")
    sources = [retrieved[h] for h in hashes] if hashes else list(retrieved.values())
    source_text = "\n".join(sources)
    content = _content_words(clean)
    numbers = _numbers_and_ids(clean)
    if not content and not numbers:
        return SentenceVerdict(sentence, True, "no checkable content")
    source_words = {w.lower() for w in _WORD.findall(source_text)}
    if numbers and not numbers <= _numbers_and_ids(source_text):
        return SentenceVerdict(sentence, False, "number or ID not in source")
    if not content:
        return SentenceVerdict(sentence, True, "numbers only, all present")
    if entail is not None:
        best = max((entail(src, clean) for src in (sources if hashes else list(retrieved.values()))), default=0.0)
        if best >= entail_threshold:
            return SentenceVerdict(sentence, True, f"entailment {best:.2f}")
        return SentenceVerdict(sentence, False, f"entailment {best:.2f} below {entail_threshold}")
    support = len(content & source_words) / len(content)
    if support >= SUPPORT_THRESHOLD:
        return SentenceVerdict(sentence, True, f"support {support:.2f}")
    return SentenceVerdict(sentence, False, f"support {support:.2f} below {SUPPORT_THRESHOLD}")


def check_draft(sentences: list[str], retrieved: dict[str, str], abstain_text: str, entail=None) -> list[SentenceVerdict]:
    return [check_sentence(s, retrieved, abstain_text, entail) for s in sentences]
