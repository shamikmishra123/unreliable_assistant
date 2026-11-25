"""Pre-hook guardrails (design §1.3 step 2, NG-4, ADR-010, ADR-015; blueprint Gates 13-14).

PII redaction and the injection shield are regex based and fail closed. Both work on a *normalised
view* of the text so that Unicode tricks do not hide an attack:

* NFKC, zero-width / bidi / control characters removed, non-ASCII digits mapped to ASCII,
  common Cyrillic / Greek / small-capital homoglyphs folded to Latin  (`fold_with_map`)
* for the injection scan only: combining marks stripped, spaced-out letters collapsed
  ("i g n o r e"), leetspeak decoded inside mixed tokens, base64 blobs decoded  (`scan_views`)

PII spans are found in the folded view and mapped back to the ORIGINAL string, so the raw
characters are what get replaced. Nothing is left behind: after redaction the result is scanned
again, and if anything still looks like PII the whole text is replaced (fail closed).

`scan(text, indirect=True)` adds the INDIRECT set (poisoned retrieved content). It runs only on
chunks, tool output and the index-time scan, never on user queries (ADR-010, ADR-015).
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

from .contracts import ErrorCode, ToolError

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

_HOMOGLYPH_SRC = {
    # Cyrillic lower / upper
    "а": "a", "в": "b", "г": "r", "е": "e", "ё": "e", "з": "3", "и": "u", "й": "u", "к": "k", "м": "m",
    "н": "h", "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ї": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w", "ѵ": "v", "ь": "b", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "У": "Y", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "u",
    "χ": "x", "γ": "y", "μ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin small capitals / IPA lookalikes
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ꜰ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j",
    "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r", "ꜱ": "s", "ᴛ": "t", "ᴜ": "u",
    "ᴠ": "v", "ᴡ": "w", "ʏ": "y", "ᴢ": "z", "ɡ": "g", "ɑ": "a", "ȷ": "j", "ı": "i",
}
_HOMOGLYPHS = dict(_HOMOGLYPH_SRC)


def fold_with_map(text: str) -> tuple[str, list[int]]:
    """Fold `text` and return (folded, idx) where idx[i] is the ORIGINAL index of folded char i."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat == "Cf":                       # zero-width, bidi, soft hyphen, tag characters
            continue
        if cat == "Cc":
            if ch == "\n":
                out.append("\n"); idx.append(i)
            elif ch in "\r\t\x0b\x0c":
                out.append(" "); idx.append(i)
            continue
        if cat in ("Zs", "Zl", "Zp"):
            out.append(" "); idx.append(i)
            continue
        if cat == "Nd" and not ch.isascii():
            digit = unicodedata.digit(ch, None)
            if digit is not None:
                out.append(str(digit)); idx.append(i)
                continue
        for c in unicodedata.normalize("NFKC", ch):
            out.append(_HOMOGLYPHS.get(c, c)); idx.append(i)
    return "".join(out), idx


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_TOKEN = re.compile(r"[A-Za-z0-9@$]+")
_ID_TOKEN = re.compile(r"^(?:LOT|SKU|PROT|APR|CAPA)[A-Za-z]*\d", re.I)
_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-_*]){3,}[A-Za-z](?![A-Za-z])")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")


def _decode_leet(view: str) -> str:
    def fix(m: re.Match[str]) -> str:
        tok = m.group(0)
        if len(re.findall(r"[A-Za-z]", tok)) < 2 or not re.search(r"[01345 7@$]", tok.replace(" ", "")):
            return tok
        if _ID_TOKEN.match(tok):              # LOT00044 / SKU99 / PROT7 must survive
            return tok
        return tok.translate(_LEET)
    return _LEET_TOKEN.sub(fix, view)


def _rot13(s: str) -> str:
    return s.translate(str.maketrans("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                     "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM"))


def _clean_view(folded: str) -> str:
    decomposed = unicodedata.normalize("NFKD", folded)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = _SPACED.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group(0)), stripped)
    return stripped.lower()


def scan_views(text: str) -> list[str]:
    """Distinct lowercase views the injection patterns are matched against."""
    folded, _ = fold_with_map(text)
    base = _clean_view(folded)
    views = [base]
    leet = _decode_leet(base)
    if leet != base:
        views.append(leet)
    for m in _B64.finditer(folded):
        try:
            raw = base64.b64decode(m.group(0), validate=False)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded and sum(c.isprintable() or c.isspace() for c in decoded) / len(decoded) > 0.9:
            views.append(_clean_view(decoded))
    if re.search(r"rot-?13", base):
        views.append(_rot13(base))
    return views


# --------------------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


_WORD_DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"}

_DOT_BR = r"\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*|\s*_dot_\s*"      # [dot] (dot) {dot} <dot> _dot_
_DOT_WORD = r"\s+dot\s+"
_DOT_ANY = r"(?:" + _DOT_BR + r"|" + _DOT_WORD + r"|\s*\.\s*)"
_AT_BR = r"(?:\s*[\[\(\{<]\s*(?:at|@)\s*[\]\)\}>]\s*|\s*@\s*|\s*_at_\s*)"
_AT_WORD = r"\s+at\s+"
_LOCAL = r"[A-Za-z0-9][A-Za-z0-9_%+\-]*(?:(?:\.|" + _DOT_BR + r"|" + _DOT_WORD + r")[A-Za-z0-9_%+\-]+)*"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
_TLD = r"[A-Za-z]{2,24}(?![A-Za-z0-9])"
_LB = r"(?<![A-Za-z0-9._%+\-])"

_EMAIL_PATTERNS = [
    # local [at] / @ / (at) label (dot label)+ tld  - dots may be ".", " dot ", "[dot]"
    re.compile(_LB + _LOCAL + _AT_BR + r"(?:" + _LABEL + _DOT_ANY + r")+" + _TLD, re.I),
    # plain word "at" only when a "dot" word or bracketed dot follows ("meet at home.com" is not an email)
    re.compile(_LB + _LOCAL + _AT_WORD + _LABEL + r"(?:(?:" + _DOT_BR + r"|" + _DOT_WORD + r")" + _LABEL + r")+"
               + r"(?![A-Za-z0-9])", re.I),
]

_TOK = (r"(?:(?<![A-Za-z])(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|niner)(?![A-Za-z])|\d)")
_SEP = r"[ \t\n().\-/_]{0,3}"
_RUN = re.compile(r"(?<![A-Za-z0-9])(?<![A-Za-z]-)\+?\(?" + _TOK + r"(?:" + _SEP + _TOK + r")+(?![A-Za-z0-9])", re.I)
_IP = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_SSN_CTX = re.compile(r"(?:ssn|social security|social-security|national id|tax id)(?:\s+(?:number|no\.?|is|#|:))*\W{0,12}$", re.I)
_SSN_AFTER = re.compile(r"^\W{0,4}(?:ssn|social security)", re.I)
_PHONE_CTX = re.compile(r"(?:phone|call|dial|mobile|cell|tel|text|fax|whatsapp|reach|contact)\W{0,16}$", re.I)
_IDENT_CTX = re.compile(r"(?:\bpo|purchase order|order|invoice|ticket|reference|serial|batch|case|account|tracking|"
                        r"shipment|asset|part|model|document|barcode|sku)\s*(?:number|no\.?|#|id)?\W{0,6}$", re.I)
_CARD_CTX = re.compile(r"(?:card|visa|mastercard|amex|credit|debit|pan)\W{0,16}$", re.I)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _digits_and_groups(run: str) -> tuple[str, list[int]]:
    digits: list[str] = []
    groups: list[int] = []
    current = 0
    pos = 0
    for m in re.finditer(r"zero|oh|one|two|three|four|five|six|seven|eight|nine|niner|\d", run, re.I):
        gap = run[pos:m.start()]
        if gap and current:
            groups.append(current); current = 0
        tok = m.group(0).lower()
        if tok.isdigit():
            digits.append(tok); current += 1
        else:
            digits.append(_WORD_DIGITS[tok])
            if current:
                groups.append(current); current = 0
            groups.append(1)
        pos = m.end()
    if current:
        groups.append(current)
    return "".join(digits), groups


def _classify_run(folded: str, m: re.Match[str]) -> str | None:
    run = m.group(0)
    if _IP.match(run.strip("()+")):
        return None
    digits, groups = _digits_and_groups(run)
    n = len(digits)
    plus = run.startswith("+")
    before = folded[max(0, m.start() - 30):m.start()]
    all_single = all(g == 1 for g in groups)

    def nanp(d: str) -> bool:
        d = d[1:] if len(d) == 11 else d
        return len(d) == 10 and d[0] in "23456789"

    if n in (13, 14, 15, 16, 17, 18, 19):
        grouped_ok = tuple(groups[:4]) == (4, 4, 4, 4) or tuple(groups) in {(4, 6, 5), (4, 4, 4, 4, 3), (4, 4, 4, 4, 2)}
        if _luhn(digits) or grouped_ok and 13 <= n <= 16 or _CARD_CTX.search(before):
            return "CARD"
        if plus and 8 <= n <= 15:
            return "PHONE"
        return None
    if n == 9:
        after = folded[m.end():m.end() + 12]
        if tuple(groups) == (3, 2, 4) or all_single or _SSN_CTX.search(before) or _SSN_AFTER.search(after):
            return "SSN"
        return None
    if n in (10, 11):
        bare = tuple(groups) in {(10,), (11,)}
        if bare and not plus and not nanp(digits):
            return None                      # 10-digit IDs such as 1234567890 are not phone numbers
        if bare and not plus and _IDENT_CTX.search(before) and not _PHONE_CTX.search(before):
            return None                      # "PO number 4852554784": an identifier, not a phone
        if plus or all_single or tuple(groups) in {(10,), (11,), (3, 3, 4), (3, 7), (1, 3, 3, 4), (1, 10), (1, 3, 7),
                                                    (3, 3, 2, 2), (3, 2, 2, 3), (3, 4, 3)}:
            return "PHONE"
        return None
    if plus and 8 <= n <= 15 and (all_single or len(groups) <= 5):
        return "PHONE"
    if 7 <= n <= 8 and all_single and re.search(r"(?:phone|call|dial|number|mobile|cell|whatsapp|fax)\W{0,16}$", before, re.I):
        return "PHONE"
    return None


def _find_pii(folded: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in _EMAIL_PATTERNS:
        for m in pattern.finditer(folded):
            spans.append((m.start(), m.end(), "EMAIL"))
    for m in _RUN.finditer(folded):
        kind = _classify_run(folded, m)
        if kind:
            spans.append((m.start(), m.end(), kind))
    return spans


_PLACEHOLDER = {"EMAIL": "[EMAIL]", "SSN": "[SSN]", "CARD": "[CARD]", "PHONE": "[PHONE]"}


def _redact_once(text: str) -> tuple[str, dict[str, int]]:
    folded, idx = fold_with_map(text)
    spans = _find_pii(folded)
    if not spans:
        return text, {}
    # map to ORIGINAL spans, merge overlaps (earliest start, longest wins)
    mapped = sorted(((idx[s], idx[e - 1] + 1, k) for s, e, k in spans), key=lambda t: (t[0], -(t[1] - t[0])))
    merged: list[list] = []
    for s, e, k in mapped:
        if merged and s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            continue
        merged.append([s, e, k])
    out, pos, counts = [], 0, {}
    for s, e, k in merged:
        out.append(text[pos:s])
        out.append(_PLACEHOLDER[k])
        counts[k] = counts.get(k, 0) + 1
        pos = e
    out.append(text[pos:])
    return "".join(out), counts


def redact_pii(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    current = text
    for _ in range(3):
        current, found = _redact_once(current)
        if not found:
            break
        for k, v in found.items():
            counts[k] = counts.get(k, 0) + v
    if _find_pii(fold_with_map(current)[0]):
        # Still looks like PII after three passes: fail closed, nothing of the text is kept.
        return RedactionResult(text="[PII_REMOVED]", counts={**counts, "UNREDACTABLE": 1})
    return RedactionResult(text=current, counts=counts)


# --------------------------------------------------------------------------------------
# Injection shield
# --------------------------------------------------------------------------------------

_I = re.IGNORECASE
_RULE_NOUNS = (r"(?:instructions?|rules?|prompts?|directions?|guidelines?|constraints?|restrictions?|"
               r"polic(?:y|ies)|guardrails?|programming|safeguards?|directives?|commands?|orders?|training|"
               r"limitations?|filters?|safety|settings)")
_OVERRIDE_VERB = r"(?:ignore|disregard|forget|override|bypass|drop|discard|overrule|abandon|neglect|circumvent|" \
                 r"nullify|supersede|void|erase|reset|clear|unlearn|stop following|stop obeying|do not follow|don't follow)"
_SYS_NOUN = r"(?:system|hidden|initial|original|developer|internal|secret|underlying|core|base|full|entire|complete|" \
            r"first|preceding|previous|prior|above|earlier)"

_DIRECT: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(
        r"\b" + _OVERRIDE_VERB + r"\b[^.\n]{0,45}\b(?:previous|prior|above|earlier|preceding|all|any|every|your|"
        r"the|these|those|system|original|initial|existing|current|safety|security|content|operating|"
        r"standing)\b[^.\n]{0,25}\b" + _RULE_NOUNS + r"\b", _I)),
    ("ignore_instructions_short", re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:everything|anything|all)\s+(?:you|that|i|we|above|before|"
        r"previously|you were|you've been|you have been)\b", _I)),
    ("ignore_multilingual", re.compile(
        r"\bignor(?:a|ez|iere?n?|are|ar)\b.{0,25}\b(?:instrucc|instruction|anweisung|istruzion|instru[cç])", _I)),
    ("system_prompt_extraction", re.compile(
        r"\b(?:reveal|show|print|repeat|display|output|dump|leak|recite|echo|expose|disclose|give me|paste|"
        r"copy|write out|tell me|share|send me|list|read out|spell out|return|provide|what(?:'s| is| are| were)|"
        r"translate|summari[sz]e)\b[^.\n]{0,45}\b(?:" + _SYS_NOUN + r")\s+(?:prompt|instructions?|message|rules|"
        r"configuration|config|guidelines?|context|programming|directives?|policy|policies)\b", _I)),
    ("system_prompt_extraction2", re.compile(
        r"\b(?:reveal|print|repeat|dump|leak|recite|echo|expose|disclose|paste|copy|verbatim)\b[^.\n]{0,30}"
        r"\byour\s+(?:prompt|instructions?|rules|programming|guidelines)\b|"
        r"\byour\s+(?:prompt|instructions?|system message|rules)\b[^.\n]{0,30}\b(?:verbatim|word for word|"
        r"exactly as written|in full|in their entirety)\b", _I)),
    ("repeat_above", re.compile(
        r"\b(?:repeat|print|show|output|copy|recite|echo|display|say|write)\b[^.\n]{0,30}\b(?:everything|all|text|"
        r"words|content|message|sentences?|what)\b[^.\n]{0,25}\b(?:above|before (?:this|my)|prior|preceding|"
        r"you were (?:told|given))\b", _I)),
    ("begin_of_context", re.compile(
        r"\b(?:beginning|start|first (?:message|line|lines|words?)|top)\s+of\s+(?:your|this|the)\s+"
        r"(?:context|conversation|prompt|instructions?|chat|session)\b", _I)),
    ("role_override", re.compile(
        r"\b(?:you are now|you're now|you are no longer|you're no longer|from now on,? you|pretend (?:to be|you)|"
        r"act as (?:an? )?(?:unrestricted|unfiltered|uncensored|jailbroken|admin|root|developer|dan|system|"
        r"different|new|evil|rogue|free)|roleplay as|role-play as|simulate (?:an? )?(?:unrestricted|"
        r"unfiltered|uncensored|jailbroken|admin|root|developer|evil)|assume the role of|switch to "
        r"(?:developer|admin|root|god|debug|unrestricted|dan) mode|new persona|your new (?:persona|identity|"
        r"role|name))\b", _I)),
    ("mode_switch", re.compile(
        r"\b(?:developer mode|god mode|jailbreak(?:en|ed)?|do anything now|\bdan\b mode|sudo mode|admin mode|"
        r"root mode|maintenance mode|debug mode|unrestricted mode|test mode enabled|safe mode (?:off|disabled))\b", _I)),
    ("no_restrictions", re.compile(
        r"\b(?:without (?:any |all |the )?(?:restrictions?|filters?|limitations?|guardrails?|safety|rules|"
        r"censorship|constraints?)|no (?:restrictions?|limits|rules|filters|guardrails|censorship) apply|"
        r"unrestricted|uncensored|unfiltered|jailbreak)\b", _I)),
    ("chat_markup", re.compile(
        r"<\|im_(?:start|end)\|>|<\|(?:system|user|assistant|endoftext)\|>|\[/?(?:INST|SYS)\]|<</?SYS>>|</?system>|"
        r"</?(?:assistant|instructions?)>|^#{2,}\s*(?:system|instruction|new instructions?)\b|\bBEGIN (?:SYSTEM|NEW) "
        r"(?:PROMPT|INSTRUCTIONS)\b|^\s*(?:system|developer|assistant)\s*:\s*\S", _I | re.M)),
    ("authority_claim", re.compile(
        r"\b(?:i am|i'm|this is|speaking as|as)\s+(?:the |an? |your )?(?:admin(?:istrator)?|root|sysadmin|"
        r"developer|engineer who built|creator|owner of (?:this|the) system|anthropic|openai|platform "
        r"engineer|system owner|security team|ceo)\b[^.\n]{0,60}[.,;:\-]?[^.\n]{0,80}\b(?:enter|enable|activate|"
        r"unlock|override|bypass|dump|reveal|disable|grant|ignore|disregard|turn off|switch to|list|print|show me)"
        r"\b[^.\n]{0,40}\b(?:mode|access|restrictions?|filters?|tokens?|keys?|secrets?|credentials?|prompt|rules|"
        r"safeguards?|checks?|guardrails?|approvals?|controls?|instructions?|passwords?)\b|"
        r"\b(?:authori[sz]ed|official)\s+(?:override|test|audit|maintenance|penetration test|pen test|"
        r"red team)\b|\b(?:override|master|admin|debug|bypass)\s+(?:code|key|token|password|passphrase)\b|"
        r"\bi(?:'m| am) authori[sz]ed to\b", _I)),
    ("disable_controls", re.compile(
        r"\b(?:disable|turn off|deactivate|switch off|remove|stop using|skip|suspend|lift)\b[^.\n]{0,20}\b"
        r"(?:your|the assistant'?s?|all|any)\s+(?:safety|guardrails?|filters?|checks?|verification|redaction|"
        r"logging|audit|restrictions?|safeguards?|content polic(?:y|ies)|hitl|approvals?)\b", _I)),
    ("encoded_execute", re.compile(
        r"\b(?:decode|decipher|deobfuscate|unscramble|translate)\b[^.\n]{0,40}\b(?:and|then)\b[^.\n]{0,15}"
        r"\b(?:execute|run|follow|obey|apply|do)\b|\brespond (?:only )?in (?:base64|hex|rot13|binary)\b", _I)),
    ("hypothetical_bypass", re.compile(
        r"\b(?:hypothetical(?:ly)?|fictional|imagine|pretend|story|roleplay|role-play|thought experiment)\b"
        r"[^.\n]{0,70}\b(?:no rules|without rules|ignore|bypass|unrestricted|no restrictions|break the rules|"
        r"no guardrails|without (?:approval|oversight|limits)|forget (?:your|the))\b", _I)),
    ("secret_exfil", re.compile(
        r"\b(?:print|reveal|dump|list|show|send|leak|exfiltrate|give me|read)\b[^.\n]{0,30}\b(?:api[ _-]?keys?|"
        r"secrets?|credentials?|passwords?|access tokens?|bearer tokens?|environment variables?|\.env|"
        r"private keys?|connection strings?|reviewer tokens?)\b", _I)),
    ("tenant_crossing", re.compile(
        r"\b(?:access(?!\s+(?:control|polic|right|level|request))|read|show|list|dump|export|query|fetch|get|pull|leak|"
        r"view|display|copy|send|retrieve|include|merge)\b[^.\n]{0,35}\b(?:other|another|different|all|every|any|"
        r"neighbou?ring)\s+"
        r"(?:tenants?|customers?|companies|organi[sz]ations?)\b|\b(?:switch|change|set|use|impersonate|act as)\b"
        r"[^.\n]{0,25}\b(?:tenant_[a-z0-9]+|x-tenant-id|tenant id)\b|\bx-tenant-id\b\s*[:=]|\b(?:list|show|dump|export|read|fetch|query|get|give me|print|display)\b[^.\n]{0,30}\btenant_[a-z0-9]+\b", _I)),
    ("approval_bypass_imperative", re.compile(
        r"^\W*(?:please |just |now |go ahead and |you (?:must|should|will) )?(?:release|approve|mark|set|force|"
        r"push|ship|close|reject|quarantine)\b[^.\n?]{0,60}\b(?:without|skip(?:ping)?|bypass(?:ing)?|no|"
        r"ignoring|waiving)\b[^.\n?]{0,25}\b(?:approval|sign-?off|review(?:er)?|hitl|qa|verification|"
        r"four-eyes|second reviewer)\b", _I)),
    ("new_instructions", re.compile(
        r"\b(?:new|updated|revised|real|actual|true|secret|hidden)\s+(?:instructions?|task|orders?|directives?|"
        r"mission|objective)\s*(?:[:\-]|are|is follows?)|\byour\s+(?:new|real|true|actual)\s+(?:task|goal|"
        r"purpose|objective|job|mission|instructions?)\b|\b(?:end|close) of (?:the )?(?:system )?"
        r"(?:prompt|context|instructions?|document)\b[^.\n]{0,15}\bnew\b", _I)),
    ("obey_following", re.compile(
        r"\b(?:follow|obey|execute|comply with|carry out)\b[^.\n]{0,20}\b(?:the following|these|this)\b"
        r"[^.\n]{0,15}\b(?:instructions?|commands?|orders?|directives?)\b[^.\n]{0,40}\b(?:instead|only|"
        r"above all|first|exactly|regardless)\b", _I)),
    ("priority_override", re.compile(
        r"\b(?:above all|regardless of|despite|overrid(?:es?|ing)|takes? precedence over|supersedes?|"
        r"more important than)\b[^.\n]{0,30}\b(?:previous|prior|your|the|any|all|earlier|system)\b[^.\n]{0,20}"
        r"\b" + _RULE_NOUNS + r"\b", _I)),
]

# Patterns that are only an attack when phrased as a command. A policy QUESTION about the same words
# ("what happens if we skip the sign-off?") is legitimate, so these are skipped on interrogative sentences.
_IMPERATIVE: list[tuple[str, re.Pattern[str]]] = [
    ("skip_review_command", re.compile(
        r"\b(?:skip|bypass|waive|omit|circumvent|ignore|disable)\b[^.\n]{0,25}\b(?:the |a |any |all )?"
        r"(?:reviewer|approvals?|sign-?off|review|hitl|four-eyes|second reviewer|verification|quality check)\b", _I)),
    ("quote_rules", re.compile(
        r"\b(?:quote|recite|repeat|list|state|give me|tell me|read out|print)\b[^.\n]{0,30}\b(?:rules|instructions|"
        r"guidelines|prompt|directives)\b[^.\n]{0,25}\b(?:you were given|you were told|you follow|you have been "
        r"given|given to you|word for word|verbatim|exactly)\b", _I)),
]
_INTERROGATIVE = re.compile(r"^\W*(?:what|how|why|when|where|who|which|can|could|is|are|does|do|did|should|would|"
                            r"will|may|might|am)\b", re.I)

_INDIRECT: list[tuple[str, re.Pattern[str]]] = [
    ("skip_control", re.compile(
        r"\b(?:skip|bypass|waive|omit|circumvent|disregard|forgo|ignore|suspend|override)\b[^.\n]{0,40}\b"
        r"(?:sign[- ]?off|approvals?|reviews?|hitl|quality checks?|qa|verification|four-eyes|second reviewer|"
        r"reviewer|authori[sz]ation|checks?)\b", _I)),
    ("no_control_needed", re.compile(
        r"\b(?:no|without any|without)\s+(?:further |additional |extra |more )?(?:sign[- ]?off|approvals?|"
        r"reviewer|review|authori[sz]ation|qa check|verification)\b[^.\n]{0,15}\b(?:is |are |being )?"
        r"(?:needed|required|necessary|mandatory|expected)\b|\b(?:approval|sign[- ]?off|review) "
        r"(?:is |are )?(?:not |no longer )?(?:needed|required|necessary)\b", _I)),
    ("pre_approved", re.compile(
        r"\b(?:pre-?approved|already approved|auto-?approv(?:e|ed|al)|approval is implied|deemed approved|"
        r"considered approved|is hereby approved|hereby (?:approve|authori[sz]e)s?|approved in advance|"
        r"blanket approval|standing approval)\b", _I)),
    ("requester_voice", re.compile(
        r"\b(?:the requester|the user|the operator|your manager|management|the ceo|the director|i|we)\s+"
        r"(?:has |have |hereby |already )*(?:approved?|authori[sz]e[sd]?|confirm(?:s|ed)?|instructs?|orders?|"
        r"commands?|grants?|certif(?:y|ies)|directs?|demands?)\b[^.\n]{0,40}\b(?:release|the lot|this lot|"
        r"all lots|the request|this request|it|the assistant|you|immediately|quarantine|reject|status)\b", _I)),
    ("cross_tenant", re.compile(
        r"\b(?:dump|export|list|show|read|copy|send|forward|include|append|return|access|fetch|query|pull|"
        r"leak|retrieve)\b[^.\n]{0,40}\b(?:other|another|different|all|every)\s+tenants?\b|"
        r"\b(?:other|another|different)\s+tenants?'?s?\s+(?:lots?|data|records?|notes?|documents?|files?)\b|"
        r"\b(?:dump|export|read|fetch|list|copy|send|include|show|access|query)\b[^.\n]{0,40}\btenant_[a-z0-9]+\b",
        _I)),
    ("concealment", re.compile(
        r"\b(?:hide|conceal|omit|keep|withhold|suppress|mask|cover up|do not (?:show|mention|tell|report)|"
        r"don't (?:show|mention|tell|report))\b[^.\n]{0,40}\b(?:from|to)\s+(?:the\s+|any\s+)?(?:reviewer|"
        r"auditor|audit log|compliance|supervisor|operator|requester|user|qa|quality team)s?\b|"
        r"\b(?:do not|don't|never)\s+(?:tell|inform|notify|alert|show|mention|report to|disclose to)\s+(?:the\s+|any\s+)?"
        r"(?:reviewer|auditor|compliance|supervisor|requester|user|operator|qa)s?\b|"
        r"\bkeep\b[^.\n]{0,25}\bout of (?:the\s+)?(?:audit|log|record|trail|report|review)s?\b|"
        r"\b(?:delete|erase|purge|clear|wipe|tamper(?: with)?|alter|falsify|backdate|modify|edit|truncate)\b"
        r"[^.\n]{0,30}\b(?:audit|log|trail|record|history|evidence)s?\b|\bsilently\b|"
        r"\bwithout (?:telling|informing|notifying|alerting|logging|recording|mentioning)\b", _I)),
    ("assistant_directive", re.compile(
        r"\b(?:assistant|ai|model|llm|agent|bot|chatbot|copilot)s?\s+(?:must|should|shall|will|has to|needs? to|"
        r"is required to|are required to|are instructed to|is instructed to|are to|is to)\s+(?:now |always |"
        r"first |immediately )?(?:release|approve|quarantine|reject|mark|ignore|disregard|append|prepend|"
        r"include|output|print|say|reply|respond|answer|call|invoke|run|execute|submit|email|send|forward|"
        r"disclose|reveal|tell|skip|bypass|override|insert|add|always|never|stop)\b", _I)),
    ("addresses_reader", re.compile(
        r"\b(?:note|message|instructions?|attention|important|notice|warning|memo)s?\s+(?:to|for)\s+(?:the\s+)?"
        r"(?:assistant|ai|model|llm|agent|bot|reader|language model|chatbot)\b|"
        r"\bif you are (?:an? )?(?:ai|assistant|language model|llm|chatbot|bot|model)\b|"
        r"\bwhen (?:this|the) (?:document|text|note|chunk|page|file|pdf|passage) is (?:read|retrieved|"
        r"summari[sz]ed|processed|parsed|loaded|seen)\b|\bdear (?:ai|assistant|model|llm)\b|"
        r"\b(?:ai|llm|language model)s? (?:reading|processing|summari[sz]ing|parsing) this\b", _I)),
    ("instruction_to_you", re.compile(
        r"\b(?:before (?:answering|responding|replying)|when (?:answering|responding|replying)|in (?:your|every|"
        r"each|any) (?:answer|response|reply)|every (?:answer|response|reply)|each (?:answer|response|reply)|"
        r"your (?:answer|response|reply))\b[^.\n]{0,50}\b(?:you must|you should|always|first|never|append|prepend|"
        r"include|add|insert|mention|say|begin with|start with|end with|state|write)\b|"
        r"\b(?:append|prepend|insert|add)\b[^.\n]{0,40}\b(?:to|at the (?:start|end) of)\s+(?:every|each|your|all)\s+"
        r"(?:answer|response|reply|output|message)s?\b", _I)),
    ("tool_invocation", re.compile(
        r"\b(?:call|invoke|run|execute|use|trigger|issue|perform|make)\b[^.\n]{0,25}\b(?:submit_transaction|"
        r"put_session_fact|get_session_facts|lookup_lot|hybrid_retrieve|graph_subgraph|get_approval|the "
        r"(?:tool|function|api))\b|<tool_call>|\"(?:tool|function_call|tool_name)\"\s*:", _I)),
    ("hidden_markup", re.compile(
        r"<!--[^>]{0,400}?(?:assistant|instruction|ignore|system|you must|override|approve|release)[^>]{0,400}?-->|"
        r"\[(?:system|admin|assistant|hidden|internal|developer)[^\]]{0,40}\]|!\[[^\]]{0,80}\]\(https?://|"
        r"<(?:script|iframe|img|style)\b|\bdisplay\s*:\s*none\b|\bfont-size\s*:\s*0|\bcolor\s*:\s*white\b", _I)),
    ("auto_release", re.compile(
        r"\b(?:release|approve|ship|quarantine|reject|close|mark)\b[^.\n]{0,30}\b(?:the lot|this lot|all lots|"
        r"lots?|the batch|the request|it|them)\b[^.\n]{0,20}\b(?:immediately|right away|directly|automatically|"
        r"at once|without (?:delay|waiting|asking|checking)|now)\b", _I)),
    ("exfil_secrets", re.compile(
        r"\b(?:print|reveal|dump|list|send|leak|exfiltrate|email|forward|post|upload)\b[^.\n]{0,40}\b(?:api[ _-]?keys?|"
        r"secrets?|credentials?|passwords?|tokens?|system prompt|environment variables?|\.env|private keys?)\b", _I)),
]

_HIDDEN_CHARS = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]")


def scan(text: str, indirect: bool = False) -> list[str]:
    """Return the names of the patterns that matched."""
    hits: list[str] = []
    for view in scan_views(text):
        for name, pattern in _DIRECT:
            if name not in hits and pattern.search(view):
                hits.append(name)
        if indirect:
            for name, pattern in _INDIRECT:
                if name not in hits and pattern.search(view):
                    hits.append(name)
    for view in scan_views(text):
        for sentence in re.split(r"(?<=[.!?\n])\s*", view):
            sentence = sentence.strip()
            if not sentence or sentence.endswith("?") or _INTERROGATIVE.match(sentence):
                continue
            for name, pattern in _IMPERATIVE:
                if name not in hits and pattern.search(sentence):
                    hits.append(name)
    if indirect and _HIDDEN_CHARS.search(text):
        hits.append("hidden_characters")
    return hits


def check_user_query(text: str) -> None:
    """Fail-closed direct-injection check on the redacted user query (ADR-010)."""
    hits = scan(text)
    if hits:
        raise ToolError(ErrorCode.INJECTION_BLOCKED, "query rejected by injection shield",
                        {"patterns": hits})


def is_poisoned(text: str) -> bool:
    """Index-time and tool-output check (ADR-010, ADR-015). Uses both pattern sets."""
    return bool(scan(text, indirect=True))
