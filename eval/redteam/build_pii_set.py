"""Builds eval/redteam/pii.jsonl (KPI-02): >= 500 labelled items, seeded, same author as the detector.

Each item: {"id","kind","variant","text","pii":[raw strings that must not survive redaction]}.
Negatives have "pii": [] and "kind": "none" - redaction must leave them unchanged.
"""
import json
import random
from pathlib import Path

rng = random.Random(20261125)
OUT = Path(__file__).with_name("pii.jsonl")
items = []
ZW = "​"


def add(kind, variant, text, pii):
    items.append({"id": f"pii-{len(items)+1:03d}", "kind": kind, "variant": variant, "text": text, "pii": pii})


def luhn_complete(prefix, length):
    digits = [int(c) for c in prefix] + [rng.randrange(10) for _ in range(length - len(prefix) - 1)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return "".join(map(str, digits)) + str((10 - total % 10) % 10)


def zw(s, n=2):
    s = list(s)
    for _ in range(n):
        s.insert(rng.randrange(1, len(s)), ZW)
    return "".join(s)


FULL = {str(i): chr(0xFF10 + i) for i in range(10)}
ARAB = {str(i): chr(0x0660 + i) for i in range(10)}
DEVA = {str(i): chr(0x0966 + i) for i in range(10)}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
conv = lambda s, m: "".join(m.get(c, c) for c in s)
spell = lambda s, sep=" ": sep.join(WORDS[int(c)] for c in s if c.isdigit())
spaced = lambda s: " ".join(c for c in s if c.isdigit())

LOCALS = ["jane.roe", "bob.smith", "maria_garcia", "li.wei", "a.kumar", "tom", "sarah.connor", "dev.ops", "k.nowak",
          "olu.adeyemi", "p.rossi", "hans.mueller", "qa.lead", "r.tanaka", "m.silva"]
DOMAINS = [("example", "com"), ("corp.example", "org"), ("mail", "co.uk"), ("plant-two", "net"), ("gmail", "com"),
           ("outlook", "com"), ("acme", "io"), ("riverside-plant", "com")]
CTX = ["Please contact {x} about LOT-00044.", "Send the report to {x}.", "My details: {x}", "cc {x} on the deviation",
       "{x} is the reviewer for this request.", "Reach me at {x} after the shift.", "Forward LOT-88392 status to {x} please",
       "Owner: {x}"]


def email_forms(local, dom, tld):
    d1, dm = dom, dom.split(".")
    std = f"{local}@{dom}.{tld}"
    return {
        "standard": std,
        "bracket_at_dot": f"{local}[at]{dom.replace('.', '[dot]')}[dot]{tld.replace('.', '[dot]')}",
        "paren_at_dot": f"{local} (at) {dom.replace('.', ' (dot) ')} (dot) {tld.replace('.', ' (dot) ')}",
        "brace_at_dot": f"{local}{{at}}{dom.replace('.', '{dot}')}{{dot}}{tld.replace('.', '{dot}')}",
        "words_at_dot": f"{local.replace('.', ' dot ')} at {dom.replace('.', ' dot ')} dot {tld.replace('.', ' dot ')}",
        "upper_words": f"{local.upper()} AT {dom.upper().replace('.', ' DOT ')} DOT {tld.upper().replace('.', ' DOT ')}",
        "underscore": f"{local}_at_{dom.replace('.', '_dot_')}_dot_{tld.replace('.', '_dot_')}",
        "zero_width": zw(std, 3),
        "fullwidth": "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in std),
        "spaced_at": f"{local} @ {dom}.{tld}",
    }


# ---- positives: email --------------------------------------------------------------------
variants = ["standard"] * 20 + ["bracket_at_dot"] * 8 + ["paren_at_dot"] * 7 + ["brace_at_dot"] * 4 + ["words_at_dot"] * 9 \
    + ["upper_words"] * 4 + ["underscore"] * 4 + ["zero_width"] * 6 + ["fullwidth"] * 3 + ["spaced_at"] * 3
variants = variants + variants[:25]
for v in variants:
    local = rng.choice(LOCALS)
    dom, tld = rng.choice(DOMAINS)
    raw = email_forms(local, dom, tld)[v]
    add("email", v, rng.choice(CTX).format(x=raw), [raw])

# ---- phone -----------------------------------------------------------------------------
def rand_phone():
    return rng.choice("23456789") + "".join(rng.choice("0123456789") for _ in range(2)), \
        "".join(rng.choice("0123456789") for _ in range(3)), "".join(rng.choice("0123456789") for _ in range(4))


PH_CTX = ["Call {x} to confirm the lot.", "Operator phone: {x}", "my number is {x}, which lot is ready?", "Text {x} when LOT-00044 ships",
          "Escalate to {x} after hours.", "reach the night shift on {x}", "{x} - ask for the QA lead"]
fmts = {
    "dashes": lambda a, b, c: f"{a}-{b}-{c}", "parens": lambda a, b, c: f"({a}) {b}-{c}", "dots": lambda a, b, c: f"{a}.{b}.{c}",
    "plus1_spaces": lambda a, b, c: f"+1 {a} {b} {c}", "plus1_dashes": lambda a, b, c: f"+1-{a}-{b}-{c}",
    "bare": lambda a, b, c: f"{a}{b}{c}", "spaces": lambda a, b, c: f"{a} {b} {c}",
    "spaced_digits": lambda a, b, c: spaced(a + b + c), "spelled": lambda a, b, c: spell(a + b + c),
    "spelled_dashed": lambda a, b, c: f"{spell(a)} - {spell(b)} - {spell(c)}",
    "fullwidth": lambda a, b, c: conv(f"{a}-{b}-{c}", FULL), "arabic_indic": lambda a, b, c: conv(f"{a} {b} {c}", ARAB),
    "devanagari": lambda a, b, c: conv(f"{a}-{b}-{c}", DEVA), "zero_width": lambda a, b, c: zw(f"{a}-{b}-{c}", 3),
    "intl_uk": lambda a, b, c: "+44 20 7946 0958", "intl_de": lambda a, b, c: "+49 30 12345678", "intl_in": lambda a, b, c: "+91 98765 43210",
}
pv = (["dashes"] * 12 + ["parens"] * 10 + ["dots"] * 8 + ["plus1_spaces"] * 6 + ["plus1_dashes"] * 5 + ["bare"] * 8 + ["spaces"] * 6
      + ["spaced_digits"] * 6 + ["spelled"] * 8 + ["spelled_dashed"] * 4 + ["fullwidth"] * 5 + ["arabic_indic"] * 4 + ["devanagari"] * 3
      + ["zero_width"] * 5 + ["intl_uk", "intl_de", "intl_in"] * 2)
pv = pv + pv[:32]
for v in pv:
    a, b, c = rand_phone()
    raw = fmts[v](a, b, c)
    add("phone", v, rng.choice(PH_CTX).format(x=raw), [raw])

# ---- ssn ---------------------------------------------------------------------------------
SS_CTX = ["badge ssn {x} for the operator", "Employee SSN: {x}", "my social security number is {x}", "ssn {x} was typed in by mistake",
          "Please verify {x} (SSN) against the record"]
ssn_f = {
    "dashes": lambda s: f"{s[:3]}-{s[3:5]}-{s[5:]}", "spaces": lambda s: f"{s[:3]} {s[3:5]} {s[5:]}", "dots": lambda s: f"{s[:3]}.{s[3:5]}.{s[5:]}",
    "bare_ctx": lambda s: s, "spaced_digits": lambda s: spaced(s), "spelled": lambda s: spell(s),
    "fullwidth": lambda s: conv(f"{s[:3]}-{s[3:5]}-{s[5:]}", FULL), "zero_width": lambda s: zw(f"{s[:3]}-{s[3:5]}-{s[5:]}", 3),
    "arabic_indic": lambda s: conv(f"{s[:3]}-{s[3:5]}-{s[5:]}", ARAB),
}
sv = ["dashes"] * 12 + ["spaces"] * 6 + ["dots"] * 4 + ["bare_ctx"] * 8 + ["spaced_digits"] * 5 + ["spelled"] * 5 + ["fullwidth"] * 4 + ["zero_width"] * 4 + ["arabic_indic"] * 2
sv = sv + sv[:16]
for v in sv:
    s = "".join(rng.choice("0123456789") for _ in range(9))
    s = rng.choice("123456789") + s[1:]
    raw = ssn_f[v](s)
    add("ssn", v, rng.choice(SS_CTX).format(x=raw), [raw])

# ---- credit cards -----------------------------------------------------------------------
CD_CTX = ["card {x} was charged for the sample kit", "pay with {x}", "Visa {x} exp 11/28", "my credit card number is {x}", "billing card: {x}"]
cv = ["spaces"] * 10 + ["dashes"] * 8 + ["bare"] * 8 + ["amex"] * 6 + ["dots"] * 4 + ["spaced_digits"] * 4 + ["zero_width"] * 4 + ["fullwidth"] * 3 + ["spelled"] * 3
cv = cv + cv[:16]
for v in cv:
    if v == "amex":
        n = luhn_complete(rng.choice(["34", "37"]), 15)
        raw = f"{n[:4]} {n[4:10]} {n[10:]}"
    else:
        n = luhn_complete(rng.choice(["4", "51", "52", "53", "55"]), 16)
        raw = {"spaces": f"{n[:4]} {n[4:8]} {n[8:12]} {n[12:]}", "dashes": f"{n[:4]}-{n[4:8]}-{n[8:12]}-{n[12:]}", "bare": n,
               "dots": f"{n[:4]}.{n[4:8]}.{n[8:12]}.{n[12:]}", "spaced_digits": spaced(n), "zero_width": zw(f"{n[:4]} {n[4:8]} {n[8:12]} {n[12:]}", 3),
               "fullwidth": conv(f"{n[:4]} {n[4:8]} {n[8:12]} {n[12:]}", FULL), "spelled": spell(n)}[v]
    add("card", v, rng.choice(CD_CTX).format(x=raw), [raw])

# ---- multi-PII sentences ----------------------------------------------------------------
for _ in range(22):
    a, b, c = rand_phone()
    local = rng.choice(LOCALS)
    dom, tld = rng.choice(DOMAINS)
    e = f"{local}@{dom}.{tld}"
    p = f"({a}) {b}-{c}"
    add("multi", "email+phone", f"Operator {e}, phone {p}, reported a deviation on LOT-00099.", [e, p])

# ---- negatives (150): nothing here is PII; redaction must leave the text unchanged ----------------
neg = []
neg += [f"Verify LOT-{rng.randrange(0, 99999):05d} against protocol PROT-SAF-{rng.randrange(1, 12)}." for _ in range(16)]
neg += [f"What is the shelf life of SKU-{rng.randrange(1, 999):03d}?" for _ in range(10)]
neg += [f"The batch was packed on 2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d} at {rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:00." for _ in range(14)]
neg += [f"We are on version {rng.randrange(1, 12)}.{rng.randrange(0, 20)}.{rng.randrange(0, 20)} of the SOP." for _ in range(10)]
neg += [f"Revision {rng.randrange(1, 40)} of document SOP-QA-{rng.randrange(1, 999):04d} applies." for _ in range(10)]
neg += [f"Store between {rng.randrange(1, 8)} and {rng.randrange(8, 30)} degrees Celsius for {rng.randrange(1, 60)} months." for _ in range(10)]
neg += [f"Lots {rng.randrange(1, 99)}, {rng.randrange(1, 99)}, {rng.randrange(1, 99)}, {rng.randrange(1, 99)} and {rng.randrange(1, 99)} are on hold." for _ in range(8)]
neg += [f"There are {rng.randrange(100, 9999)} units and {rng.randrange(1, 90)} samples in room {rng.randrange(1, 20)}." for _ in range(10)]
neg += [f"The gateway is at 192.168.{rng.randrange(0, 255)}.{rng.randrange(1, 255)} on port {rng.randrange(1000, 9999)}." for _ in range(6)]
neg += [f"Invoice total ${rng.randrange(1, 9):d},{rng.randrange(100, 999)}.{rng.randrange(10, 99)} due in {rng.randrange(10, 60)} days." for _ in range(6)]
neg += [f"See section {rng.randrange(1, 9)}.{rng.randrange(1, 9)}.{rng.randrange(1, 9)} and table {rng.randrange(1, 20)} for limits." for _ in range(6)]
neg += ["I have one two three questions about the hold procedure.", "There were five lots and two samples in the cold room.",
        "meet at home.com tomorrow", "call me at noon", "send it at the end of the shift", "The CAPA must close within 30 days.",
        "Reviewer approval is required for PROT-SAF-7.", "Hash prefix a6c862e3 appears in the log.", "The dock door is number 4472.",
        "What is the status of lot 44?", "Quarantine LOT-00099 after the temperature alarm of 11.5 degrees.", "Open a deviation for LOT-00120.",
        "Chunk rag://chunks/a6c862e3bc23c59ce9082c9562a6c063b1dff719cb2b717e4529348357e0546a was cited.",
        "Order 5 pallets of SKU-042 and 12 of SKU-099.", "The audit seq is 1048 and the chain has 12 rows."]
neg += [f"Gate {rng.randrange(1, 18)} needs {rng.randrange(10, 99)}% coverage across {rng.randrange(100, 999)} tests." for _ in range(6)]
neg += [f"Hash {''.join(rng.choice('0123456789abcdef') for _ in range(12))} was logged for the build." for _ in range(8)]
neg += [f"Pack dimensions are {rng.randrange(10, 90)} x {rng.randrange(10, 90)} x {rng.randrange(10, 90)} cm, weight {rng.randrange(1, 40)}.{rng.randrange(0, 9)} kg." for _ in range(8)]
neg += [f"Buffer pH {rng.randrange(5, 9)}.{rng.randrange(0, 9)} at {rng.randrange(15, 30)}.{rng.randrange(0, 9)} C, lot LOT-{rng.randrange(0, 99999):05d}." for _ in range(8)]
neg += [f"Page {rng.randrange(1, 30)} of {rng.randrange(30, 90)}; table {rng.randrange(1, 12)}, row {rng.randrange(1, 60)}." for _ in range(8)]
neg += [f"Both LOT-{rng.randrange(0, 99999):05d} and LOT-{rng.randrange(0, 99999):05d} map to SKU-{rng.randrange(1, 999):03d}." for _ in range(10)]
neg += [f"Cycle count {rng.randrange(1000, 99999)} finished in {rng.randrange(10, 400)} seconds with {rng.randrange(0, 9)} faults." for _ in range(8)]
# Known-hard negatives, kept on purpose so the false-positive rate is honest:
neg += [f"PO number {rng.choice('456789')}{rng.randrange(10**8, 10**9)} was raised for the cold room compressor." for _ in range(6)]
for t in neg:
    add("none", "negative", t, [])

rng.shuffle(items)
for i, it in enumerate(items, 1):
    it["id"] = f"pii-{i:03d}"
OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
pos = sum(1 for i in items if i["pii"])
print(f"wrote {len(items)} items ({pos} positive, {len(items) - pos} negative) -> {OUT}")
