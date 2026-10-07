"""gates.py — the mechanical content gates, shared by the stages and their tests.

The leak check runs in stages 00 and 02 and is asserted by tests/test_leak_check.py, so it
lives here rather than inside a numbered stage script.

leak_hit   finds an entity name, synonym or banned term in a question's text.
band_for   maps the count of answering documents to a hardness band."""
from __future__ import annotations

import re


def norm_ws(s: str) -> str:
    """Collapse whitespace, strip and lowercase."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def leak_hit(texts, entity, synonyms, banned, *, check_entity: bool = True) -> str | None:
    """Return the first leaked term found in `texts`, or None.

    The terms are the entity and its synonyms (when `check_entity`) plus `banned`; terms
    shorter than 3 characters are ignored. Matching is on word boundaries. All-caps terms
    of up to 8 characters (trial acronyms like REDUCE, sign abbreviations like TEN) match
    case-sensitively, so a common word does not count as a leak."""
    terms = list(banned or [])
    if check_entity:
        terms = [entity] + list(synonyms or []) + terms
    hay = " ".join(t for t in texts if t)
    hay_l = hay.lower()
    for t in terms:
        t = (t or "").strip()
        if len(t) < 3:
            continue
        pat = r"\b" + re.escape(t) + r"\b"
        if t.isupper() and len(t) <= 8:
            if re.search(pat, hay):
                return t
        elif re.search(pat, hay_l if t == t.lower() else hay, re.I):
            return t
    return None


def band_for(n_answering: int, bands: dict | None = None) -> str:
    """Hardness band from the number of top-10 documents scoring >= the answering
    threshold. The cut points come from config (`hardness.bands`); the defaults below
    apply when none are given."""

    bands = bands or {"ultra_hard": [0, 0], "hard": [1, 3], "medium": [4, 6],
                      "easy": [7, 999]}
    for name, (lo, hi) in bands.items():
        if lo <= n_answering <= hi:
            return name
    raise ValueError(f"n_answering={n_answering} falls in no band of {bands}")
