"""Read one agreement's text with the sentence model: for each CUAD clause type, the sentences the model scores at or
above that type's threshold, best first, quoted as written with their character offset.

Facts only: a passage is a sentence of the agreement, quoted as written. The reader never says what a clause means,
whether it is enforceable, or what follows when a clause is not found.
"""
from __future__ import annotations

import re

from .clause_types import CLAUSE_TYPES
from .model import SentenceModel, model

MAX_PASSAGES = 2
MAX_PASSAGE_CHARS = 600
NOT_ADVICE = ("Each entry quotes what an agreement says and where; it is not legal advice and does not say what a "
              "clause means, whether it is enforceable, or what follows when a clause is not found.")

_VERB = re.compile(r"\b(?:shall|may|will|must|is|are|be|has|have|agrees?|grants?|means)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.;\n](?:[^.;]|\.(?=\d)|\n(?!\s*\n))*[.;]?")


def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """(sentence with its white space collapsed, start, end) for each sentence; text[start:end] is the sentence as
    written."""
    out = []
    for m in _SENTENCE.finditer(text):
        s = " ".join(m.group(0).split())
        if len(s) > 3:
            out.append((s, m.start(), m.end()))
    return out


def sentences(text: str) -> list[tuple[str, int]]:
    """(sentence with its white space collapsed, character offset) for each sentence of the text."""
    return [(s, start) for s, start, _ in sentence_spans(text)]


def heading_only(sentence: str) -> bool:
    """A heading on its own line ("12. Non-Compete."): never quoted by itself."""
    words = re.sub(r"^\s*(?:section\s+|article\s+)?[\dIVXLC]+(?:\.\d+)*\.?\s+", "", sentence, flags=re.IGNORECASE).split()
    return len(words) <= 6 and not _VERB.search(sentence)


def model_used(m: SentenceModel, clause: str) -> bool:
    """Whether the model is used for this type ("model"), or the type is switched off ("off") by the evaluation's
    switch-off rule (docs/evaluation/results.md)."""
    return m.clauses[clause]["use"] == "model"


def passage(s: str, off: int, p: float) -> dict:
    cut = s[:MAX_PASSAGE_CHARS] + ("..." if len(s) > MAX_PASSAGE_CHARS else "")
    return {"text": cut, "offset": off, "score": round(p, 4)}


def scored_sentences(text: str) -> list[tuple[str, int]]:
    return [(s, off) for s, off in sentences(text) if not heading_only(s)]


def read(text: str, m: SentenceModel | None = None) -> dict[str, dict]:
    """{clause key: {"label", "found", "passages": [{"text", "offset", "score"}]}} for the types the model is used for."""
    m = m or model()
    keys = [k for k in CLAUSE_TYPES if model_used(m, k)]
    hits: dict[str, list[tuple[float, int, str]]] = {k: [] for k in keys}
    for s, off in scored_sentences(text):
        probs = m.probabilities(s)
        for k in keys:
            if probs[k] >= m.clauses[k]["threshold"]:
                hits[k].append((probs[k], off, s))
    out = {}
    for k in keys:
        best = sorted(hits[k], key=lambda h: (-h[0], h[1]))[:MAX_PASSAGES]
        out[k] = {"label": CLAUSE_TYPES[k][0], "found": bool(best),
                  "passages": [passage(s, off, p) for p, off, s in best]}
    return out
