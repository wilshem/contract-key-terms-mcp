"""Why the model scored a clause type as it did.

For a sentence, each word or word pair i has a normalised TF-IDF value x_i. For a clause type with weights w_i and
intercept b, its contribution is w_i * x_i, the logit is b + sum(w_i * x_i), and the probability is sigmoid(logit).
The top terms are the largest contributions; the clause's top terms are its largest weights over the vocabulary.
"""
from __future__ import annotations

from .clause_types import CLAUSE_TYPES
from .model import SentenceModel, model
from .reader import NOT_ADVICE, model_used, passage, scored_sentences

TOP_SENTENCES = 3
TOP_TERMS = 10
TOP_CLAUSE_TERMS = 15


def contributions(m: SentenceModel, clause: str, sentence: str) -> tuple[float, list[dict]]:
    x = m.features(sentence)
    w = m.clauses[clause]["weights"]
    rows = [{"term": m.terms[i], "tf_idf": round(v, 6), "weight": round(w.get(i, 0.0), 6),
             "contribution": w.get(i, 0.0) * v} for i, v in x.items()]
    return m.logit(clause, x), rows


def clause_top_terms(m: SentenceModel, clause: str, n: int = TOP_CLAUSE_TERMS) -> list[dict]:
    w = m.clauses[clause]["weights"]
    top = sorted(((v, i) for i, v in w.items() if v > 0), reverse=True)[:n]
    return [{"term": m.terms[i], "weight": round(v, 6)} for v, i in top]


def explain(text: str, clause: str, m: SentenceModel | None = None) -> dict:
    m = m or model()
    if clause not in CLAUSE_TYPES:
        raise ValueError(f"unknown clause type {clause!r}; call list_clause_types for the keys")
    scored = []
    for s, off in scored_sentences(text):
        logit, rows = contributions(m, clause, s)
        scored.append((m.sigmoid(logit), off, s, logit, rows))
    scored.sort(key=lambda r: (-r[0], r[1]))
    best = []
    for p, off, s, logit, rows in scored[:TOP_SENTENCES]:
        pos = sorted((r for r in rows if r["contribution"] > 0), key=lambda r: -r["contribution"])[:TOP_TERMS]
        neg = sorted((r for r in rows if r["contribution"] < 0), key=lambda r: r["contribution"])[:3]
        best.append({**passage(s, off, p), "probability": round(p, 4), "logit": round(logit, 6),
                     "top_terms": [{**r, "contribution": round(r["contribution"], 6)} for r in pos],
                     "top_negative_terms": [{**r, "contribution": round(r["contribution"], 6)} for r in neg]})
    c = m.clauses[clause]
    return {"clause_type": clause, "label": CLAUSE_TYPES[clause][0], "cuad_category": CLAUSE_TYPES[clause][1],
            "threshold": c["threshold"], "intercept": round(c["intercept"], 6), "model_used": model_used(m, clause),
            "best_sentences": best, "clause_top_terms": clause_top_terms(m, clause), "not_advice": NOT_ADVICE}
