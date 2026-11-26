import math

import pytest

from key_terms_mcp import explain

S = "Neither party may assign this Agreement without the prior written consent of the other party."


def test_contributions_sum_to_logit(m):
    logit, rows = explain.contributions(m, "anti_assignment", S)
    c = m.clauses["anti_assignment"]
    assert math.isclose(logit - c["intercept"], sum(r["contribution"] for r in rows), abs_tol=1e-9)
    assert math.isclose(m.sigmoid(logit), m.probabilities(S)["anti_assignment"], abs_tol=1e-12)


def test_explain_shape_and_order(m):
    out = explain.explain(S + " The Supplier shall deliver goods.", "anti_assignment", m)
    assert out["clause_type"] == "anti_assignment" and out["model_used"] is True
    best = out["best_sentences"]
    assert 1 <= len(best) <= explain.TOP_SENTENCES
    assert [b["probability"] for b in best] == sorted((b["probability"] for b in best), reverse=True)
    top = best[0]["top_terms"]
    assert top and all(t["contribution"] > 0 for t in top)
    assert [t["contribution"] for t in top] == sorted((t["contribution"] for t in top), reverse=True)
    assert all(t["contribution"] < 0 for t in best[0]["top_negative_terms"])
    assert "not legal advice" in out["not_advice"]


def test_clause_top_terms_are_largest_weights(m):
    top = explain.clause_top_terms(m, "governing_law", 15)
    w = sorted((v for v in m.clauses["governing_law"]["weights"].values() if v > 0), reverse=True)[:15]
    assert [t["weight"] for t in top] == [round(v, 6) for v in w]


def test_unknown_clause(m):
    with pytest.raises(ValueError):
        explain.explain(S, "no_such_type", m)
