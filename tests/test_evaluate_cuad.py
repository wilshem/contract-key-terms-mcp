"""The evaluation script's pieces, and a small offline run on a made-up CUAD-format fixture (no download)."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "cuad_fixture.json"
_spec = importlib.util.spec_from_file_location("evaluate_cuad", ROOT / "scripts" / "evaluate_cuad.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_cuad"] = ev
_spec.loader.exec_module(ev)


def test_fixed_settings():
    assert ev.SEED == 20261127 and ev.FOLDS == 5 and ev.INNER_FOLDS == 4
    assert ev.SWITCH_OFF_F1 == 0.50
    assert ev.SWITCH_OFF_RULE == (
        "A clause type is switched off (the model is not used for it, and extract_key_terms does not report it) "
        "when its cross-validated document-level F1, pooled over the five held-out folds, is below 0.50. "
        "Switched-off types are still scored and listed separately in the results, outside the averages.")
    assert ev.RECIPE["vectorizer"] == {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True, "lowercase": True,
                                       "max_features": 30000}
    assert ev.RECIPE["classifier"]["C"] == 1.0 and ev.RECIPE["classifier"]["class_weight"] == "balanced"


def test_label_overlap_rules():
    text = "Intro line here is long enough.\nThe laws of Ohio shall govern this Agreement.\nEnd of it all here.\n"
    a = text.index("laws of Ohio")
    c = {"title": "t", "text": text, "spans": {k: [] for k in ev.KEYS}}
    c["spans"]["governing_law"] = [(a, a + len("laws of Ohio"))]
    rows = ev.label(c)
    hit = [r for r in rows if "governing_law" in r["pos"]]
    assert len(hit) == 1 and "Ohio" in hit[0]["text"]
    assert "governing_law" not in hit[0]["strict"]  # the span covers less than half the sentence
    c["spans"]["governing_law"] = [(text.index("The laws"), text.index("Agreement.") + 10)]
    assert any("governing_law" in r["strict"] for r in ev.label(c))


def test_whitespace_only_overlap_does_not_count():
    text = "First sentence is right here.\n\nSecond sentence is right here.\n"
    gap = text.index("\n\n")
    assert not ev._touches(text, (0, gap + 2), (gap, gap + 2))


def test_best_threshold_ties_go_high():
    assert ev.best_threshold([0.9, 0.8, 0.1], [True, True, False]) == 0.8
    assert ev.best_threshold([0.0, 0.0], [False, False]) == 0.99


def test_groups_join_near_duplicates_only():
    base = " ".join(f"word{i}" for i in range(400))
    near = base.replace("word7 ", "changed ")
    other = " ".join(f"other{i}" for i in range(400))
    g = ev.groups([base, near, other])
    assert g[0] == g[1] != g[2]


def test_folds_keep_groups_together_and_are_seeded():
    group_of = [0, 0, 1, 2, 3, 3, 3, 4, 5, 6]
    f1 = ev.assign_folds(group_of, 3, 1)
    assert f1 == ev.assign_folds(group_of, 3, 1)
    for g in set(group_of):
        assert len({f1[i] for i, x in enumerate(group_of) if x == g}) == 1


def test_fit_rule_text():
    assert "0.10" in ev.FIT_RULE and "0.90" in ev.FIT_RULE


def test_prf():
    assert ev.prf(0, 0, 0)["f1"] == 0.0
    assert ev.prf(1, 1, 0) == {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "tp": 1, "fp": 1, "fn": 0}


def test_model_bytes_reproducible():
    m = {"a": 1}
    assert ev.model_bytes(m) == ev.model_bytes(m)


def _args(**kw):
    import argparse
    d = {"folds": 2, "inner_folds": 2, "seed": 1, "max_features": 200, "only_fold": None, "write": False}
    d.update(kw)
    return argparse.Namespace(**d)


def test_fixture_run_end_to_end():
    pytest.importorskip("sklearn")
    cuad = json.loads(FIXTURE.read_text())
    r = ev.run(cuad, "fixture", _args(write=True))
    assert r["scored_contracts"] == 16
    gl = r["per_type"]["governing_law"]["document"]
    assert gl["tp"] + gl["fn"] == 12
    assert set(r["switched_off"]) >= {"change_of_control"}  # nothing to learn from in the fixture
    assert r["model"]["use"]["governing_law"] in ("model", "off")
    assert r["types_on"].keys() == r["all_36"].keys()
    assert r["all_36_intervals"]["document"]["f1"][0] <= r["all_36"]["document"]["micro"]["f1"]
    b = r["baseline_best_term"]
    assert len(b["per_type"]["governing_law"]["terms_by_fold"]) == 2
    assert b["all_36"]["document"]["micro"]["f1"] > 0
    assert "Baseline on the same folds" in ev.markdown(r)
    dg = r["diagnostic"]
    assert dg["verdict"] in ("overfits", "underfits", "neither")
    assert len(dg["weakest_5"]) == 5 and len(dg["strongest_5"]) == 5
    assert dg["sentence_ceiling"]["sentence_f1_ceiling_main_rule"] == 1.0
    assert 0 < dg["sentence_ceiling"]["char_precision"] <= 1 and dg["sentence_ceiling"]["span_recall_ceiling"] == 1.0


def test_sentence_model_matches_numpy_path():
    pytest.importorskip("sklearn")
    cuad = json.loads(FIXTURE.read_text())
    docs = ev.contracts(cuad)
    labelled = [ev.label(d) for d in docs]
    model = ev.train(labelled, 1, 200)
    texts = [r["text"] for d in labelled[:4] for r in d]
    fast = ev.numpy_probabilities(model, texts)
    slow = ev.sentence_model_probabilities(model, texts)
    for i, row in enumerate(slow):
        for j, k in enumerate(ev.KEYS):
            assert math.isclose(row[k], float(fast[i, j]), abs_tol=1e-9)
