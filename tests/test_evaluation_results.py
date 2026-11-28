"""The published evaluation results belong to the model, thresholds and code in this repository.

Fails when the model, a threshold, a clause type's switch, or the evaluation code (scripts/evaluate_cuad.py, the
reader, the model reader or the clause types) changes without scripts/evaluate_cuad.py --write being run again.
"""
import hashlib
import json
from pathlib import Path

from key_terms_mcp.model import MODEL_PATH, MODEL_SHA256

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "docs" / "evaluation" / "results.json").read_text())


def test_results_are_for_the_released_model():
    assert RESULTS["model"]["sha256"] == MODEL_SHA256 == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def test_thresholds_and_switches_match_the_model(m):
    assert RESULTS["model"]["thresholds"] == {k: c["threshold"] for k, c in m.clauses.items()}
    assert RESULTS["model"]["use"] == {k: c["use"] for k, c in m.clauses.items()}
    assert sorted(k for k, c in m.clauses.items() if c["use"] == "off") == RESULTS["switched_off"]


def test_evaluation_code_unchanged_since_the_results():
    for path, sha in RESULTS["code_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, f"{path} changed: re-run the evaluation"


def test_results_are_from_the_full_recipe():
    assert RESULTS["folds"] == 5 and RESULTS["inner_folds"] == 4 and RESULTS["seed"] == 20261127
    assert RESULTS["recipe"]["vectorizer"]["max_features"] == 30000
    assert RESULTS["cuad"]["contracts"] == 510 and RESULTS["scored_contracts"] == 510
    assert RESULTS["cuad"]["sha256"] == "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
