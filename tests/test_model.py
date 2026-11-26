import gzip
import json
import math
from pathlib import Path

import pytest

from key_terms_mcp.model import MODEL_PATH, MODEL_SHA256, ModelIntegrityError, SentenceModel, check_sha256

GOLDEN = json.loads((Path(__file__).parent / "golden_probabilities.json").read_text())


def test_sha256_matches_release():
    check_sha256(MODEL_PATH.read_bytes())


def test_shape(m):
    assert len(m.vocabulary) == 30000 and len(m.idf) == 30000 and len(m.clauses) == 36
    assert sum(c["use"] == "either" for c in m.clauses.values()) == 33


def test_probabilities_match_original_implementation(m):
    for row in GOLDEN["sentences"]:
        got = m.probabilities(row["text"])
        assert got.keys() == row["probabilities"].keys()
        for k, v in row["probabilities"].items():
            assert math.isclose(got[k], v, rel_tol=0, abs_tol=1e-12), (row["text"], k)


def test_tampered_model_refused(tmp_path):
    raw = gzip.decompress(MODEL_PATH.read_bytes())
    data = json.loads(raw)
    data["clauses"]["anti_assignment"]["intercept"] += 1.0
    bad = tmp_path / "m.json.gz"
    bad.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ModelIntegrityError):
        SentenceModel.load(bad)
    assert MODEL_SHA256 != ""


def test_terms_inverts_vocabulary(m):
    t = m.terms
    for term, i in list(m.vocabulary.items())[:500]:
        assert t[i] == term
