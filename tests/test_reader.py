from conftest import SAMPLES

from key_terms_mcp import reader
from key_terms_mcp.clause_types import CLAUSE_TYPES


def test_sentences_split_and_offsets():
    text = "First term here. Second term; third one\nstill third.\n\nFourth 1.5 days."
    got = reader.sentences(text)
    assert [s for s, _ in got] == ["First term here.", "Second term;", "third one still third.", "Fourth 1.5 days."]
    for s, off in got:
        assert text[off:].lstrip().startswith(s.split()[0])


def test_heading_only():
    assert reader.heading_only("12. Non-Compete.")
    assert not reader.heading_only("12. The Supplier shall not compete.")


def test_read_keys_and_limits(m):
    out = reader.read((SAMPLES / "services_agreement.txt").read_text(), m)
    used = {k for k in CLAUSE_TYPES if m.clauses[k]["use"] == "either"}
    assert set(out) == used and len(out) == 33
    for k, v in out.items():
        assert v["label"] == CLAUSE_TYPES[k][0]
        assert len(v["passages"]) <= reader.MAX_PASSAGES and v["found"] == bool(v["passages"])
        scores = [p["score"] for p in v["passages"]]
        assert scores == sorted(scores, reverse=True)
        for p in v["passages"]:
            assert p["score"] >= round(m.clauses[k]["threshold"], 4) - 1e-4
            assert len(p["text"]) <= reader.MAX_PASSAGE_CHARS + 3


def test_rules_only_types_not_reported(m):
    out = reader.read("Neither party may assign this Agreement without consent.", m)
    for k, c in m.clauses.items():
        if c["use"] != "either":
            assert k not in out


def test_long_sentence_cut(m):
    s = "Neither party may assign this Agreement without the prior written consent of the other party " + "x" * 900
    p = reader.passage(s, 0, 0.9)
    assert p["text"].endswith("...") and len(p["text"]) == reader.MAX_PASSAGE_CHARS + 3


def test_finds_assignment(m):
    out = reader.read("Neither party may assign this Agreement without the prior written consent of the other party.", m)
    assert out["anti_assignment"]["found"]


def test_empty_text(m):
    assert not any(v["found"] for v in reader.read("", m).values())
