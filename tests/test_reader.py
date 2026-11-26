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


class StubModel:
    """A model with chosen scores: every type used, threshold 0.5; score(sentence) gives each sentence's score."""

    def __init__(self, score):
        self.score = score
        self.clauses = {k: {"use": "either", "threshold": 0.5} for k in CLAUSE_TYPES}

    def probabilities(self, sentence):
        return dict.fromkeys(self.clauses, self.score(sentence))


def test_sentence_exactly_at_threshold_is_reported():
    text = "The Supplier shall deliver the goods on time."
    out = reader.read(text, StubModel(lambda s: 0.5))
    assert out["anti_assignment"]["found"]
    assert [p["text"] for p in out["anti_assignment"]["passages"]] == [text]


def test_at_most_two_passages_when_more_pass():
    text = ("The Supplier shall deliver the goods. The Buyer shall pay the price. "
            "Either party may end this Agreement on notice.")
    out = reader.read(text, StubModel(lambda s: 0.9))
    for v in out.values():
        assert len(v["passages"]) == reader.MAX_PASSAGES == 2


def test_heading_never_quoted_by_itself():
    text = "12. Assignment.\nNeither party may assign this Agreement without consent."
    out = reader.read(text, StubModel(lambda s: 0.99))
    quoted = {p["text"] for v in out.values() for p in v["passages"]}
    assert next(s for s, _ in reader.sentences(text)) == "Assignment."  # the heading as the splitter sees it
    assert "Assignment." not in quoted and "12. Assignment." not in quoted
    assert "Neither party may assign this Agreement without consent." in quoted
