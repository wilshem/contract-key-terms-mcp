"""The published retrieval results belong to the code and chunk size in this repository.

Fails when the retrieval evaluation code (scripts/evaluate_retrieval.py and the files it lists in CODE_FILES) or
CHUNK_WORDS changes without scripts/evaluate_retrieval.py --write being run again.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from key_terms_mcp.chunking import CHUNK_WORDS

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "docs" / "evaluation"
RESULTS = json.loads((EVAL / "retrieval_results.json").read_text())
CHOICE = json.loads((EVAL / "retrieval_chunk_choice.json").read_text())
CODE_FILES = ("scripts/evaluate_retrieval.py", "scripts/evaluate_cuad.py", "src/key_terms_mcp/chunking.py",
              "src/key_terms_mcp/bm25.py", "src/key_terms_mcp/embeddings.py", "src/key_terms_mcp/reader.py")


def stale(results: dict, choice: dict) -> list[str]:
    """Why these results no longer match the tree (empty when they do)."""
    why = []
    if set(results["code_sha256"]) != set(CODE_FILES):
        why.append("the recorded code files differ from the list the results must cover")
    for path, sha in results["code_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            why.append(f"{path} changed")
    if results["chunk_words"] != CHUNK_WORDS:
        why.append(f"results are for {results['chunk_words']}-word chunks, CHUNK_WORDS is {CHUNK_WORDS}")
    if choice["chosen"] != CHUNK_WORDS:
        why.append(f"the dev-set choice was {choice['chosen']}, CHUNK_WORDS is {CHUNK_WORDS}")
    return why


def test_retrieval_results_match_the_tree():
    assert stale(RESULTS, CHOICE) == [], "re-run scripts/evaluate_retrieval.py --write"


def test_code_list_matches_the_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", ROOT / "scripts" / "evaluate_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert tuple(mod.CODE_FILES) == CODE_FILES


@pytest.mark.parametrize("break_it", ["code", "chunk_words", "choice", "missing_file"])
def test_check_fails_on_a_stale_record(break_it):
    """The switch-off: each kind of stale record is caught."""
    r, c = copy.deepcopy(RESULTS), copy.deepcopy(CHOICE)
    if break_it == "code":
        r["code_sha256"]["src/key_terms_mcp/chunking.py"] = "0" * 64
    elif break_it == "chunk_words":
        r["chunk_words"] = CHUNK_WORDS // 2
    elif break_it == "choice":
        c["chosen"] = CHUNK_WORDS // 2
    else:
        del r["code_sha256"]["src/key_terms_mcp/bm25.py"]
    assert stale(r, c)
