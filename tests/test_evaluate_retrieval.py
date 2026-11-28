"""The retrieval evaluation's truncation count, on the --choose and --write path (evaluate()), uses the fixed
Embedder.truncated: padding off while counting, so one long chunk does not make the others count."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "cuad_fixture.json"
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_counts_truncation_per_chunk():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")
    from key_terms_mcp import embeddings
    try:
        real = embeddings.Embedder()
    except (RuntimeError, OSError, ImportError):
        pytest.skip("embedding model not set up")
    er, ev = _load("evaluate_retrieval"), _load("evaluate_cuad")
    cuad = json.loads(FIXTURE.read_text())
    docs = ev.contracts(cuad)
    long_doc = dict(docs[0], text=docs[0]["text"] + "\n" + ("a-b-c " * 2000) + ".")
    docs = [long_doc, *docs[1:]]

    class CountingEmbedder:
        tok = real.tok
        truncated = embeddings.Embedder.truncated

        def passages(self, texts):
            return np.ones((len(texts), 4)) / 2

        def query(self, text):
            return np.ones(4) / 2

    idx = list(range(len(docs)))
    r = er.evaluate(docs, idx, er.questions(cuad), CountingEmbedder(), 256, pool=False)
    texts = [c.text for i in idx for c in er.chunks(docs[i]["text"], 256)]
    one_by_one = sum(real.truncated([t]) for t in texts)
    assert r["truncated_chunks"] == one_by_one < len(texts)
    assert one_by_one >= 1  # 256-word pieces of "a-b-c" are about 1,280 tokens each
