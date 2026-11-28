"""search_clauses: chunking, keyword and vector search through the guard, refusals, limits, and no network."""
import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from key_terms_mcp import chunking
from key_terms_mcp.bm25 import BM25, tokens
from key_terms_mcp.paths import Guard
from key_terms_mcp.search import NO_EXTRA, FolderIndex
from key_terms_mcp.server import SAMPLES, build


class FakeEmbedder:
    """Deterministic: each word adds a fixed pseudo-random vector (no model, no numpy needed beyond the search)."""

    dims = 32

    def _vec(self, text):
        np = pytest.importorskip("numpy")
        v = np.zeros(self.dims)
        for w in tokens(text):
            h = hashlib.sha256(w.encode()).digest()
            v += np.frombuffer(h, dtype=np.uint8)[: self.dims] / 255 - 0.5
        n = np.linalg.norm(v)
        return v / n if n else v

    def passages(self, texts):
        np = pytest.importorskip("numpy")
        return np.array([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def payload(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc)
    return json.loads(res.content[0].text)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network use attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_chunks_cover_text_in_order_with_offsets():
    text = (SAMPLES / "services_agreement.txt").read_text()
    cs = chunking.chunks(text)
    assert cs and all(len(c.text.split()) <= chunking.CHUNK_WORDS for c in cs)
    assert [c.start for c in cs] == sorted(c.start for c in cs)
    for c in cs:
        src = iter("".join(text[c.start:c.end].split()))
        assert all(ch in src for ch in "".join(c.text.split()))  # its characters, in order, from its own span


def _over_limit_ok(text, words):
    """No chunk exceeds the limit except by one carried heading (its first piece)."""
    for c in chunking.chunks(text, words):
        n = len(c.text.split())
        if n > words:
            first = next(p for p in chunking._pieces(text, words) if p[0] == c.start)
            assert first[3] and n - first[2] <= words, (n, c.text[:60])


def test_dot_leaders_left_out_and_limit_holds():
    toc = "".join(f"{i}. Section {i} . . . . . . . . . . . . . . . . . . . . {i}\n" for i in range(1, 30))
    body = "".join(f"{i}. Heading {i}.\n" + "The Supplier shall deliver the goods on time. " * 9 + "\n"
                   for i in range(1, 12))
    text = toc + body
    for words in (16, 64, 128):
        _over_limit_ok(text, words)
        assert all(". . ." not in c.text for c in chunking.chunks(text, words))


def test_limit_on_samples():
    for f in SAMPLES.glob("*.txt"):
        for words in (32, 64, 128, 256):
            _over_limit_ok(f.read_text(), words)


def test_long_sentence_cut_into_windows():
    text = " ".join(f"word{i}" for i in range(300)) + "."
    cs = chunking.chunks(text, words=128)
    assert [len(c.text.split()) for c in cs] == [128, 128, 44]


def test_heading_moves_to_next_chunk():
    body = "The Supplier shall deliver the goods on time and in full every single month. "
    text = body * 6 + "\n12. Assignment.\n" + "Neither party may assign this Agreement without consent of the other party.\n"
    cs = chunking.chunks(text, words=90)
    assert any(c.text.startswith("Assignment. Neither party") for c in cs)  # "12." splits off as its own piece


def test_bm25_prefers_matching_chunk():
    s = BM25([tokens("governed by the laws of Delaware"), tokens("payment within thirty days")])
    a, b = s.scores(tokens("which laws govern"))
    assert a > b == 0.0


def test_keyword_search_on_samples(no_network):
    out = FolderIndex(Guard.create(SAMPLES)).search("governing law of the agreement", 3, "keyword")
    assert out["files_searched"] == 4 and out["files_skipped"] == 0  # 3 contracts and SOURCE.md and len(out["results"]) == 3
    top = out["results"][0]
    text = (SAMPLES / top["file"]).read_text()
    assert " ".join(text[top["offset"]:].split()).startswith(top["text"][:40])


def test_vector_search_with_fake_embedder(no_network):
    pytest.importorskip("numpy")
    out = FolderIndex(Guard.create(SAMPLES), FakeEmbedder).search("assign this agreement consent", 2, "vector")
    assert len(out["results"]) == 2 and out["method"] == "vector"
    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)


def test_vector_without_model_is_a_clear_error():
    with pytest.raises(RuntimeError) as e:
        FolderIndex(Guard.create(SAMPLES)).search("anything", 1, "vector")
    assert str(e.value) == NO_EXTRA


@pytest.mark.parametrize("q,k,m", [("", 5, "keyword"), ("x" * 1001, 5, "keyword"), ("q", 0, "keyword"),
                                   ("q", 21, "keyword"), ("q", 5, "magic")])
def test_bad_arguments_refused(q, k, m):
    with pytest.raises(ValueError):
        FolderIndex(Guard.create(SAMPLES)).search(q, k, m)


def test_refused_files_are_skipped_and_never_named(tmp_path):
    allowed = tmp_path / "allowed"
    (allowed / "secret").mkdir(parents=True)
    (allowed / "ok.txt").write_text("This Agreement shall be governed by the laws of Ohio.")
    (allowed / "secret" / "hidden.txt").write_text("This Agreement shall be governed by the laws of Texas.")
    outside = tmp_path / "outside.txt"
    outside.write_text("This Agreement shall be governed by the laws of Utah.")
    os.symlink(outside, allowed / "link.txt")
    os.link(allowed / "ok.txt", tmp_path / "second_name.txt")  # ok.txt now has two hard links
    (allowed / "plain.md").write_text("Notices go by email. The laws of Nevada govern.")
    g = Guard.create(allowed, deny_dirs=[str(allowed / "secret")], deny_names=[], home=tmp_path, env={})
    out = FolderIndex(g).search("which laws govern", 10, "keyword")
    assert out["files_searched"] == 1 and out["files_skipped"] == 3  # denied folder, outside link, hard link
    blob = json.dumps(out)
    for word in ("Texas", "Utah", "Ohio", "secret", "hidden", "outside", "link.txt", "ok.txt"):
        assert word not in blob
    assert {r["file"] for r in out["results"]} == {"plain.md"}


def test_index_rebuilds_a_changed_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("The laws of Ohio govern this Agreement.")
    idx = FolderIndex(Guard.create(tmp_path, home=tmp_path, env={}))
    assert "Ohio" in idx.search("laws", 1, "keyword")["results"][0]["text"]
    f.write_text("The laws of Nevada govern this Agreement, and nothing else is said here.")
    assert "Nevada" in idx.search("laws", 1, "keyword")["results"][0]["text"]


async def test_search_tool_over_mcp_without_network(no_network):
    server = build(Guard.create(SAMPLES))
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        assert names == ["explain_prediction", "extract_key_terms", "list_clause_types", "search_clauses"]
        out = payload(await client.call_tool("search_clauses", {"question": "termination on notice", "k": 2,
                                                                "method": "keyword"}))
        assert len(out["results"]) == 2 and out["results"][0]["file"].endswith(".txt")


def test_server_import_loads_no_search_dependencies():
    """The base install: importing and building the server loads none of numpy, onnxruntime or tokenizers."""
    import subprocess
    import sys
    code = ("import sys; from key_terms_mcp.server import build, SAMPLES; from key_terms_mcp.paths import Guard; "
            "build(Guard.create(SAMPLES)); "
            "print(sorted(m for m in ('numpy', 'onnxruntime', 'tokenizers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


@pytest.fixture
def no_search_extra(monkeypatch):
    """As if the [search] extra were not installed."""
    import sys
    for name in ("numpy", "onnxruntime", "tokenizers"):
        monkeypatch.setitem(sys.modules, name, None)


async def test_without_extra_three_tools_and_keyword_work(no_search_extra, no_network):
    server = build(Guard.create(SAMPLES))
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        types = payload(await client.call_tool("list_clause_types", {}))
        assert len(types) == 36
        out = payload(await client.call_tool("extract_key_terms", {"path": "services_agreement.txt"}))
        assert out["clauses"]["governing_law"]["found"]
        ex = payload(await client.call_tool("explain_prediction", {"path": "supply_agreement.txt",
                                                                    "clause_type": "governing_law"}))
        assert ex["best_sentences"]
        kw = payload(await client.call_tool("search_clauses", {"question": "governing law", "k": 2,
                                                               "method": "keyword"}))
        assert len(kw["results"]) == 2
        default = payload(await client.call_tool("search_clauses", {"question": "governing law", "k": 2}))
        assert default["method"] == "keyword" and default["results"] == kw["results"]
        vec = await client.call_tool("search_clauses", {"question": "governing law", "k": 2, "method": "vector"})
        assert vec.isError
        assert NO_EXTRA in vec.content[0].text
        assert NO_EXTRA == ("vector search needs the optional [search] extra: run uv sync --extra search, then uv run "
                            "contract-key-terms-mcp-setup; keyword search (method=\"keyword\") works without it")


def test_truncated_count_does_not_depend_on_batch():
    """With the real tokenizer (skipped when the model is not set up): a long text must not make short ones count."""
    pytest.importorskip("tokenizers")
    from key_terms_mcp import embeddings
    try:
        e = embeddings.Embedder()
    except (RuntimeError, OSError, ImportError):
        pytest.skip("embedding model not set up")
    long_text = "word " * 2000
    assert e.truncated(["short text", "another short one"]) == 0
    assert e.truncated(["short text", long_text, "another short one"]) == 1
    assert e.truncated(["short text"]) == 0  # padding and truncation restored afterwards


def test_onnxruntime_telemetry_switched_off_before_any_session(monkeypatch):
    """Embedder calls onnxruntime.disable_telemetry_events() before it creates its InferenceSession."""
    import sys
    import types

    from key_terms_mcp import embeddings
    calls = []
    ort = types.ModuleType("onnxruntime")
    ort.disable_telemetry_events = lambda: calls.append("disable_telemetry_events")

    class Opts:
        enable_cpu_mem_arena = True
        intra_op_num_threads = 0

    class Session:
        def __init__(self, *a, **k):
            calls.append("InferenceSession")

        def get_inputs(self):
            return []

    ort.SessionOptions, ort.InferenceSession = Opts, Session
    tok = types.ModuleType("tokenizers")

    class Tok:
        @classmethod
        def from_str(cls, s):
            return cls()

        def enable_truncation(self, n):
            pass

        def enable_padding(self):
            pass

    tok.Tokenizer = Tok
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)
    pytest.importorskip("numpy")
    monkeypatch.setattr(embeddings, "check_files", lambda folder: {"tokenizer.json": b"{}", "onnx/model.onnx": b""})
    embeddings.Embedder()
    assert calls == ["disable_telemetry_events", "InferenceSession"]


def test_evaluate_retrieval_switches_telemetry_off_before_the_embedder():
    src = (Path(__file__).resolve().parent.parent / "scripts" / "evaluate_retrieval.py").read_text()
    assert src.index("embeddings.onnxruntime_without_telemetry()") < src.index("embeddings.Embedder()")
    assert "ORT_DISABLE_TELEMETRY" not in src
