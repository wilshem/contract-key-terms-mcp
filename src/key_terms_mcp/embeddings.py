"""The local sentence-embedding model for vector search: BAAI/bge-small-en-v1.5 (MIT licence), run with ONNX Runtime.

The model files are not in this repository. `contract-key-terms-mcp-setup` downloads them once, from the pinned
revision below, and checks each file's SHA-256. Every time the model loads, each file's SHA-256 is checked again
before it is used, the same way as the clause model's. The server never downloads anything.

Passages are embedded as they are; questions get the model's recommended search instruction in front. A text is
the [CLS] token's final hidden state, scaled to length 1, so a dot product is the cosine similarity.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

NAME = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
LICENCE = "MIT"
SOURCE = f"https://huggingface.co/{NAME}/tree/{REVISION}"
FILES = {  # path in the model repository -> SHA-256
    "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512
DIMENSIONS = 384
ENV_DIR = "KEY_TERMS_EMBED_DIR"


class EmbeddingModelMissing(RuntimeError):
    """The embedding model has not been set up."""


class EmbeddingIntegrityError(RuntimeError):
    """An embedding model file is not the pinned one."""


def model_dir(env: dict | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get(ENV_DIR):
        return Path(env[ENV_DIR]).expanduser()
    return Path("~/.cache/contract-key-terms-mcp").expanduser() / f"bge-small-en-v1.5@{REVISION[:12]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_files(folder: Path) -> dict[str, bytes]:
    """Each pinned file, read once and checked against its SHA-256 before anything uses it."""
    out = {}
    for rel, expected in FILES.items():
        p = folder / rel
        if not p.is_file():
            raise EmbeddingModelMissing(f"embedding model not set up: run contract-key-terms-mcp-setup ({rel} missing)")
        raw = p.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != expected:
            raise EmbeddingIntegrityError(f"{rel}: sha256 {got} does not match the pinned {expected}")
        out[rel] = raw
    return out


def onnxruntime_without_telemetry():
    """Import onnxruntime and switch its telemetry off before any InferenceSession is created.

    Uses the switch onnxruntime documents: onnxruntime.disable_telemetry_events(), "Disables platform-specific
    telemetry collection." (Python API docs, https://onnxruntime.ai/docs/api/python/api_summary.html, section
    "Telemetry events"; the same docstring is in onnxruntime 1.32.0). The docs do not say which platforms that
    covers, so they do not state that it covers Linux.
    """
    import onnxruntime as ort

    ort.disable_telemetry_events()
    return ort


class Embedder:
    """Needs the optional [search] extra (numpy, onnxruntime, tokenizers)."""

    def __init__(self, folder: Path | None = None, threads: int | None = None):
        import numpy as np
        from tokenizers import Tokenizer

        ort = onnxruntime_without_telemetry()  # before the InferenceSession below

        raw = check_files(folder or model_dir())
        self.np = np
        self.tok = Tokenizer.from_str(raw["tokenizer.json"].decode("utf-8"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False  # lower peak memory; results are the same
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(raw["onnx/model.onnx"], sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def _embed(self, texts: list[str], batch: int = 32):
        np = self.np
        out = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together: less padding
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok.encode_batch([texts[i] for i in idx])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                    "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64)}
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            cls = hidden[:, 0, :]
            out[idx] = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
        return out

    def passages(self, texts: list[str]):
        return self._embed(texts)

    def query(self, text: str):
        return self._embed([QUERY_INSTRUCTION + text])[0]

    def truncated(self, texts: list[str]) -> int:
        """How many of these texts are longer than the model reads (MAX_TOKENS tokens). Padding is off while
        counting: with it on, every text in a batch would be padded to the longest one and counted with it."""
        self.tok.no_truncation()
        self.tok.no_padding()
        try:
            return sum(len(e.ids) > MAX_TOKENS for e in self.tok.encode_batch(texts))
        finally:
            self.tok.enable_truncation(MAX_TOKENS)
            self.tok.enable_padding()
