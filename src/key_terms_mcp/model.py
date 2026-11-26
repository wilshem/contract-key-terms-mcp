"""The CUAD-trained sentence model, read with the Python standard library alone.

Word and word-pair TF-IDF features, computed the same way as scikit-learn's TfidfVectorizer with the settings in
VECTORIZER, then one logistic regression per CUAD clause type. The weights were trained on CUAD v1 (The Atticus
Project, CC BY 4.0); see NOTICE.md.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

MODEL_PATH = Path(__file__).parent / "models" / "key_terms_model.json.gz"
MODEL_SHA256 = "216c8391a754ef46627ff9b7860f3e34b80951890744e564c80a7984d80ef13b"


class ModelIntegrityError(RuntimeError):
    """The model file is not the one this package was released with."""


def check_sha256(raw: bytes, expected: str = MODEL_SHA256) -> None:
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise ModelIntegrityError(f"model file sha256 {got} does not match the released {expected}")


class SentenceModel:
    VECTORIZER: ClassVar[dict] = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True, "lowercase": True}
    _TOKEN = re.compile(r"(?u)\b\w\w+\b")

    def __init__(self, data: dict):
        self.vocabulary: dict[str, int] = data["vocabulary"]
        self.idf: list[float] = data["idf"]
        self.clauses = {k: {**c, "weights": {int(i): w for i, w in c["weights"].items()}}
                        for k, c in data["clauses"].items()}
        self._terms: list[str] | None = None

    @classmethod
    def load(cls, path: Path = MODEL_PATH, expected_sha256: str = MODEL_SHA256) -> SentenceModel:
        raw = path.read_bytes()
        check_sha256(raw, expected_sha256)
        return cls(json.loads(gzip.decompress(raw).decode("utf-8")))

    @property
    def terms(self) -> list[str]:
        """Feature index -> word or word pair (the vocabulary inverted once, on first use)."""
        if self._terms is None:
            terms = [""] * len(self.idf)
            for t, i in self.vocabulary.items():
                terms[i] = t
            self._terms = terms
        return self._terms

    def features(self, sentence: str) -> dict[int, float]:
        words = self._TOKEN.findall(sentence.lower())
        counts = Counter(words + [f"{a} {b}" for a, b in pairwise(words)])
        x = {self.vocabulary[t]: (1 + math.log(n)) * self.idf[self.vocabulary[t]]
             for t, n in counts.items() if t in self.vocabulary}
        norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
        return {i: v / norm for i, v in x.items()}

    @staticmethod
    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-max(min(z, 50), -50)))

    def logit(self, clause: str, x: dict[int, float]) -> float:
        c = self.clauses[clause]
        return c["intercept"] + sum(v * c["weights"].get(i, 0.0) for i, v in x.items())

    def probabilities(self, sentence: str) -> dict[str, float]:
        x = self.features(sentence)
        return {k: self.sigmoid(self.logit(k, x)) for k in self.clauses}


_MODEL: list[SentenceModel] = []  # loaded once, on first use


def model() -> SentenceModel:
    if not _MODEL:
        _MODEL.append(SentenceModel.load())
    return _MODEL[0]
