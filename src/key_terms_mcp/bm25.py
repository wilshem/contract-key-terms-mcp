"""Keyword search: Okapi BM25 over chunks, in pure Python. Words are found the same way as the clause model's
(lower case, runs of two or more letters or digits). k1 and b are the usual values, fixed and not tuned."""
from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.2
B = 0.75
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[list[str]]):
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = [t for t in dict.fromkeys(query) if t in self.idf]
        out = []
        for tf, n in zip(self.tf, self.len, strict=True):
            s = 0.0
            norm = K1 * (1 - B + B * n / self.avg) if self.avg else K1
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + norm)
            out.append(s)
        return out
