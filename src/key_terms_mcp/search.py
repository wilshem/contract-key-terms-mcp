"""search_clauses: the passages in the allowed folder that best match a question, by keyword search (BM25).

Every file goes through the same guard as the other tools (resolved first, then the inside, deny, suffix, size,
link and hard-link checks); a refused file is skipped and counted, never named. The folder is walked without
following links. Chunks are kept in memory only, for the server's life, and are rebuilt for a file when its size,
time stamp or inode changes. Nothing is written to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25, tokens
from .chunking import Chunk, chunks
from .paths import SUFFIXES, Guard, PathRefused
from .reader import MAX_PASSAGE_CHARS, NOT_ADVICE

MAX_FILES = 500
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_QUESTION_CHARS = 1000
MAX_K = 20


@dataclass
class _File:
    key: tuple
    chunks: list[Chunk]
    tokens: list[list[str]]


@dataclass
class FolderIndex:
    guard: Guard
    files: dict[str, _File] = field(default_factory=dict)

    def _candidates(self) -> list[str]:
        out = []
        for root, dirs, names in os.walk(self.guard.allowed, followlinks=False):
            dirs.sort()
            for n in sorted(names):
                if Path(n).suffix.lower() in SUFFIXES:
                    out.append(os.path.relpath(os.path.join(root, n), self.guard.allowed))
        return out

    def refresh(self) -> dict:
        """Read the folder through the guard; returns counts (never names) of files used and skipped."""
        seen, skipped, total = set(), 0, 0
        for rel in self._candidates():
            if len(seen) >= MAX_FILES:
                skipped += 1
                continue
            try:
                r = self.guard.resolve(rel)
                st = os.stat(r)
                key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                name = r.relative_to(self.guard.allowed).as_posix()
                if total + st.st_size > MAX_TOTAL_BYTES:
                    skipped += 1
                    continue
                if name not in self.files or self.files[name].key != key:
                    _, text = self.guard.read_text(rel)
                    cs = chunks(text)
                    self.files[name] = _File(key, cs, [tokens(c.text) for c in cs])
                seen.add(name)
                total += st.st_size
            except (PathRefused, OSError):
                skipped += 1
        for name in set(self.files) - seen:
            del self.files[name]
        return {"files_searched": len(seen), "files_skipped": skipped}

    def search(self, question: str, k: int = 5) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be 1 to {MAX_QUESTION_CHARS} characters")
        if not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be 1 to {MAX_K}")
        counts = self.refresh()
        rows = [(name, c) for name, f in sorted(self.files.items()) for c in f.chunks]
        if not rows:
            return {**counts, "results": [], "not_advice": NOT_ADVICE}
        scores = BM25([t for f in (self.files[n] for n in sorted(self.files)) for t in f.tokens]).scores(
            tokens(question))
        best = sorted(range(len(rows)), key=lambda i: (-scores[i], i))[:k]
        results = []
        for i in best:
            name, c = rows[i]
            text = c.text[:MAX_PASSAGE_CHARS] + ("..." if len(c.text) > MAX_PASSAGE_CHARS else "")
            results.append({"file": name, "offset": c.start, "text": text, "score": round(float(scores[i]), 4)})
        return {**counts, "results": results, "not_advice": NOT_ADVICE}
