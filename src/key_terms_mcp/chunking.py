"""Split an agreement into passages for search: the reader's own sentences, packed in order into chunks of up to
CHUNK_WORDS words. A sentence longer than that is cut into windows of CHUNK_WORDS words. A heading-only line that
would end a chunk starts the next one instead, so it stays with the text it heads. A chunk's text is its pieces
joined by single spaces (only what the reader treats as sentences: text between them, such as dot leaders in a
table of contents, is left out). Each chunk keeps the character offsets of the text it came from, so a result can
always point back to where it is in the file.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .reader import heading_only, sentence_spans

CHUNK_WORDS = 256  # chosen on the development contracts: docs/evaluation/retrieval_chunk_choice.json
_WORD = re.compile(r"\S+")


class Chunk(NamedTuple):
    text: str   # its pieces, white space collapsed, joined by single spaces
    start: int  # character offsets in the original text: text[start:end]
    end: int


def _pieces(text: str, words: int) -> list[tuple[int, int, int, bool]]:
    """(start, end, word count, heading) per sentence, long sentences cut into windows of `words` words."""
    out = []
    for s, start, end in sentence_spans(text):
        spans = [m.span() for m in _WORD.finditer(text, start, end)]
        if len(spans) <= words:
            out.append((start, end, len(spans), heading_only(s)))
            continue
        for i in range(0, len(spans), words):
            w = spans[i:i + words]
            out.append((w[0][0], w[-1][1], len(w), False))
    return out


def chunks(text: str, words: int = CHUNK_WORDS) -> list[Chunk]:
    out: list[Chunk] = []
    group: list[tuple[int, int, int, bool]] = []

    def flush(keep_heading: bool) -> list[tuple[int, int, int, bool]]:
        carry = []
        if keep_heading and len(group) > 1 and group[-1][3]:
            carry = [group.pop()]
        if group:
            a, b = group[0][0], group[-1][1]
            out.append(Chunk(" ".join(w for g in group for w in text[g[0]:g[1]].split()), a, b))
        return carry

    for p in _pieces(text, words):
        if group and sum(g[2] for g in group) + p[2] > words:
            group = flush(True)
        group.append(p)
    flush(False)
    return out
