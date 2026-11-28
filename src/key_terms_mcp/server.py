"""MCP server: read-only tools over the CUAD-trained key-terms sentence model, and passage search.

Runs over stdio only. Reads only .txt and .md files inside one allowed folder fixed at start (the bundled samples by
default). Makes no network calls. Writes nothing, unless its usage log is turned on (--log or KEY_TERMS_LOG=on).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import explain as _explain
from . import reader
from .clause_types import CLAUSE_TYPES
from .model import model
from .paths import Guard
from .search import FolderIndex
from .usage_log import UsageLog

SAMPLES = Path(__file__).resolve().parent / "samples"
ENV_ALLOWED = "KEY_TERMS_ALLOWED_DIR"


def build(guard: Guard, log: UsageLog | None = None) -> FastMCP:
    mcp = FastMCP("contract-key-terms")
    log = log or UsageLog(None)
    index = FolderIndex(guard)

    def tool(fn):
        return mcp.tool()(log.wrap(fn))

    @tool
    def list_clause_types() -> list[dict]:
        """The 36 CUAD clause types: key, plain label, CUAD category name, the model's threshold, and whether the
        model is used for that type (for switched-off types it is not, and extract_key_terms does not report
        them)."""
        m = model()
        return [{"key": k, "label": label, "cuad_category": cuad, "threshold": m.clauses[k]["threshold"],
                 "model_used": reader.model_used(m, k)} for k, (label, cuad) in CLAUSE_TYPES.items()]

    @tool
    def extract_key_terms(path: str) -> dict:
        """Key terms of one agreement (.txt or .md inside the allowed folder; a relative path is read from it): for
        each clause type, whether the model found it and up to two sentences quoted as written, best first."""
        r, text = guard.read_text(path)
        return {"file": r.relative_to(guard.allowed).as_posix(), "clauses": reader.read(text),
                "not_advice": reader.NOT_ADVICE}

    @tool
    def explain_prediction(path: str, clause_type: str) -> dict:
        """Why the model scored one clause type as it did in one agreement: its best sentences with the words and
        word pairs that pushed the score up (and down) the most, and the clause type's top-weighted terms overall."""
        r, text = guard.read_text(path)
        return {"file": r.relative_to(guard.allowed).as_posix(), **_explain.explain(text, clause_type)}

    @tool
    def search_clauses(question: str, k: int = 5) -> dict:
        """The passages in the allowed folder's .txt and .md files that best match a question, best first: file,
        character offset, the passage (cut at 600 characters) and its keyword (BM25) score. Files the guard refuses
        are skipped and only counted."""
        return index.search(question, k)

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="contract-key-terms-mcp", description=__doc__)
    ap.add_argument("--allow", metavar="DIR", default=os.environ.get(ENV_ALLOWED) or str(SAMPLES),
                    help=f"the one folder files may be read from (default: the bundled samples; or ${ENV_ALLOWED})")
    ap.add_argument("--log", action="store_true",
                    help="write the usage log (time, tool, duration, success; never text, queries or paths)")
    args = ap.parse_args(argv)
    build(Guard.create(args.allow), UsageLog.from_settings(args.log)).run()  # stdio


if __name__ == "__main__":
    main()
