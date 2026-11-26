"""MCP server: three read-only tools over the CUAD-trained key-terms sentence model.

Runs over stdio only. Reads only .txt and .md files inside one allowed folder fixed at start (the bundled samples by
default). Makes no network calls.
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

SAMPLES = Path(__file__).resolve().parent / "samples"
ENV_ALLOWED = "KEY_TERMS_ALLOWED_DIR"


def build(guard: Guard) -> FastMCP:
    mcp = FastMCP("contract-key-terms")

    @mcp.tool()
    def list_clause_types() -> list[dict]:
        """The 36 CUAD clause types: key, plain label, CUAD category name, the model's threshold, and whether the
        model is used for that type (for 3 types it is not, and extract_key_terms does not report them)."""
        m = model()
        return [{"key": k, "label": label, "cuad_category": cuad, "threshold": m.clauses[k]["threshold"],
                 "model_used": reader.model_used(m, k)} for k, (label, cuad) in CLAUSE_TYPES.items()]

    @mcp.tool()
    def extract_key_terms(path: str) -> dict:
        """Key terms of one agreement (.txt or .md inside the allowed folder; a relative path is read from it): for
        each clause type, whether the model found it and up to two sentences quoted as written, best first."""
        r, text = guard.read_text(path)
        return {"file": r.relative_to(guard.allowed).as_posix(), "clauses": reader.read(text),
                "not_advice": reader.NOT_ADVICE}

    @mcp.tool()
    def explain_prediction(path: str, clause_type: str) -> dict:
        """Why the model scored one clause type as it did in one agreement: its best sentences with the words and
        word pairs that pushed the score up (and down) the most, and the clause type's top-weighted terms overall."""
        r, text = guard.read_text(path)
        return {"file": r.relative_to(guard.allowed).as_posix(), **_explain.explain(text, clause_type)}

    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="contract-key-terms-mcp", description=__doc__)
    ap.add_argument("--allow", metavar="DIR", default=os.environ.get(ENV_ALLOWED) or str(SAMPLES),
                    help=f"the one folder files may be read from (default: the bundled samples; or ${ENV_ALLOWED})")
    args = ap.parse_args(argv)
    build(Guard.create(args.allow)).run()  # stdio


if __name__ == "__main__":
    main()
