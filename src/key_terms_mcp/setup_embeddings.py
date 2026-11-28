"""contract-key-terms-mcp-setup: download the embedding model for vector search, once.

This is the only part of the package that uses the network, and only when you run it yourself. The server never
imports it. Each file comes from the pinned revision of BAAI/bge-small-en-v1.5 and is kept only if its SHA-256
matches the one written in embeddings.py; files that are already there and match are not downloaded again.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from .embeddings import FILES, LICENCE, NAME, REVISION, model_dir, sha256_file


def url(rel: str) -> str:
    return f"https://huggingface.co/{NAME}/resolve/{REVISION}/{rel}"


def fetch(rel: str, dest: Path, expected: str) -> str:
    if dest.is_file() and sha256_file(dest) == expected:
        return "already there"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url(rel), timeout=60) as r, part.open("wb") as f:
        while block := r.read(1 << 20):
            h.update(block)
            f.write(block)
    if h.hexdigest() != expected:
        part.unlink()
        raise SystemExit(f"{rel}: sha256 {h.hexdigest()} does not match the pinned {expected}; nothing kept")
    os.replace(part, dest)
    return "downloaded"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="contract-key-terms-mcp-setup", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, help="where to put the model (default: the user cache folder)")
    args = ap.parse_args(argv)
    folder = args.dir or model_dir()
    print(f"{NAME} (licence {LICENCE}) at revision {REVISION} -> {folder}", file=sys.stderr)
    for rel, sha in FILES.items():
        print(f"  {rel}: {fetch(rel, folder / rel, sha)}, sha256 {sha}", file=sys.stderr)
    if args.dir:
        print(f"set KEY_TERMS_EMBED_DIR={folder} for the server", file=sys.stderr)


if __name__ == "__main__":
    main()
