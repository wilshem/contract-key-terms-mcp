import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "key_terms_mcp"
BANNED = {"requests", "httpx", "urllib", "http", "socket", "aiohttp", "urllib3", "ftplib", "smtplib"}


def test_no_network_imports():
    found = []
    for f in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            found += [(f.name, n) for n in names if n.split(".")[0] in BANNED]
    assert found == []
