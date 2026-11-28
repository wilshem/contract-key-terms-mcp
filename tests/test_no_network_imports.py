import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "key_terms_mcp"
BANNED = {"requests", "httpx", "urllib", "http", "socket", "aiohttp", "urllib3", "ftplib", "smtplib"}


SETUP = "setup_embeddings.py"  # the explicit download command: the only module allowed the network


def test_no_network_imports():
    found = []
    for f in SRC.rglob("*.py"):
        if f.name == SETUP:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            found += [(f.name, n) for n in names if n.split(".")[0] in BANNED]
    assert found == []


def test_server_never_imports_the_setup_command():
    """Nothing the server loads imports setup_embeddings (checked on the source, transitively)."""
    seen, todo = set(), ["server"]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        tree = ast.parse((SRC / f"{mod}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                names = [node.module] if node.module else [a.name for a in node.names]
                todo += [n for n in names if (SRC / f"{n}.py").exists()]
    assert "setup_embeddings" not in seen and "search" in seen and "usage_log" in seen
