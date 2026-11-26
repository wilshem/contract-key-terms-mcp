"""Fail if the repository holds anything that should not be public: big files other than the model, data folders,
or private paths, hostnames or e-mail addresses. Run in CI on every pull request."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "src/key_terms_mcp/models/key_terms_model.json.gz"
MAX_BYTES = 1_000_000
SELF = "scripts/check_public_ready.py"  # holds the patterns themselves
PRIVATE = re.compile(r"IJGB|pip-data|/home/\w+|\\\\wsl|C:\\\\Users|@itjustgetsbetter|\.local\b|BEGIN [A-Z ]*PRIVATE KEY")


def main() -> int:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    problems = []
    for f in files:
        p = ROOT / f
        if "data" in Path(f).parts[:-1]:
            problems.append(f"{f}: inside a data/ folder")
        if f != MODEL and p.stat().st_size > MAX_BYTES:
            problems.append(f"{f}: larger than {MAX_BYTES} bytes")
        if f == SELF or f == MODEL or f.endswith((".gz", ".gif", ".png")):
            continue
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if PRIVATE.search(line):
                problems.append(f"{f}:{n}: private path, host or address")
    print("\n".join(problems) or f"public-ready check: {len(files)} files, no problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
