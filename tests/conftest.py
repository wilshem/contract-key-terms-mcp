from pathlib import Path

import pytest

from key_terms_mcp.model import model
from key_terms_mcp.paths import Guard

SAMPLES = Path(__file__).resolve().parent.parent / "src" / "key_terms_mcp" / "samples"


@pytest.fixture(scope="session")
def m():
    return model()


@pytest.fixture
def home(tmp_path):
    """A fake home with the denied work folders, and an allowed folder beside them."""
    h = tmp_path / "home"
    for d in ("IJGB/pilots/p1", "IJGB/clients/c1", "IJGB/pip-data", "IJGB/internal", "IJGB/project/data", "allowed"):
        (h / d).mkdir(parents=True)
    for f in ("IJGB/pilots/p1/a.txt", "IJGB/clients/c1/a.txt", "IJGB/pip-data/a.txt", "IJGB/internal/a.txt",
              "IJGB/project/data/a.txt", "IJGB/project/ok.txt"):
        (h / f).write_text("Neither party may assign this Agreement without consent.")
    (h / "allowed" / "ok.txt").write_text("This Agreement shall be governed by the laws of the State of Delaware.")
    (h / "outside.txt").write_text("outside")
    return h


@pytest.fixture
def guard(home):
    return Guard.create(home / "allowed", home=home)
