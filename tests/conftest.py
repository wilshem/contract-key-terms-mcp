from pathlib import Path

import pytest

from key_terms_mcp.model import model
from key_terms_mcp.paths import Guard

SAMPLES = Path(__file__).resolve().parent.parent / "src" / "key_terms_mcp" / "samples"


@pytest.fixture(scope="session")
def m():
    return model()


DENY_DIRS = ["~/work/private", "~/work/records"]
DENY_NAMES = ["data"]


@pytest.fixture
def home(tmp_path):
    """A fake home with private folders, a folder holding a "data" folder, and an allowed folder beside them."""
    h = tmp_path / "home"
    for d in ("work/private/p1", "work/records", "work/project/data", ".ssh", "allowed"):
        (h / d).mkdir(parents=True)
    for f in ("work/private/p1/a.txt", "work/records/a.txt", "work/project/data/a.txt", ".ssh/a.txt",
              "work/project/ok.txt"):
        (h / f).write_text("Neither party may assign this Agreement without consent.")
    (h / "allowed" / "ok.txt").write_text("This Agreement shall be governed by the laws of the State of Delaware.")
    (h / "outside.txt").write_text("outside")
    return h


def make_guard(allowed, home, **kw):
    return Guard.create(allowed, deny_dirs=DENY_DIRS, deny_names=DENY_NAMES, home=home, env={}, **kw)


@pytest.fixture
def guard(home):
    return make_guard(home / "allowed", home)
