"""Switch-offs: each guard is turned off in turn, and the case it exists for then gets through. This shows each
refusal in test_paths.py comes from its own check, and that the tests would catch the check being removed."""
import os

import pytest

from key_terms_mcp import paths
from key_terms_mcp.model import MODEL_PATH, ModelIntegrityError, SentenceModel
from key_terms_mcp.paths import Guard, PathRefused


def test_inside_check(home):
    on = Guard.create(home / "allowed", home=home)
    with pytest.raises(PathRefused):
        on.read_text("../outside.txt")
    off = Guard.create(home / "allowed", home=home, check_inside=False)
    assert off.read_text("../outside.txt")[1] == "outside"


def test_denied_check(home):
    on = Guard.create(home, home=home)
    with pytest.raises(PathRefused):
        on.read_text("IJGB/pilots/p1/a.txt")
    off = Guard.create(home, home=home, check_denied=False)
    assert off.read_text("IJGB/pilots/p1/a.txt")


def test_size_check(home):
    with pytest.raises(PathRefused):
        Guard.create(home / "allowed", home=home, max_bytes=10).read_text("ok.txt")
    assert Guard.create(home / "allowed", home=home, max_bytes=10, check_size=False).read_text("ok.txt")


def test_open_check(home, monkeypatch):
    """A link swapped in after the check: O_NOFOLLOW and the inode comparison refuse it."""
    link = home / "allowed" / "swapped.txt"
    os.symlink(home / "outside.txt", link)
    real = (home / "allowed" / "ok.txt")
    monkeypatch.setattr(Guard, "resolve", lambda self, path: link)
    monkeypatch.setattr(paths.os, "stat", lambda p, *a, **k: os.lstat(real) if p == link else os.lstat(p))
    with pytest.raises(PathRefused):
        Guard.create(home / "allowed", home=home).read_text("ok.txt")
    off = Guard.create(home / "allowed", home=home, check_open=False)
    assert off.read_text("ok.txt")[1] == "outside"


def test_sha256_check(tmp_path):
    import gzip
    import hashlib
    import json
    data = json.loads(gzip.decompress(MODEL_PATH.read_bytes()))
    data["clauses"]["governing_law"]["intercept"] += 5.0
    bad = tmp_path / "m.json.gz"
    bad.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ModelIntegrityError):
        SentenceModel.load(bad)
    # with the check pointed at the tampered bytes, the tampered model loads: only the check stops it
    loaded = SentenceModel.load(bad, expected_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())
    assert loaded.clauses["governing_law"]["intercept"] == data["clauses"]["governing_law"]["intercept"]
