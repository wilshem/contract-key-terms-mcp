import os

import pytest
from conftest import make_guard

from key_terms_mcp.paths import ENV_DENY_DIRS, ENV_DENY_NAMES, Guard, PathRefused


def refused(guard, path, words):
    with pytest.raises(PathRefused) as e:
        guard.read_text(path)
    assert words in str(e.value)


def test_relative_and_absolute_inside(guard, home):
    assert guard.read_text("ok.txt")[0] == home / "allowed" / "ok.txt"
    assert guard.read_text(str(home / "allowed" / "ok.txt"))[1].startswith("This Agreement")


def test_dotdot_outside(guard):
    refused(guard, "../outside.txt", "outside the allowed folder")


def test_absolute_outside(guard, home):
    refused(guard, str(home / "outside.txt"), "outside the allowed folder")


def test_symlink_to_outside(guard, home):
    os.symlink(home / "outside.txt", home / "allowed" / "link.txt")
    refused(guard, "link.txt", "outside the allowed folder")


def test_symlink_inside_ok(guard, home):
    os.symlink(home / "allowed" / "ok.txt", home / "allowed" / "alias.txt")
    assert guard.read_text("alias.txt")[0] == home / "allowed" / "ok.txt"


def test_symlink_loop(guard, home):
    os.symlink(home / "allowed" / "b.txt", home / "allowed" / "a.txt")
    os.symlink(home / "allowed" / "a.txt", home / "allowed" / "b.txt")
    refused(guard, "a.txt", "cannot be resolved")


def test_missing(guard):
    refused(guard, "nope.txt", "not found")


def test_directory_and_suffix(guard, home):
    (home / "allowed" / "d.txt").mkdir()
    refused(guard, "d.txt", "not a regular file")
    (home / "allowed" / "x.pdf").write_text("x")
    refused(guard, "x.pdf", "only .txt, .md")


def test_fifo(guard, home):
    os.mkfifo(home / "allowed" / "f.txt")
    refused(guard, "f.txt", "not a regular file")


def test_oversize(home):
    g = make_guard(home / "allowed", home, max_bytes=10)
    refused(g, "ok.txt", "larger than")


def test_empty_and_nul(guard):
    refused(guard, "", "empty or invalid")
    refused(guard, "a\x00.txt", "empty or invalid")


@pytest.mark.parametrize("rel", ["work/private/p1/a.txt", "work/records/a.txt", "work/project/data/a.txt",
                                 ".ssh/a.txt"])
def test_denied_even_inside_allowed(home, rel):
    g = make_guard(home, home)  # the allowed folder contains the denied ones
    refused(g, rel, "denied folder")
    assert g.read_text("work/project/ok.txt")


def test_symlink_into_denied(home):
    g = make_guard(home, home)
    os.symlink(home / "work/private/p1/a.txt", home / "allowed" / "p.txt")
    refused(g, "allowed/p.txt", "denied folder")


@pytest.mark.parametrize("rel", ["work/private", "work/private/p1", "work/records", "work/project/data", ".ssh"])
def test_allowed_folder_cannot_be_denied(home, rel):
    with pytest.raises(PathRefused):
        make_guard(home / rel, home)


def test_allowed_folder_must_exist(home):
    with pytest.raises(PathRefused):
        make_guard(home / "missing", home)


def test_settings_from_environment(home):
    env = {ENV_DENY_DIRS: os.pathsep.join(["~/work/private", str(home / "work/records")]), ENV_DENY_NAMES: "data"}
    g = Guard.create(home, home=home, env=env)
    for rel in ("work/private/p1/a.txt", "work/records/a.txt", "work/project/data/a.txt"):
        refused(g, rel, "denied folder")
    assert g.read_text("work/project/ok.txt")


def test_defaults_always_apply(home):
    g = Guard.create(home, home=home, env={})  # no settings: only the defaults
    refused(g, ".ssh/a.txt", "denied folder")
    assert g.read_text("work/private/p1/a.txt")  # not denied without the setting
    g2 = Guard.create(home, deny_dirs=[], home=home, env={})
    refused(g2, ".ssh/a.txt", "denied folder")  # an empty list never removes the defaults
