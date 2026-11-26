import os

import pytest

from key_terms_mcp.paths import Guard, PathRefused


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
    g = Guard.create(home / "allowed", home=home, max_bytes=10)
    refused(g, "ok.txt", "larger than")


def test_empty_and_nul(guard):
    refused(guard, "", "empty or invalid")
    refused(guard, "a\x00.txt", "empty or invalid")


@pytest.mark.parametrize("rel", ["IJGB/pilots/p1/a.txt", "IJGB/clients/c1/a.txt", "IJGB/pip-data/a.txt",
                                 "IJGB/internal/a.txt", "IJGB/project/data/a.txt"])
def test_denied_even_inside_allowed(home, rel):
    g = Guard.create(home, home=home)  # the allowed folder contains the denied ones
    refused(g, rel, "denied folder")
    assert g.read_text("IJGB/project/ok.txt")


def test_symlink_into_denied(home):
    g = Guard.create(home, home=home)
    os.symlink(home / "IJGB/pilots/p1/a.txt", home / "allowed" / "p.txt")
    refused(g, "allowed/p.txt", "denied folder")


@pytest.mark.parametrize("rel", ["IJGB/pilots", "IJGB/pilots/p1", "IJGB/clients", "IJGB/pip-data", "IJGB/internal",
                                 "IJGB/project/data"])
def test_allowed_folder_cannot_be_denied(home, rel):
    with pytest.raises(PathRefused):
        Guard.create(home / rel, home=home)


def test_allowed_folder_must_exist(home):
    with pytest.raises(PathRefused):
        Guard.create(home / "missing", home=home)
