"""The path limit: the server reads only regular .txt or .md files inside one allowed folder, fixed at start.

Every path is resolved first (symbolic links followed, ".." removed), then checked: inside the allowed folder, not
inside a denied folder, a regular file with an allowed suffix, under the size cap. The file is then opened without
following a link, and its device and inode are compared with the checked file, so a swap between check and open is
refused. Errors name the rule that refused, never the file's contents.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = (".txt", ".md")
MAX_BYTES = 2 * 1024 * 1024


class PathRefused(ValueError):
    """A path the server will not read."""


def denied_roots(home: Path | None = None) -> list[Path]:
    """Folders never read, even inside the allowed folder: private work and client data folders on the author's
    machine. Kept in code so no setting can remove them."""
    base = (home or Path.home()) / "IJGB"
    return [base / "pilots", base / "clients", base / "pip-data", base / "internal"]


def _inside(p: Path, root: Path) -> bool:
    return p == root or p.is_relative_to(root)


def _resolved(p: Path) -> Path:
    try:
        return p.resolve(strict=False)
    except (OSError, RuntimeError):  # a link loop
        return p


def is_denied(p: Path, home: Path | None = None) -> bool:
    home = home or Path.home()
    work = _resolved(home / "IJGB")
    if any(_inside(p, _resolved(r)) for r in denied_roots(home)):
        return True
    if not _inside(p, work):
        return False
    parts = p.relative_to(work).parts
    folders = parts if p.is_dir() else parts[:-1]
    return "data" in folders  # any folder named "data" under the work folder


@dataclass(frozen=True)
class Guard:
    allowed: Path
    home: Path
    max_bytes: int = MAX_BYTES
    # switches, for the switch-off tests only: each turns one check off
    check_inside: bool = True
    check_denied: bool = True
    check_open: bool = True
    check_size: bool = True

    @classmethod
    def create(cls, allowed: str | os.PathLike, home: Path | None = None, **switches) -> Guard:
        home = home or Path.home()
        try:
            root = Path(allowed).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathRefused(f"allowed folder does not exist or cannot be resolved: {e.__class__.__name__}") from None
        if not root.is_dir():
            raise PathRefused("allowed folder is not a folder")
        if is_denied(root, home):
            raise PathRefused("allowed folder is, or is inside, a denied folder")
        return cls(allowed=root, home=home, **switches)

    def resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise PathRefused("empty or invalid path")
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.allowed / p
        try:
            r = p.resolve(strict=True)
        except FileNotFoundError:
            raise PathRefused("file not found") from None
        except (OSError, RuntimeError):
            raise PathRefused("path cannot be resolved (for example a link loop)") from None
        if self.check_inside and not _inside(r, self.allowed):
            raise PathRefused("path is outside the allowed folder")
        if self.check_denied and is_denied(r, self.home):
            raise PathRefused("path is inside a denied folder")
        if r.suffix.lower() not in SUFFIXES:
            raise PathRefused(f"only {', '.join(SUFFIXES)} files are read")
        return r

    def read_text(self, path: str) -> tuple[Path, str]:
        r = self.resolve(path)
        st = os.stat(r)
        if not stat.S_ISREG(st.st_mode):
            raise PathRefused("not a regular file")
        if self.check_size and st.st_size > self.max_bytes:
            raise PathRefused(f"file is larger than {self.max_bytes} bytes")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(r, flags if self.check_open else os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        except OSError:
            raise PathRefused("file could not be opened safely (it may have changed)") from None
        with os.fdopen(fd, "rb") as fh:
            fs = os.fstat(fh.fileno())
            if self.check_open and (not stat.S_ISREG(fs.st_mode) or (fs.st_dev, fs.st_ino) != (st.st_dev, st.st_ino)):
                raise PathRefused("file changed between check and open")
            raw = fh.read(self.max_bytes + 1)
        if self.check_size and len(raw) > self.max_bytes:
            raise PathRefused(f"file is larger than {self.max_bytes} bytes")
        return r, raw.decode("utf-8", errors="replace")
