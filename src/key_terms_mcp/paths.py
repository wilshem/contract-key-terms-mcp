"""The path limit: the server reads only regular .txt or .md files inside one allowed folder, fixed at start.

Every path is resolved first (symbolic links followed, ".." removed), then checked: inside the allowed folder, not
inside a denied folder, a regular file with an allowed suffix, under the size cap. The file is then opened without
following a link, and its device and inode are compared with the checked file, so a swap between check and open is
refused. A file with more than one hard link is refused, since another of its names could be in a denied folder.
Denied folders are compared ignoring case. Errors name the rule that refused, never the file's contents.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = (".txt", ".md")
MAX_BYTES = 2 * 1024 * 1024
ENV_DENY_DIRS = "KEY_TERMS_DENY_DIRS"    # folders never read, separated by the system's path separator
ENV_DENY_NAMES = "KEY_TERMS_DENY_NAMES"  # folder names never read, wherever they are, separated the same way
# never read, whatever the settings say: the usual places for keys and credentials
DEFAULT_DENY_DIRS = ("~/.ssh", "~/.gnupg", "~/.aws", "~/.config", "~/.netrc", "~/.docker", "~/.kube")


class PathRefused(ValueError):
    """A path the server will not read."""


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(os.pathsep) if v.strip()]


def _expand(entry: str, home: Path) -> Path:
    """A setting's folder: "~" means the home folder; a relative path is read from the current folder."""
    p = home / entry[2:] if entry == "~" or entry.startswith("~/") else Path(entry)
    return _resolved(p)


def _inside(p: Path, root: Path) -> bool:
    return p == root or p.is_relative_to(root)


def _folded(p: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in p.parts)


def _inside_ignoring_case(p: Path, root: Path) -> bool:
    """Like _inside, but names are compared case-folded, on every system ("~/.SSH" is inside "~/.ssh")."""
    pf, rf = _folded(p), _folded(root)
    return pf[: len(rf)] == rf


def _resolved(p: Path) -> Path:
    try:
        return p.resolve(strict=False)
    except (OSError, RuntimeError):  # a link loop
        return p


@dataclass(frozen=True)
class Guard:
    allowed: Path
    deny_dirs: tuple[Path, ...] = ()
    deny_names: frozenset[str] = frozenset()
    max_bytes: int = MAX_BYTES
    # switches, for the switch-off tests only: each turns one check off
    check_inside: bool = True
    check_denied: bool = True
    check_open: bool = True
    check_size: bool = True
    check_links: bool = True
    ignore_case: bool = True

    @classmethod
    def create(cls, allowed: str | os.PathLike, deny_dirs: list[str] | None = None,
               deny_names: list[str] | None = None, home: Path | None = None, env: dict | None = None,
               **switches) -> Guard:
        """The guard, fixed for the server's life. Denied folders are the defaults plus deny_dirs (or, when None, the
        KEY_TERMS_DENY_DIRS setting); denied folder names are deny_names (or the KEY_TERMS_DENY_NAMES setting)."""
        home = home or Path.home()
        env = os.environ if env is None else env
        dirs = list(DEFAULT_DENY_DIRS) + (_split(env.get(ENV_DENY_DIRS)) if deny_dirs is None else list(deny_dirs))
        names = _split(env.get(ENV_DENY_NAMES)) if deny_names is None else list(deny_names)
        try:
            root = _expand(str(allowed), home).resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathRefused(f"allowed folder does not exist or cannot be resolved: {e.__class__.__name__}") from None
        if not root.is_dir():
            raise PathRefused("allowed folder is not a folder")
        guard = cls(allowed=root, deny_dirs=tuple(_expand(d, home) for d in dirs),
                    deny_names=frozenset(n.casefold() if switches.get("ignore_case", True) else n for n in names),
                    **switches)
        if guard.is_denied(root, folder=True):
            raise PathRefused("allowed folder is, or is inside, a denied folder")
        return guard

    def is_denied(self, p: Path, folder: bool = False) -> bool:
        """Denied folders and folder names are compared ignoring case, on every system, so a folder that differs from
        a denied one only in case is refused too."""
        inside = _inside_ignoring_case if self.ignore_case else _inside
        if any(inside(p, d) for d in self.deny_dirs):
            return True
        folders = p.parts if folder else p.parts[:-1]
        return any((part.casefold() if self.ignore_case else part) in self.deny_names for part in folders)

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
        if self.check_denied and self.is_denied(r):
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
            if self.check_links and fs.st_nlink > 1:
                # a hard link is another name for the same file, possibly in a denied folder, which no path shows
                raise PathRefused("file has more than one hard link")
            raw = fh.read(self.max_bytes + 1)
        if self.check_size and len(raw) > self.max_bytes:
            raise PathRefused(f"file is larger than {self.max_bytes} bytes")
        return r, raw.decode("utf-8", errors="replace")
