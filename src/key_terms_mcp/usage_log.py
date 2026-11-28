"""Usage log: one JSON line per tool call, OFF unless turned on.

Turn it on with --log or KEY_TERMS_LOG=on. Each line holds only: the time (UTC), the tool's name, how long the call
took, whether it succeeded, the error's class name when it did not (never its message), and the server version.
It never holds document text, query text, paths or arguments. The file is the user's own, outside the allowed
folder by default (~/.contract-key-terms-mcp/usage.jsonl, or KEY_TERMS_LOG_FILE), created, on Linux
and macOS, with mode 0600 in a folder of mode 0700 and opened without following a link; it is kept to about 1 MB
with one older copy. If the log cannot be written, the tool call still succeeds.
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import os
import time
from pathlib import Path

from . import __version__

ENV_LOG = "KEY_TERMS_LOG"
ENV_LOG_FILE = "KEY_TERMS_LOG_FILE"
DEFAULT_FILE = Path("~/.contract-key-terms-mcp/usage.jsonl")
MAX_BYTES = 1024 * 1024
FIELDS = ("ts", "tool", "duration_ms", "ok", "error", "version")


class UsageLog:
    def __init__(self, path: Path | None):
        self.path = path  # None: off
        self.failures = 0

    @classmethod
    def from_settings(cls, flag: bool = False, env: dict | None = None) -> UsageLog:
        env = os.environ if env is None else env
        on = flag or env.get(ENV_LOG, "").strip().lower() in ("on", "1", "true", "yes")
        if not on:
            return cls(None)
        return cls(Path(env.get(ENV_LOG_FILE) or DEFAULT_FILE).expanduser())

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, tool: str, seconds: float, error: BaseException | None) -> None:
        if self.path is None:
            return
        line = json.dumps({"ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"), "tool": tool,
                           "duration_ms": round(seconds * 1000), "ok": error is None,
                           "error": type(error).__name__ if error is not None else None,
                           "version": __version__}, separators=(",", ":")) + "\n"
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                if self.path.stat().st_size > MAX_BYTES and not self.path.is_symlink():
                    os.replace(self.path, self.path.with_name(self.path.name + ".1"))
            except FileNotFoundError:
                pass
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            self.failures += 1

    def wrap(self, fn):
        """The tool function, logged. The signature is kept, so the tool's parameters are unchanged."""
        @functools.wraps(fn)
        def logged(*args, **kwargs):
            t0, err = time.monotonic(), None
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                err = e
                raise
            finally:
                self.write(fn.__name__, time.monotonic() - t0, err)
        return logged
