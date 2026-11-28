"""The usage log: off by default, exact fields when on, never text, queries or paths, refuses a linked file."""
import json
import os
import stat

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from key_terms_mcp.paths import Guard
from key_terms_mcp.server import SAMPLES, build
from key_terms_mcp.usage_log import FIELDS, UsageLog

MARKER_Q = "zqxmarkerquestion"
MARKER_PATH = "zqxmarkerpath.txt"


def test_off_by_default():
    assert not UsageLog.from_settings(env={}).enabled
    assert not UsageLog.from_settings(env={"KEY_TERMS_LOG": "off"}).enabled


def test_on_by_flag_or_setting(tmp_path):
    f = tmp_path / "u.jsonl"
    assert UsageLog.from_settings(flag=True, env={"KEY_TERMS_LOG_FILE": str(f)}).path == f
    assert UsageLog.from_settings(env={"KEY_TERMS_LOG": "on", "KEY_TERMS_LOG_FILE": str(f)}).path == f


async def _calls(server):
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        await client.call_tool("list_clause_types", {})
        await client.call_tool("search_clauses", {"question": MARKER_Q, "k": 1})
        await client.call_tool("extract_key_terms", {"path": MARKER_PATH})  # refused: file not found


async def test_logged_fields_and_no_text_queries_or_paths(tmp_path):
    f = tmp_path / "state" / "usage.jsonl"
    await _calls(build(Guard.create(SAMPLES), UsageLog(f)))
    raw = f.read_text()
    lines = [json.loads(x) for x in raw.splitlines()]
    assert [x["tool"] for x in lines] == ["list_clause_types", "search_clauses", "extract_key_terms"]
    for x in lines:
        assert tuple(x) == FIELDS and isinstance(x["duration_ms"], int)
    assert [x["ok"] for x in lines] == [True, True, False]
    assert lines[2]["error"] == "PathRefused"
    for banned in (MARKER_Q, MARKER_PATH, "zqxmarker", str(SAMPLES), "Agreement", "file not found"):
        assert banned not in raw
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(f.parent.stat().st_mode) == 0o700


async def test_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    await _calls(build(Guard.create(SAMPLES), UsageLog.from_settings(env={})))
    assert not any(tmp_path.rglob("*.jsonl"))


def test_linked_log_file_refused(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep\n")
    link = tmp_path / "usage.jsonl"
    os.symlink(target, link)
    log = UsageLog(link)
    log.write("list_clause_types", 0.01, None)
    assert target.read_text() == "keep\n" and log.failures == 1


def test_log_failure_never_breaks_the_call(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = UsageLog(blocker / "sub" / "usage.jsonl")  # parent is a file: cannot be created
    assert log.wrap(lambda: 42)() == 42 and log.failures == 1


def test_errors_still_raise_and_are_logged(tmp_path):
    log = UsageLog(tmp_path / "u.jsonl")

    def boom():
        raise ValueError("secret detail")
    with pytest.raises(ValueError):
        log.wrap(boom)()
    line = json.loads((tmp_path / "u.jsonl").read_text())
    assert line["ok"] is False and line["error"] == "ValueError" and "secret" not in json.dumps(line)


def test_rotates_at_about_one_megabyte(tmp_path):
    f = tmp_path / "u.jsonl"
    f.write_bytes(b"x" * (1024 * 1024 + 1))
    UsageLog(f).write("list_clause_types", 0.0, None)
    assert (tmp_path / "u.jsonl.1").stat().st_size > 1024 * 1024 and len(f.read_text().splitlines()) == 1
