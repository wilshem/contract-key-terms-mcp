import json
import socket

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from key_terms_mcp.model import model
from key_terms_mcp.paths import Guard
from key_terms_mcp.server import SAMPLES, build


def payload(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc)
    return json.loads(res.content[0].text)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network use attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


async def test_tools_listed_and_called_without_network(no_network):
    server = build(Guard.create(SAMPLES))
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        assert names == ["explain_prediction", "extract_key_terms", "list_clause_types", "search_clauses"]
        types = payload(await client.call_tool("list_clause_types", {}))
        used = sum(c["use"] == "model" for c in model().clauses.values())
        assert len(types) == 36 and sum(t["model_used"] for t in types) == used
        out = payload(await client.call_tool("extract_key_terms", {"path": "services_agreement.txt"}))
        assert out["file"] == "services_agreement.txt" and len(out["clauses"]) == used
        ex = payload(await client.call_tool("explain_prediction", {"path": "supply_agreement.txt",
                                                                    "clause_type": "governing_law"}))
        assert ex["clause_type"] == "governing_law" and ex["best_sentences"]


async def test_refusal_is_an_error_without_contents(tmp_path):
    (tmp_path / "secret.txt").write_text("TOP SECRET CONTENT")
    server = build(Guard.create(SAMPLES))
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool("extract_key_terms", {"path": str(tmp_path / "secret.txt")})
        assert res.isError
        text = " ".join(c.text for c in res.content if hasattr(c, "text"))
        assert "outside the allowed folder" in text and "TOP SECRET" not in text
