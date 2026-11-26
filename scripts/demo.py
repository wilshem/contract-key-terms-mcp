"""The scripted terminal demo behind docs/demo.gif: an MCP client calls the three tools on a bundled sample.

`transcript()` returns the steps (a typed command, its real output lines, a pause); `main()` prints them with a
typing effect, and scripts/render_demo_gif.py draws the same steps into the GIF.
"""
import asyncio
import json
import sys
import time

from mcp.shared.memory import create_connected_server_and_client_session

from key_terms_mcp.paths import Guard
from key_terms_mcp.server import SAMPLES, build


def _result(res):
    sc = res.structuredContent
    return sc.get("result", sc) if sc else json.loads(res.content[0].text)


async def transcript() -> list[tuple[str, str | float]]:
    steps: list[tuple[str, str | float]] = []

    def cmd(t):
        steps.append(("cmd", t))

    def out(t):
        steps.append(("out", t))

    def pause(s):
        steps.append(("pause", s))

    guard = Guard.create(SAMPLES, deny_dirs=[], deny_names=[], env={})
    async with create_connected_server_and_client_session(build(guard)._mcp_server) as client:
        cmd("list_clause_types()")
        types = _result(await client.call_tool("list_clause_types", {}))
        out(f"  {len(types)} CUAD clause types, model used for {sum(t['model_used'] for t in types)}")
        for t in types[:4]:
            out(f"  - {t['key']}: {t['label']}")
        out("  ...")
        pause(1.5)
        cmd('extract_key_terms(path="services_agreement.txt")')
        res = _result(await client.call_tool("extract_key_terms", {"path": "services_agreement.txt"}))
        for k, v in res["clauses"].items():
            if v["found"]:
                p = v["passages"][0]
                out(f"  {k}  score {p['score']}")
                out(f"    \"{p['text'][:80]}...\"")
        pause(2.5)
        cmd('explain_prediction(path="services_agreement.txt", clause_type="termination_for_convenience")')
        ex = _result(await client.call_tool("explain_prediction", {"path": "services_agreement.txt",
                                                                    "clause_type": "termination_for_convenience"}))
        b = ex["best_sentences"][0]
        out(f"  probability {b['probability']}  threshold {ex['threshold']}")
        for t in b["top_terms"][:5]:
            out(f"    +{t['contribution']:.2f}  \"{t['term']}\"")
        pause(2.0)
        cmd('extract_key_terms(path="../server.py")')
        res = await client.call_tool("extract_key_terms", {"path": "../server.py"})
        text = " ".join(c.text for c in res.content if hasattr(c, "text"))
        out("  refused (error): " + text[:80])
        pause(3.0)
    return steps


def main() -> None:
    for kind, value in asyncio.run(transcript()):
        if kind == "cmd":
            for ch in f"$ {value}":
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(0.02)
            sys.stdout.write("\n")
        elif kind == "out":
            print(value)
        else:
            time.sleep(float(value))


if __name__ == "__main__":
    main()
