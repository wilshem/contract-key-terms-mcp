"""The scripted terminal demo behind docs/demo.gif: an MCP client calls the three tools on a bundled sample."""
import asyncio
import json
import sys
import time

from mcp.shared.memory import create_connected_server_and_client_session

from key_terms_mcp.paths import Guard
from key_terms_mcp.server import SAMPLES, build


def say(text, delay=0.02):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")


def result(res):
    sc = res.structuredContent
    return sc.get("result", sc) if sc else json.loads(res.content[0].text)


async def main():
    async with create_connected_server_and_client_session(build(Guard.create(SAMPLES))._mcp_server) as client:
        say("$ list_clause_types()")
        types = result(await client.call_tool("list_clause_types", {}))
        print(f"  {len(types)} CUAD clause types, model used for {sum(t['model_used'] for t in types)}")
        for t in types[:4]:
            print(f"  - {t['key']}: {t['label']}")
        print("  ...")
        await asyncio.sleep(1.5)
        say('\n$ extract_key_terms(path="services_agreement.txt")')
        out = result(await client.call_tool("extract_key_terms", {"path": "services_agreement.txt"}))
        for k, v in out["clauses"].items():
            if v["found"]:
                p = v["passages"][0]
                print(f"  {k}  score {p['score']}\n    \"{p['text'][:90]}...\"")
        await asyncio.sleep(2)
        say('\n$ explain_prediction(path="services_agreement.txt", clause_type="termination_for_convenience")')
        ex = result(await client.call_tool("explain_prediction", {"path": "services_agreement.txt",
                                                                   "clause_type": "termination_for_convenience"}))
        b = ex["best_sentences"][0]
        print(f"  probability {b['probability']}  threshold {ex['threshold']}")
        for t in b["top_terms"][:5]:
            print(f"    +{t['contribution']:.2f}  \"{t['term']}\"")
        await asyncio.sleep(1.5)
        say('\n$ extract_key_terms(path="../secret.txt")')
        res = await client.call_tool("extract_key_terms", {"path": "../secret.txt"})
        print("  refused:", " ".join(c.text for c in res.content if hasattr(c, "text"))[:80])
        await asyncio.sleep(2.5)


if __name__ == "__main__":
    asyncio.run(main())
