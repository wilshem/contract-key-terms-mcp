"""Evaluate search_clauses' retrieval on CUAD v1: vector search (bge-small-en-v1.5) against keyword search (BM25).

Ground truth: for each contract and each of the 36 clause types the lawyers marked in it, a chunk is relevant when
it overlaps a marked span of that type by at least one non-space character. The question is CUAD's own question
for that clause type, as written; a second, short form (the clause type's CUAD name alone) is reported beside it.

Contracts: split once, with a fixed seed and near-duplicate groups kept together, into a dev set (one fifth) and a
test set (four fifths). The chunk size is chosen on the dev set only, by the rule in CHUNK_RULE; every reported
number comes from the test set. Nothing in search is trained on CUAD labels: BM25's word weights come from the text
being searched, and the embedding model is used as released. (The clause model, trained on all 510 contracts, is
not used by search, so no contract is held out from it, and none needs to be.)

Headline: search within one contract (each question asked of that contract's chunks). Beside it: search over the
whole test pool at once.

Usage:
  uv run --group eval --extra search python scripts/evaluate_retrieval.py --contracts 5      # light timing check
  uv run --group eval --extra search python scripts/evaluate_retrieval.py --choose            # dev set: chunk size
  uv run --group eval --extra search python scripts/evaluate_retrieval.py --write             # test set: results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import evaluate_cuad as ev

from key_terms_mcp import embeddings
from key_terms_mcp.bm25 import BM25, K1, B, tokens
from key_terms_mcp.chunking import CHUNK_WORDS, chunks

SEED = 20261128
CANDIDATE_WORDS = (64, 128, 256)
CHUNK_RULE = ("on the dev set, the chunk size (64, 128 or 256 words) with the highest mean of vector and keyword "
              "within-contract recall@3 (CUAD questions); ties go to the smaller size")
KS = (1, 3, 5)
OUT = ROOT / "docs" / "evaluation"
CODE_FILES = ("scripts/evaluate_retrieval.py", "scripts/evaluate_cuad.py", "src/key_terms_mcp/chunking.py",
              "src/key_terms_mcp/bm25.py", "src/key_terms_mcp/embeddings.py", "src/key_terms_mcp/reader.py")


def questions(cuad: dict) -> dict[str, dict[str, str]]:
    """{clause key: {"cuad": CUAD's question, "name": the CUAD category name}}."""
    out = {}
    for doc in cuad["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                cat = qa["id"].rsplit("__", 1)[-1]
                key = ev.BY_CATEGORY.get(cat)
                if key and key not in out:
                    out[key] = {"cuad": qa["question"], "name": cat}
    return out


def split(docs: list[dict], seed: int) -> tuple[list[int], list[int]]:
    fold = ev.assign_folds(ev.groups([d["text"] for d in docs]), 5, seed)
    return [i for i, f in enumerate(fold) if f == 0], [i for i, f in enumerate(fold) if f != 0]


def relevant(doc: dict, cs: list, key: str) -> set[int]:
    return {j for j, c in enumerate(cs) if any(ev._touches(doc["text"], (c.start, c.end), sp)
                                              for sp in doc["spans"][key])}


def rank_metrics(order: list[int], rel: set[int]) -> dict:
    first = next((n for n, j in enumerate(order) if j in rel), None)
    out = {f"recall@{k}": int(first is not None and first < k) for k in KS}
    out["rr"] = 1 / (first + 1) if first is not None else 0.0
    return out


def _order(scores) -> list[int]:
    return sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))


def evaluate(docs: list[dict], idx: list[int], qs: dict, embedder, words: int, pool: bool = True) -> dict:
    """Within-contract (and, if pool, whole-pool) metrics for both methods and both question forms."""
    import numpy as np

    t0 = time.monotonic()
    cs = {i: chunks(docs[i]["text"], words) for i in idx}
    texts = [c.text for i in idx for c in cs[i]]
    vecs = embedder.passages(texts)
    offs, n = {}, 0
    for i in idx:
        offs[i] = (n, n + len(cs[i]))
        n += len(cs[i])
    embed_s = time.monotonic() - t0
    qv = {(k, form): embedder.query(q[form]) for k, q in qs.items() for form in ("cuad", "name")}
    toks = {i: [tokens(c.text) for c in cs[i]] for i in idx}
    rows = []  # one per (contract, clause type, form, method)
    for i in idx:
        bm = BM25(toks[i])
        a, b = offs[i]
        for k in qs:
            if not docs[i]["spans"][k]:
                continue
            rel = relevant(docs[i], cs[i], k)
            for form in ("cuad", "name"):
                rows.append({"key": k, "form": form, "method": "keyword",
                             **rank_metrics(_order(bm.scores(tokens(qs[k][form]))), rel)})
                rows.append({"key": k, "form": form, "method": "vector",
                             **rank_metrics(_order(vecs[a:b] @ qv[(k, form)]), rel)})
    result = {"within_contract": summarise(rows, qs), "chunks": n, "embed_seconds": round(embed_s, 1),
              "chunk_words_mean": round(float(np.mean([len(t.split()) for t in texts])), 1) if texts else 0,
              "chunk_words_max": max((len(t.split()) for t in texts), default=0),
              "truncated_chunks": embedder.truncated(texts) if hasattr(embedder, "truncated") else None}
    if pool:
        bm = BM25([t for i in idx for t in toks[i]])
        owner = [(i, j) for i in idx for j in range(len(cs[i]))]
        prow = []
        for k in qs:
            rel = {n for n, (i, j) in enumerate(owner)
                   if docs[i]["spans"][k] and any(ev._touches(docs[i]["text"], (cs[i][j].start, cs[i][j].end), sp)
                                                  for sp in docs[i]["spans"][k])}
            if not rel:
                continue
            for form in ("cuad", "name"):
                for method, scores in (("keyword", bm.scores(tokens(qs[k][form]))), ("vector", vecs @ qv[(k, form)])):
                    order = _order(scores)
                    prow.append({"key": k, "form": form, "method": method, **rank_metrics(order, rel),
                                 "precision@5": sum(j in rel for j in order[:5]) / 5})
        result["whole_pool"] = summarise(prow, qs, extra=("precision@5",))
    return result


def summarise(rows: list[dict], qs: dict, extra: tuple = ()) -> dict:
    out = {}
    for method in ("vector", "keyword"):
        for form in ("cuad", "name"):
            sel = [r for r in rows if r["method"] == method and r["form"] == form]
            names = [f"recall@{k}" for k in KS] + ["rr", *extra]
            per = {}
            for k in qs:
                rk = [r for r in sel if r["key"] == k]
                if rk:
                    per[k] = {m: round(sum(r[m] for r in rk) / len(rk), 4) for m in names} | {"queries": len(rk)}
            out[f"{method}/{form}"] = {
                "micro": {m: round(sum(r[m] for r in sel) / len(sel), 4) if sel else 0.0 for m in names},
                "macro": {m: round(sum(p[m] for p in per.values()) / len(per), 4) if per else 0.0 for m in names},
                "queries": len(sel), "per_type": per}
    return out


def choose(dev: dict[int, dict]) -> int:
    def score(w: int) -> float:
        wc = dev[w]["within_contract"]
        return (wc["vector/cuad"]["micro"]["recall@3"] + wc["keyword/cuad"]["micro"]["recall@3"]) / 2
    return max(sorted(dev), key=lambda w: (score(w), -w))


def code_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cuad-zip", type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contracts", type=int, help="time the first N dev contracts only; writes nothing")
    mode.add_argument("--choose", action="store_true", help="choose the chunk size on the dev set")
    mode.add_argument("--write", action="store_true", help="score the test set and write the results")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cuad, sha = ev.get_cuad(args.cuad_zip)
    docs = ev.contracts(cuad)
    qs = questions(cuad)
    dev, test = split(docs, args.seed)
    t0 = time.monotonic()
    embeddings.onnxruntime_without_telemetry()  # documented switch, before any InferenceSession (see embeddings.py)
    emb = embeddings.Embedder()
    load_s = time.monotonic() - t0
    if args.contracts:
        r = evaluate(docs, dev[:args.contracts], qs, emb, CHUNK_WORDS, pool=False)
        print(json.dumps({"contracts": args.contracts, "chunks": r["chunks"], "embed_seconds": r["embed_seconds"],
                          "model_load_seconds": round(load_s, 1), "seconds": round(time.monotonic() - t0, 1)}))
        return
    choice_file = args.out_dir / "retrieval_chunk_choice.json"
    if args.choose:
        devr = {w: evaluate(docs, dev, qs, emb, w, pool=False) for w in CANDIDATE_WORDS}
        w = choose(devr)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        choice_file.write_text(json.dumps({"rule": CHUNK_RULE, "chosen": w, "dev_contracts": len(dev),
                                           "seed": args.seed, "dev": devr}, indent=1, sort_keys=True) + "\n")
        print(f"chosen chunk size: {w} words (set CHUNK_WORDS in src/key_terms_mcp/chunking.py)", file=sys.stderr)
        return
    choice = json.loads(choice_file.read_text())
    if choice["chosen"] != CHUNK_WORDS:
        raise SystemExit(f"CHUNK_WORDS is {CHUNK_WORDS} but the dev set chose {choice['chosen']}")
    r = evaluate(docs, test, qs, emb, CHUNK_WORDS)
    result = {"cuad": {"release": "CUAD v1, https://doi.org/10.5281/zenodo.4595826", "sha256": sha,
                       "contracts": len(docs)},
              "seed": args.seed, "dev_contracts": len(dev), "test_contracts": len(test),
              "test_set": sorted(docs[i]["title"] for i in test),
              "embedding_model": {"name": embeddings.NAME, "revision": embeddings.REVISION,
                                  "licence": embeddings.LICENCE, "files": embeddings.FILES,
                                  "query_instruction": embeddings.QUERY_INSTRUCTION,
                                  "max_tokens": embeddings.MAX_TOKENS},
              "chunk_words": CHUNK_WORDS, "chunk_rule": CHUNK_RULE, "bm25": {"k1": K1, "b": B},
              "relevance": "a chunk is relevant when it overlaps a lawyer-marked span of the clause type by at least "
                           "one non-space character",
              "questions": {"cuad": "CUAD's own question for the clause type, as written",
                            "name": "the clause type's CUAD category name alone"},
              "truncated_chunks": emb.truncated([c.text for i in test for c in chunks(docs[i]["text"],
                                                                                       CHUNK_WORDS)]),
              **r, "versions": ev._versions(), "seconds": round(time.monotonic() - t0)}
    result["code_sha256"] = code_hashes()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "retrieval_results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (args.out_dir / "retrieval_results.md").write_text(markdown(result))
    print(f"wrote {args.out_dir / 'retrieval_results.json'}", file=sys.stderr)


def markdown(r: dict) -> str:
    def table(block: dict, extra: tuple = ()) -> str:
        cols = [f"recall@{k}" for k in KS] + ["rr", *extra]
        head = "| Method | Question | " + " | ".join(c.replace("rr", "MRR") for c in cols) + " | Queries |"
        lines = [head, "| --- | --- | " + " | ".join("---:" for _ in cols) + " | ---: |"]
        for name, b in block.items():
            m, f = name.split("/")
            lines.append(f"| {m} | {'CUAD question' if f == 'cuad' else 'type name'} | "
                         + " | ".join(f"{b['micro'][c]:.3f}" for c in cols) + f" | {b['queries']} |")
        return "\n".join(lines)

    return f"""# search_clauses: retrieval results on CUAD v1

Written by `scripts/evaluate_retrieval.py` (seed {r['seed']}); the numbers are in `retrieval_results.json`.

Test set: {r['test_contracts']} contracts ({r['dev_contracts']} more were used only to choose the chunk size:
{r['chunk_rule']}). Chunks of about {r['chunk_words']} words (longest {r['chunk_words_max']}: a heading
kept with the text it heads can add a few); {r['chunks']} in all, mean {r['chunk_words_mean']} words. Embedding model: {r['embedding_model']['name']} at revision {r['embedding_model']['revision'][:12]}
({r['embedding_model']['licence']} licence). Keyword search: BM25 (k1 {r['bm25']['k1']}, b {r['bm25']['b']}).
{r['relevance'].capitalize()}. recall@k is the share of questions with a relevant chunk among the top k; MRR is the
mean of 1 / the rank of the first relevant chunk.

## Within one contract (headline)

Each question is asked of one contract that has the clause type, over that contract's chunks.

{table(r['within_contract'])}

## Over the whole test set at once

Each clause type's question is asked once, over every test contract's chunks together.

{table(r['whole_pool'], ('precision@5',))}
"""


if __name__ == "__main__":
    main()
