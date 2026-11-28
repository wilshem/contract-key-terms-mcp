"""Evaluate the clause model on CUAD v1 by 5-fold cross-validation by contract, and train the model that ships.

What it does, in order:
1. Gets CUAD_v1.zip (downloads it once into .cache/cuad/, git-ignored, and checks its sha256), and reads
   CUAD_v1.json from it. Only three programs in this repository use the network, and only when you run them:
   this script and scripts/evaluate_retrieval.py, which download CUAD (get_cuad, below) if it is not already
   cached, and the contract-key-terms-mcp-setup command, which downloads the embedding model. The server never
   does.
2. Splits each contract into sentences with the server's own reader, and labels a sentence positive for a clause
   type when it overlaps a lawyer-marked span of that type by at least one non-space character.
3. Groups near-duplicate contracts (Jaccard >= 0.8 on word 5-grams) so each group stays in one fold, and assigns
   the groups to 5 folds with a fixed seed.
4. For each fold: trains on the other 4 folds' contracts only (vocabulary, IDF and weights), picks each clause
   type's threshold by an inner 4-fold cross-validation on those training contracts (best document-level F1 on a
   0.01 grid, ties to the higher threshold), and scores the held-out contracts with the server's own SentenceModel.
5. Reports document-level and sentence-level precision, recall and F1 per clause type, micro and macro averages
   and bootstrap intervals, and applies the switch-off rule below.
6. With --write: trains the final model on all 510 contracts with the same recipe, takes its thresholds from the
   cross-validated (out-of-fold) probabilities, and writes the model, the results and the golden probabilities.

The recipe, the seed and the switch-off rule are fixed here, before any results were seen, and were not tuned.

Usage:
  uv run --group eval python scripts/evaluate_cuad.py --only-fold 0     # one fold: time and memory check
  uv run --group eval python scripts/evaluate_cuad.py --write           # the full run
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from key_terms_mcp.clause_types import CLAUSE_TYPES
from key_terms_mcp.model import SentenceModel
from key_terms_mcp.reader import heading_only, sentence_spans

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
CUAD_SHA256 = "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
CUAD_MEMBER = "CUAD_v1/CUAD_v1.json"
CACHE = ROOT / ".cache" / "cuad"
OUT = ROOT / "docs" / "evaluation"
MODEL_FILE = ROOT / "src" / "key_terms_mcp" / "models" / "key_terms_model.json.gz"
MODEL_PY = ROOT / "src" / "key_terms_mcp" / "model.py"
GOLDEN = ROOT / "tests" / "golden_probabilities.json"
# the files whose change makes the results stale (tests/test_evaluation_results.py checks them)
CODE_FILES = ("scripts/evaluate_cuad.py", "src/key_terms_mcp/reader.py", "src/key_terms_mcp/model.py",
              "src/key_terms_mcp/clause_types.py")

SEED = 20261127
FOLDS = 5
INNER_FOLDS = 4
RECIPE = {
    "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True, "lowercase": True,
                   "max_features": 30000},
    "classifier": {"model": "logistic regression, one per clause type", "solver": "liblinear", "C": 1.0,
                   "class_weight": "balanced"},
    "threshold": "best document-level F1 on a 0.01 grid (0.01 to 0.99), ties to the higher threshold, from "
                 "out-of-fold probabilities on training contracts only",
    "near_duplicates": "contracts with Jaccard similarity >= 0.8 on word 5-grams are kept in one fold",
    "decimals": 4,
}
SWITCH_OFF_F1 = 0.50
SWITCH_OFF_RULE = ("A clause type is switched off (the model is not used for it, and extract_key_terms does not "
                   "report it) when its cross-validated document-level F1, pooled over the five held-out folds, is "
                   "below 0.50. Switched-off types are still scored and listed separately in the results, outside "
                   "the averages.")
MATCH_RULES = {
    "sentence": "the server's own sentence splitter; heading-only lines are dropped, as the server does",
    "positive": "a sentence is positive for a clause type when it overlaps a lawyer-marked span of that type by at "
                "least one non-space character",
    "strict": "a sentence is positive only when at least half of its non-space characters lie inside lawyer-marked "
              "spans of that type (reported beside the main rule)",
    "flagged": "a sentence is flagged when its probability is at or above the clause type's threshold",
    "document": "a contract contains a clause type when the lawyers marked at least one span of it; the model finds "
                "it when at least one sentence is flagged",
    "span_recall": "share of lawyer-marked spans that overlap at least one flagged sentence",
}
BOOTSTRAP = 1000
GRID = [round(i / 100, 2) for i in range(1, 100)]
KEYS = list(CLAUSE_TYPES)
BY_CATEGORY = {cuad: key for key, (_, cuad) in CLAUSE_TYPES.items()}


# ---------- data ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_cuad(zip_path: Path | None) -> tuple[dict, str]:
    """CUAD_v1.json, and the zip's sha256 (checked against the release)."""
    path = zip_path or CACHE / "CUAD_v1.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CUAD_URL}", file=sys.stderr)
        part = path.with_suffix(".part")
        with urllib.request.urlopen(CUAD_URL, timeout=60) as r, part.open("wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        part.rename(path)
    got = sha256_file(path)
    if got != CUAD_SHA256:
        raise SystemExit(f"{path}: sha256 {got} is not the CUAD v1 release ({CUAD_SHA256})")
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(CUAD_MEMBER)), got


def contracts(cuad: dict) -> list[dict]:
    """[{title, text, spans: {key: [(start, end)]}}] for every contract, with the 36 clause types' spans."""
    out = []
    for doc in cuad["data"]:
        for para in doc["paragraphs"]:
            spans: dict[str, list[tuple[int, int]]] = {k: [] for k in KEYS}
            for qa in para["qas"]:
                key = BY_CATEGORY.get(qa["id"].rsplit("__", 1)[-1])
                if key is None:  # document name, parties and dates: not clause types
                    continue
                for a in qa["answers"]:
                    spans[key].append((a["answer_start"], a["answer_start"] + len(a["text"])))
            out.append({"title": doc["title"], "text": para["context"], "spans": spans})
    return out


def _touches(text: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return lo < hi and bool(text[lo:hi].strip())


def _nonspace(text: str, lo: int, hi: int) -> int:
    return sum(not c.isspace() for c in text[lo:hi])


def label(contract: dict) -> list[dict]:
    """The contract's scored sentences, each with its positive types under the main and the strict rule."""
    text, rows = contract["text"], []
    for s, start, end in sentence_spans(text):
        if heading_only(s):
            continue
        pos, strict = set(), set()
        total = _nonspace(text, start, end) or 1
        for k, spans in contract["spans"].items():
            hit = [sp for sp in spans if _touches(text, (start, end), sp)]
            if not hit:
                continue
            pos.add(k)
            covered = [False] * (end - start)
            for a, b in hit:
                for i in range(max(a, start), min(b, end)):
                    covered[i - start] = True
            if sum(c and not text[start + i].isspace() for i, c in enumerate(covered)) * 2 >= total:
                strict.add(k)
        rows.append({"text": s, "start": start, "end": end, "pos": pos, "strict": strict})
    return rows


# ---------- near-duplicate groups and folds ----------

def _shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "big")
            for i in range(max(len(words) - n + 1, 1))}


def groups(texts: list[str], threshold: float = 0.8, k: int = 256) -> list[int]:
    """Group id per contract: contracts with exact Jaccard >= threshold on word 5-grams share a group (transitively).
    Candidate pairs come from bottom-k sketches; each candidate is confirmed exactly."""
    sets = [_shingles(t) for t in texts]
    sketches = [set(sorted(s)[:k]) for s in sets]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            union = sorted(sketches[i] | sketches[j])[:k]
            est = sum(h in sketches[i] and h in sketches[j] for h in union) / max(len(union), 1)
            if est >= threshold - 0.1:
                exact = len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1)
                if exact >= threshold:
                    parent[find(i)] = find(j)
    roots = {r: n for n, r in enumerate(dict.fromkeys(find(i) for i in range(len(texts))))}
    return [roots[find(i)] for i in range(len(texts))]


def assign_folds(group_of: list[int], n: int, seed: int) -> list[int]:
    """Fold per contract: groups shuffled with the seed, then each (largest first) to the fold with fewest contracts."""
    members: dict[int, list[int]] = {}
    for i, g in enumerate(group_of):
        members.setdefault(g, []).append(i)
    order = list(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(members[g]))  # stable: equal sizes keep the shuffled order
    sizes, fold = [0] * n, [0] * len(group_of)
    for g in order:
        f = min(range(n), key=lambda x: (sizes[x], x))
        for i in members[g]:
            fold[i] = f
        sizes[f] += len(members[g])
    return fold


# ---------- training (scikit-learn, eval group only) ----------

def train(docs: list[list[dict]], seed: int, max_features: int) -> dict:
    """The recipe on these contracts' sentences: a model dict in the shipped format, thresholds not yet set."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    v = RECIPE["vectorizer"]
    vec = TfidfVectorizer(ngram_range=tuple(v["ngram_range"]), min_df=v["min_df"], sublinear_tf=v["sublinear_tf"],
                          lowercase=v["lowercase"], max_features=max_features, dtype=np.float64)
    rows = [r for d in docs for r in d]
    x = vec.fit_transform([r["text"] for r in rows])
    d = RECIPE["decimals"]
    clauses = {}
    for k in KEYS:
        y = [k in r["pos"] for r in rows]
        if any(y) and not all(y):
            c = RECIPE["classifier"]
            lr = LogisticRegression(solver=c["solver"], C=c["C"], class_weight=c["class_weight"], max_iter=1000,
                                    random_state=seed).fit(x, y)
            weights = {str(i): round(float(w), d) for i, w in enumerate(lr.coef_[0]) if round(float(w), d) != 0.0}
            intercept = round(float(lr.intercept_[0]), d)
        else:  # no positive (or no negative) sentence to learn from: never flags
            weights, intercept = {}, -50.0
        clauses[k] = {"threshold": 0.99, "use": "model", "intercept": intercept, "weights": weights}
    vocab = {t: int(i) for t, i in sorted(vec.vocabulary_.items(), key=lambda t: t[1])}
    return {"source": "", "vectorizer": {**RECIPE["vectorizer"], "max_features": max_features},
            "vocabulary": vocab, "idf": [round(float(f), d) for f in vec.idf_], "clauses": clauses}


def numpy_probabilities(model: dict, texts: list[str]):
    """Probabilities with scikit-learn's TF-IDF and the model's rounded IDF and weights: rows texts, columns KEYS.
    Used for the inner cross-validation (fast) and for the golden file; the held-out folds use SentenceModel."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = model["vectorizer"]
    vec = TfidfVectorizer(ngram_range=tuple(v["ngram_range"]), sublinear_tf=v["sublinear_tf"],
                          lowercase=v["lowercase"], vocabulary=model["vocabulary"], dtype=np.float64)
    vec.fit(["x"])
    vec.idf_ = np.array(model["idf"], dtype=float)
    x = vec.transform(texts)
    w = np.zeros((len(model["idf"]), len(KEYS)))
    b = np.zeros(len(KEYS))
    for j, k in enumerate(KEYS):
        c = model["clauses"][k]
        for i, val in c["weights"].items():
            w[int(i), j] = val
        b[j] = c["intercept"]
    z = np.clip(x @ w + b, -50, 50)
    return 1 / (1 + np.exp(-z))


def sentence_model_probabilities(model: dict, texts: list[str]) -> list[dict[str, float]]:
    m = SentenceModel(model)
    return [m.probabilities(t) for t in texts]


# ---------- thresholds and metrics ----------

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def best_threshold(doc_max: list[float], truth: list[bool]) -> float:
    """Best document-level F1 on the grid; ties go to the higher threshold."""
    best, best_f1 = GRID[-1], -1.0
    for t in GRID:
        tp = sum(m >= t and y for m, y in zip(doc_max, truth, strict=True))
        fp = sum(m >= t and not y for m, y in zip(doc_max, truth, strict=True))
        fn = sum(m < t and y for m, y in zip(doc_max, truth, strict=True))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 >= best_f1:
            best, best_f1 = t, f1
    return best


def oof_thresholds(train_docs: list[list[dict]], train_spans: list[dict], inner_fold: list[int], n: int, seed: int,
                   max_features: int) -> dict[str, float]:
    """Thresholds from out-of-fold probabilities on these training contracts only."""
    doc_max = [dict.fromkeys(KEYS, 0.0) for _ in train_docs]
    for f in range(n):
        fit = [d for d, g in zip(train_docs, inner_fold, strict=True) if g != f]
        held = [i for i, g in enumerate(inner_fold) if g == f]
        model = train(fit, seed, max_features)
        texts = [r["text"] for i in held for r in train_docs[i]]
        probs = numpy_probabilities(model, texts) if texts else []
        row = 0
        for i in held:
            for _ in train_docs[i]:
                for j, k in enumerate(KEYS):
                    doc_max[i][k] = max(doc_max[i][k], float(probs[row, j]))
                row += 1
    return {k: best_threshold([dm[k] for dm in doc_max], [bool(s[k]) for s in train_spans]) for k in KEYS}


def contract_counts(text: str, rows: list[dict], spans: dict, probs: list[dict], thresholds: dict) -> dict:
    """Per clause type, this contract's counts: document tp/fp/fn, sentence tp/fp/fn (main and strict), spans hit."""
    out = {}
    for k in KEYS:
        t = thresholds[k]
        flagged = [p[k] >= t for p in probs]
        c = {"doc": [0, 0, 0], "sent": [0, 0, 0], "strict": [0, 0, 0], "spans": [0, len(spans[k])],
             "max_p": max((p[k] for p in probs), default=0.0)}
        found, present = any(flagged), bool(spans[k])
        c["doc"] = [int(found and present), int(found and not present), int(present and not found)]
        for r, fl in zip(rows, flagged, strict=True):
            for name, pos in (("sent", k in r["pos"]), ("strict", k in r["strict"])):
                c[name][0] += fl and pos
                c[name][1] += fl and not pos
                c[name][2] += pos and not fl
        hit_rows = [(r["start"], r["end"]) for r, fl in zip(rows, flagged, strict=True) if fl]
        c["spans"][0] = sum(any(_touches(text, sp, h) for h in hit_rows) for sp in spans[k])
        out[k] = c
    return out


def summarise(counts: list[dict], keys: list[str]) -> dict:
    """Per type and averaged metrics from per-contract counts."""
    per = {}
    for k in keys:
        tot = {n: [sum(c[k][n][i] for c in counts) for i in range(3)] for n in ("doc", "sent", "strict")}
        hit, all_spans = sum(c[k]["spans"][0] for c in counts), sum(c[k]["spans"][1] for c in counts)
        per[k] = {"document": prf(*tot["doc"]), "sentence": prf(*tot["sent"]),
                  "sentence_strict": prf(*tot["strict"]),
                  "span_recall": round(hit / all_spans, 4) if all_spans else 0.0, "spans": all_spans,
                  "contracts_with_type": tot["doc"][0] + tot["doc"][2]}
    avg = {}
    for level in ("document", "sentence", "sentence_strict"):
        tp, fp, fn = (sum(per[k][level][n] for k in keys) for n in ("tp", "fp", "fn"))
        avg[level] = {"micro": prf(tp, fp, fn),
                      "macro_f1": round(sum(per[k][level]["f1"] for k in keys) / len(keys), 4) if keys else 0.0}
    return {"per_type": per, "average": avg}


def bootstrap(counts: list[dict], keys: list[str], seed: int, n: int = BOOTSTRAP) -> dict:
    """95% intervals for the micro averages, resampling contracts with replacement."""
    import numpy as np

    levels = ("doc", "sent", "strict")
    a = np.array([[[sum(c[k][lv][i] for k in keys) for i in range(3)] for lv in levels] for c in counts], dtype=float)
    rng = np.random.default_rng(seed)
    stats = {lv: {"precision": [], "recall": [], "f1": []} for lv in levels}
    for _ in range(n):
        s = a[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        for li, lv in enumerate(levels):
            tp, fp, fn = s[li]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            stats[lv]["precision"].append(p)
            stats[lv]["recall"].append(r)
            stats[lv]["f1"].append(2 * p * r / (p + r) if p + r else 0.0)
    names = {"doc": "document", "sent": "sentence", "strict": "sentence_strict"}
    return {names[lv]: {m: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                        for m, v in stats[lv].items()} for lv in levels}


# ---------- baseline: the best single term, learned per fold ----------

def best_terms(train_docs: list[list[dict]], train_spans: list[dict], max_features: int):
    """Per clause type, the one word or word pair (from the training vocabulary) whose rule "a contract containing
    it has the type" gives the best document-level F1 on the training contracts; ties go to the first term in the
    vocabulary's order. Returns (vectorizer, {key: (term index, term)})."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    v = RECIPE["vectorizer"]
    vec = CountVectorizer(ngram_range=tuple(v["ngram_range"]), min_df=v["min_df"], lowercase=v["lowercase"],
                          max_features=max_features, binary=True, dtype=np.int64)
    rows = [r["text"] for d in train_docs for r in d]
    x = vec.fit_transform(rows)
    owner = np.repeat(np.arange(len(train_docs)), [len(d) for d in train_docs])
    to_doc = csr_matrix((np.ones(len(rows), dtype=np.int64), (owner, np.arange(len(rows)))),
                        shape=(len(train_docs), len(rows)))
    present = ((to_doc @ x) > 0).astype(np.int64)  # contracts x terms
    terms = vec.get_feature_names_out()
    out = {}
    for k in KEYS:
        y = np.array([bool(sp[k]) for sp in train_spans], dtype=np.int64)
        tp = np.asarray(present.T @ y).ravel()
        fp = np.asarray(present.sum(axis=0)).ravel() - tp
        fn = y.sum() - tp
        f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        i = int(np.argmax(f1))
        out[k] = (i, str(terms[i]))
    return vec, out


def baseline_flags(vec, chosen: dict, rows: list[dict]) -> list[dict[str, float]]:
    """1.0 for a sentence that contains the type's chosen term, else 0.0 (scored with threshold 0.5)."""
    x = vec.transform([r["text"] for r in rows]).tocsc()
    cols = {k: set(x[:, i].nonzero()[0].tolist()) for k, (i, _) in chosen.items()}
    return [{k: float(n in cols[k]) for k in KEYS} for n in range(len(rows))]


# ---------- the run ----------

def model_bytes(model: dict) -> bytes:
    """The model file, byte for byte reproducible (fixed key order, no gzip time stamp)."""
    return gzip.compress(json.dumps(model, separators=(",", ":")).encode("utf-8"), mtime=0)


def code_hashes() -> dict[str, str]:
    return {f: sha256_file(ROOT / f) for f in CODE_FILES}


def run(cuad: dict, cuad_sha: str, args) -> dict:
    t0 = time.monotonic()
    docs = contracts(cuad)
    labelled = [label(d) for d in docs]
    group_of = groups([d["text"] for d in docs])
    fold = assign_folds(group_of, args.folds, args.seed)
    print(f"{len(docs)} contracts, {sum(map(len, labelled))} sentences, {len(set(group_of))} groups "
          f"({time.monotonic() - t0:.0f} s)", file=sys.stderr)

    oof_max = [dict.fromkeys(KEYS, 0.0) for _ in docs]
    counts: list[dict | None] = [None] * len(docs)
    base_counts: list[dict | None] = [None] * len(docs)
    train_counts: list[dict] = []  # diagnostic: each fold model scored on its own training contracts
    base_terms: dict[str, list[str]] = {k: [] for k in KEYS}
    fold_thresholds = {}
    folds = range(args.folds) if args.only_fold is None else [args.only_fold]
    for f in folds:
        tr = [i for i in range(len(docs)) if fold[i] != f]
        te = [i for i in range(len(docs)) if fold[i] == f]
        inner = assign_folds([group_of[i] for i in tr], args.inner_folds, args.seed + 1 + f)
        thr = oof_thresholds([labelled[i] for i in tr], [docs[i]["spans"] for i in tr], inner, args.inner_folds,
                             args.seed, args.max_features)
        model = train([labelled[i] for i in tr], args.seed, args.max_features)
        for k in KEYS:
            model["clauses"][k]["threshold"] = thr[k]
        fold_thresholds[f] = thr
        train_counts += training_scores(model, [docs[i] for i in tr], [labelled[i] for i in tr], thr)
        bvec, chosen = best_terms([labelled[i] for i in tr], [docs[i]["spans"] for i in tr], args.max_features)
        for k in KEYS:
            base_terms[k].append(chosen[k][1])
        half = dict.fromkeys(KEYS, 0.5)
        for i in te:
            base_counts[i] = contract_counts(docs[i]["text"], labelled[i], docs[i]["spans"],
                                             baseline_flags(bvec, chosen, labelled[i]), half)
        for i in te:
            probs = sentence_model_probabilities(model, [r["text"] for r in labelled[i]])
            counts[i] = contract_counts(docs[i]["text"], labelled[i], docs[i]["spans"], probs, thr)
            for k in KEYS:
                oof_max[i][k] = counts[i][k]["max_p"]
        print(f"fold {f}: {len(tr)} training and {len(te)} held-out contracts ({time.monotonic() - t0:.0f} s)",
              file=sys.stderr)

    scored = [i for i in range(len(docs)) if counts[i] is not None]
    cs = [counts[i] for i in scored]
    all36 = summarise(cs, KEYS)
    off = sorted(k for k in KEYS if all36["per_type"][k]["document"]["f1"] < SWITCH_OFF_F1)
    on = [k for k in KEYS if k not in off]
    result = {
        "cuad": {"release": "CUAD v1, https://doi.org/10.5281/zenodo.4595826", "sha256": cuad_sha,
                 "contracts": len(docs), "sentences": sum(map(len, labelled))},
        "seed": args.seed, "folds": args.folds, "inner_folds": args.inner_folds,
        "recipe": {**RECIPE, "vectorizer": {**RECIPE["vectorizer"], "max_features": args.max_features}},
        "match_rules": MATCH_RULES, "switch_off_rule": SWITCH_OFF_RULE,
        "near_duplicate_groups": {"groups": len(set(group_of)),
                                  "largest": max(group_of.count(g) for g in set(group_of)),
                                  "contracts_in_shared_groups": sum(group_of.count(g) > 1 for g in group_of)},
        "fold_of_contract": {docs[i]["title"]: fold[i] for i in range(len(docs))},
        "fold_thresholds": {str(f): t for f, t in fold_thresholds.items()},
        "scored_contracts": len(scored),
        "switched_off": off,
        "types_on": summarise(cs, on)["average"],
        "types_on_intervals": bootstrap(cs, on, args.seed) if on else {},
        "all_36": all36["average"],
        "all_36_intervals": bootstrap(cs, KEYS, args.seed),
        "baseline_best_term": _baseline(base_counts, scored, on, base_terms, args.seed),
        "diagnostic": diagnostic(train_counts, cs, docs, labelled),
        "per_type": all36["per_type"],
        "spans_touching_no_scored_sentence": _spans_missed(docs, labelled),
        "versions": _versions(),
        "seconds": round(time.monotonic() - t0),
    }
    if args.write:
        final = train(labelled, args.seed, args.max_features)
        final["source"] = "CUAD v1, all 510 contracts (The Atticus Project, CC BY 4.0)"
        for k in KEYS:
            final["clauses"][k]["threshold"] = best_threshold([m[k] for m in oof_max],
                                                              [bool(d["spans"][k]) for d in docs])
            final["clauses"][k]["use"] = "off" if k in off else "model"
        raw = model_bytes(final)
        result["model"] = {"sha256": hashlib.sha256(raw).hexdigest(), "vocabulary": len(final["vocabulary"]),
                           "thresholds": {k: final["clauses"][k]["threshold"] for k in KEYS},
                           "use": {k: final["clauses"][k]["use"] for k in KEYS}}
        result["_final_model"] = final
        result["_raw"] = raw
    return result


# ---------- diagnostic (does not change any reported figure) ----------

FIT_RULE = ("document-level micro F1 over all 36 types, training contracts vs held-out contracts (each fold's "
            "model scored on its own training contracts with its own thresholds): overfits when training minus "
            "held-out is 0.10 or more; underfits when the gap is under 0.10 and training F1 is under 0.90; "
            "otherwise neither")


def training_scores(model: dict, tdocs: list[dict], trows: list[list[dict]], thr: dict) -> list[dict]:
    """Per training contract counts, scored with scikit-learn's TF-IDF and the fold model's rounded weights."""
    out = []
    for d, rows in zip(tdocs, trows, strict=True):
        if not rows:
            out.append(contract_counts(d["text"], rows, d["spans"], [], thr))
            continue
        p = numpy_probabilities(model, [r["text"] for r in rows])
        probs = [{k: float(p[n, j]) for j, k in enumerate(KEYS)} for n in range(len(rows))]
        out.append(contract_counts(d["text"], rows, d["spans"], probs, thr))
    return out


def ceiling(docs: list[dict], labelled: list[list[dict]]) -> dict:
    """How well sentences can line up with the lawyers' spans, whatever the model does.

    Under the main rule a sentence is positive exactly when it touches a span, so a perfect sentence classifier
    scores 1.0 by construction. The limits come from the sentence units themselves: spans that touch no scored
    sentence can never be found (span-recall ceiling), and a perfect classifier that flags every touching sentence
    still quotes text outside the spans (character precision) and misses marked text in dropped heading lines
    (character recall)."""
    per = {}
    for k in KEYS:
        spans_all = spans_hit = marked = marked_in = flagged_chars = flagged_marked = 0
        for d, rows in zip(docs, labelled, strict=True):
            text, sp = d["text"], d["spans"][k]
            if not sp:
                continue
            mask = bytearray(len(text))
            for a, b in sp:
                mask[a:b] = b"\x01" * (b - a)
            pos = sorted({i for a, b in sp for i in range(a, b) if not text[i].isspace()})
            marked += len(pos)
            inside = bytearray(len(text))
            for r in rows:
                if k in r["pos"]:
                    inside[r["start"]:r["end"]] = b"\x01" * (r["end"] - r["start"])
            for r in rows:
                if k in r["pos"]:
                    seg = text[r["start"]:r["end"]]
                    flagged_chars += sum(not c.isspace() for c in seg)
                    flagged_marked += sum(1 for n, c in enumerate(seg) if not c.isspace() and mask[r["start"] + n])
            marked_in += sum(1 for i in pos if inside[i])
            spans_all += len(sp)
            spans_hit += sum(any(_touches(text, s, (r["start"], r["end"])) for r in rows) for s in sp)
        p = flagged_marked / flagged_chars if flagged_chars else 0.0
        rc = marked_in / marked if marked else 0.0
        per[k] = {"span_recall_ceiling": round(spans_hit / spans_all, 4) if spans_all else 0.0,
                  "char_precision": round(p, 4), "char_recall": round(rc, 4),
                  "char_f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
                  "_n": (spans_all, spans_hit, marked, marked_in, flagged_chars, flagged_marked)}
    tot = [sum(per[k]["_n"][i] for k in KEYS) for i in range(6)]
    for k in KEYS:
        del per[k]["_n"]
    p = tot[5] / tot[4] if tot[4] else 0.0
    rc = tot[3] / tot[2] if tot[2] else 0.0
    return {"sentence_f1_ceiling_main_rule": 1.0,
            "span_recall_ceiling": round(tot[1] / tot[0], 4) if tot[0] else 0.0,
            "char_precision": round(p, 4), "char_recall": round(rc, 4),
            "char_f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0, "per_type": per}


def diagnostic(train_counts: list[dict], held_counts: list[dict], docs: list[dict], labelled: list[list[dict]]) -> dict:
    tr, he = summarise(train_counts, KEYS), summarise(held_counts, KEYS)
    order = sorted(KEYS, key=lambda k: he["per_type"][k]["document"]["f1"])
    gap = round(tr["average"]["document"]["micro"]["f1"] - he["average"]["document"]["micro"]["f1"], 4)
    train_f1 = tr["average"]["document"]["micro"]["f1"]
    verdict = "overfits" if gap >= 0.10 else ("underfits" if train_f1 < 0.90 else "neither")

    def pick(keys: list[str]) -> dict:
        return {k: {lv: {"training_f1": tr["per_type"][k][lv]["f1"], "held_out_f1": he["per_type"][k][lv]["f1"]}
                    for lv in ("document", "sentence")} for k in keys}

    return {"fit_rule": FIT_RULE, "verdict": verdict, "document_gap": gap,
            "all_36": {lv: {"training": tr["average"][lv], "held_out": he["average"][lv]}
                       for lv in ("document", "sentence")},
            "weakest_5": pick(order[:5]), "strongest_5": pick(order[-5:]),
            "per_type_training": {k: {lv: tr["per_type"][k][lv]["f1"] for lv in ("document", "sentence")}
                                  for k in KEYS},
            "sentence_ceiling": ceiling(docs, labelled)}


def _baseline(base_counts: list, scored: list[int], on: list[str], terms: dict, seed: int) -> dict:
    bc = [base_counts[i] for i in scored]
    b36 = summarise(bc, KEYS)
    return {"rule": "per clause type, the one word or word pair whose presence best predicts the type at contract "
                    "level (F1) on the training contracts of each fold; a sentence is flagged when it contains it",
            "all_36": b36["average"], "all_36_intervals": bootstrap(bc, KEYS, seed),
            "types_on": summarise(bc, on)["average"],
            "per_type": {k: {"document_f1": b36["per_type"][k]["document"]["f1"],
                             "sentence_f1": b36["per_type"][k]["sentence"]["f1"], "terms_by_fold": terms[k]}
                         for k in KEYS}}


def _spans_missed(docs: list[dict], labelled: list[list[dict]]) -> int:
    n = 0
    for d, rows in zip(docs, labelled, strict=True):
        for spans in d["spans"].values():
            n += sum(not any(_touches(d["text"], sp, (r["start"], r["end"])) for r in rows) for sp in spans)
    return n


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("sklearn", "numpy", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out


def write(result: dict, out_dir: Path) -> None:
    final, raw = result.pop("_final_model"), result.pop("_raw")
    MODEL_FILE.write_bytes(raw)
    src = MODEL_PY.read_text()
    MODEL_PY.write_text(re.sub(r'MODEL_SHA256 = "[0-9a-f]{64}"', f'MODEL_SHA256 = "{result["model"]["sha256"]}"', src))
    texts = [row["text"] for row in json.loads(GOLDEN.read_text())["sentences"]]
    probs = numpy_probabilities(final, texts)
    GOLDEN.write_text(json.dumps({"note": "probabilities from scikit-learn's TF-IDF with the released model's "
                                          "vocabulary, IDF and weights; written by scripts/evaluate_cuad.py",
                                  "sentences": [{"text": t, "probabilities": {k: float(probs[i, j])
                                                                              for j, k in enumerate(KEYS)}}
                                                for i, t in enumerate(texts)]}, indent=1) + "\n")
    result["code_sha256"] = code_hashes()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out_dir / "results.md").write_text(markdown(result))


def markdown(r: dict) -> str:
    def ci(iv: dict, level: str, m: str) -> str:
        lo_hi = iv.get(level, {}).get(m)
        return f" ({lo_hi[0]:.3f}–{lo_hi[1]:.3f})" if lo_hi else ""

    def table(avg: dict, iv: dict) -> str:
        out = ["| Level | Precision | Recall | F1 | Macro F1 |", "| --- | --- | --- | --- | ---: |"]
        for level, name in (("document", "Document"), ("sentence", "Sentence"),
                            ("sentence_strict", "Sentence (strict rule)")):
            m = avg[level]["micro"]
            out.append(f"| {name} | {m['precision']:.3f}{ci(iv, level, 'precision')} | {m['recall']:.3f}"
                       f"{ci(iv, level, 'recall')} | {m['f1']:.3f}{ci(iv, level, 'f1')} | "
                       f"{avg[level]['macro_f1']:.3f} |")
        return "\n".join(out)

    rows = ["| Clause type | Contracts with it | Document P / R / F1 | Sentence P / R / F1 | Span recall | Used |",
            "| --- | ---: | --- | --- | ---: | --- |"]
    for k, (lbl, _) in CLAUSE_TYPES.items():
        p = r["per_type"][k]
        d, s = p["document"], p["sentence"]
        rows.append(f"| {lbl} | {p['contracts_with_type']} | {d['precision']:.2f} / {d['recall']:.2f} / "
                    f"{d['f1']:.2f} | {s['precision']:.2f} / {s['recall']:.2f} / {s['f1']:.2f} | "
                    f"{p['span_recall']:.2f} | {'off' if k in r['switched_off'] else 'yes'} |")
    rules = "\n".join(f"- {k}: {v}" for k, v in r["match_rules"].items())
    return f"""# Clause model: cross-validated results on CUAD v1

Written by `scripts/evaluate_cuad.py` (seed {r['seed']}); the numbers are in `results.json`. Do not edit by hand.

{r['folds']}-fold cross-validation by contract over {r['cuad']['contracts']} CUAD v1 contracts
({r['cuad']['sentences']} sentences). Each fold's model was trained only on the other folds' contracts, with its
thresholds chosen inside those training contracts, and scored on the held-out contracts. Near-duplicate contracts
were kept in one fold ({r['near_duplicate_groups']['contracts_in_shared_groups']} contracts share a group). The model
that ships was trained the same way on all {r['cuad']['contracts']} contracts, with its thresholds from these
cross-validated scores; these results are the estimate for it.

Micro averages with 95% bootstrap intervals (resampling contracts) in brackets; macro F1 is the mean of the per-type
F1.

## All 36 clause types

{table(r['all_36'], r['all_36_intervals'])}

## The {len(KEYS) - len(r['switched_off'])} types the model is used for

These {len(KEYS) - len(r['switched_off'])} types were chosen using these same results, so their averages are
somewhat optimistic; the all-36 figures above are not affected by that choice.

{table(r['types_on'], r['types_on_intervals'])}

## Fit diagnostic

Verdict: {r['diagnostic']['verdict']}, by this rule: {r['diagnostic']['fit_rule']}.

| Level (all 36 types, micro F1) | Training contracts | Held-out contracts |
| --- | ---: | ---: |
| Document | {r['diagnostic']['all_36']['document']['training']['micro']['f1']:.3f} | {r['diagnostic']['all_36']['document']['held_out']['micro']['f1']:.3f} |
| Sentence | {r['diagnostic']['all_36']['sentence']['training']['micro']['f1']:.3f} | {r['diagnostic']['all_36']['sentence']['held_out']['micro']['f1']:.3f} |

Sentence units against the lawyers' spans, for a perfect sentence classifier: span recall at most
{r['diagnostic']['sentence_ceiling']['span_recall_ceiling']:.3f}, character precision
{r['diagnostic']['sentence_ceiling']['char_precision']:.3f}, character recall
{r['diagnostic']['sentence_ceiling']['char_recall']:.3f}.

## Baseline on the same folds

Rule: {r['baseline_best_term']['rule']}. Over all 36 types: document micro F1
{r['baseline_best_term']['all_36']['document']['micro']['f1']:.3f}{ci(r['baseline_best_term']['all_36_intervals'], 'document', 'f1')},
sentence micro F1
{r['baseline_best_term']['all_36']['sentence']['micro']['f1']:.3f}{ci(r['baseline_best_term']['all_36_intervals'], 'sentence', 'f1')}.

## Switched off

{r['switch_off_rule']}

Switched off: {', '.join(CLAUSE_TYPES[k][0] for k in r['switched_off']) or 'none'}.

## Match rules

{rules}

## Per clause type

{chr(10).join(rows)}
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cuad-zip", type=Path, help="an existing CUAD_v1.zip (checked); default: download to .cache/")
    ap.add_argument("--cuad-json", type=Path, help="a CUAD-format JSON file instead of the release (tests only)")
    ap.add_argument("--only-fold", type=int, help="run one outer fold only, write nothing (time and memory check)")
    ap.add_argument("--write", action="store_true", help="train the final model and write it and the results")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-features", type=int, default=RECIPE["vectorizer"]["max_features"])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.write and args.only_fold is not None:
        ap.error("--write needs every fold")
    if args.cuad_json:
        cuad, sha = json.loads(args.cuad_json.read_text()), sha256_file(args.cuad_json)
    else:
        cuad, sha = get_cuad(args.cuad_zip)
    result = run(cuad, sha, args)
    if args.write:
        write(result, args.out_dir)
        print(f"wrote {args.out_dir / 'results.json'}; model sha256 {result['model']['sha256']}", file=sys.stderr)
    else:
        summary = {k: result[k] for k in ("scored_contracts", "switched_off", "types_on", "seconds")}
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
