# search_clauses: retrieval results on CUAD v1

Written by `scripts/evaluate_retrieval.py` (seed 20261128); the numbers are in `retrieval_results.json`.

Keyword search is the shipped method. Vector search was evaluated here but is not shipped, because its runtime
library (onnxruntime) tries to make network calls that its documented switch-off does not stop on Linux. The script
and the vector search code are not in this release; they are kept in the tag `retrieval-eval-2026-11`, whose
scripts must be run with networking cut off (for example inside `bwrap --unshare-net`).

Test set: 408 contracts (102 more were used only to choose the chunk size:
on the dev set, the chunk size (64, 128 or 256 words) with the highest mean of vector and keyword within-contract recall@3 (CUAD questions); ties go to the smaller size). Chunks of about 256 words (longest 262: a heading
kept with the text it heads can add a few); 13443 in all, mean 226.4 words. Embedding model: BAAI/bge-small-en-v1.5 at revision 5c38ec7c405e
(MIT licence). Keyword search: BM25 (k1 1.2, b 0.75).
A chunk is relevant when it overlaps a lawyer-marked span of the clause type by at least one non-space character. recall@k is the share of questions with a relevant chunk among the top k; MRR is the
mean of 1 / the rank of the first relevant chunk.

## Within one contract (headline)

Each question is asked of one contract that has the clause type, over that contract's chunks.

| Method | Question | recall@1 | recall@3 | recall@5 | MRR | Queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vector | CUAD question | 0.357 | 0.574 | 0.682 | 0.504 | 3486 |
| vector | type name | 0.446 | 0.657 | 0.750 | 0.581 | 3486 |
| keyword | CUAD question | 0.426 | 0.661 | 0.754 | 0.573 | 3486 |
| keyword | type name | 0.497 | 0.691 | 0.768 | 0.619 | 3486 |

## Over the whole test set at once

Each clause type's question is asked once, over every test contract's chunks together.

| Method | Question | recall@1 | recall@3 | recall@5 | MRR | precision@5 | Queries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | CUAD question | 0.528 | 0.750 | 0.778 | 0.654 | 0.522 | 36 |
| vector | type name | 0.583 | 0.889 | 0.889 | 0.729 | 0.589 | 36 |
| keyword | CUAD question | 0.417 | 0.750 | 0.833 | 0.604 | 0.450 | 36 |
| keyword | type name | 0.500 | 0.806 | 0.833 | 0.653 | 0.517 | 36 |
