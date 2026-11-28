# Clause model: cross-validated results on CUAD v1

Written by `scripts/evaluate_cuad.py` (seed 20261127); the numbers are in `results.json`. Do not edit by hand.

5-fold cross-validation by contract over 510 CUAD v1 contracts
(109920 sentences). Each fold's model was trained only on the other folds' contracts, with its
thresholds chosen inside those training contracts, and scored on the held-out contracts. Near-duplicate contracts
were kept in one fold (6 contracts share a group). The model
that ships was trained the same way on all 510 contracts, with its thresholds from these
cross-validated scores; these results are the estimate for it.

Micro averages with 95% bootstrap intervals (resampling contracts) in brackets; macro F1 is the mean of the per-type
F1.

## All 36 clause types

| Level | Precision | Recall | F1 | Macro F1 |
| --- | --- | --- | --- | ---: |
| Document | 0.754 (0.742–0.767) | 0.849 (0.837–0.860) | 0.799 (0.790–0.808) | 0.709 |
| Sentence | 0.552 (0.539–0.564) | 0.561 (0.544–0.580) | 0.557 (0.545–0.569) | 0.485 |
| Sentence (strict rule) | 0.547 (0.534–0.560) | 0.570 (0.553–0.589) | 0.559 (0.547–0.571) | 0.487 |

## The 30 types the model is used for

These 30 types were chosen using these same results, so their averages are
somewhat optimistic; the all-36 figures above are not affected by that choice.

| Level | Precision | Recall | F1 | Macro F1 |
| --- | --- | --- | --- | ---: |
| Document | 0.790 (0.778–0.802) | 0.862 (0.850–0.872) | 0.824 (0.816–0.833) | 0.772 |
| Sentence | 0.574 (0.560–0.587) | 0.570 (0.553–0.589) | 0.572 (0.560–0.585) | 0.530 |
| Sentence (strict rule) | 0.569 (0.556–0.582) | 0.579 (0.561–0.598) | 0.574 (0.563–0.587) | 0.531 |

## Fit diagnostic

Verdict: overfits, by this rule: document-level micro F1 over all 36 types, training contracts vs held-out contracts (each fold's model scored on its own training contracts with its own thresholds): overfits when training minus held-out is 0.10 or more; underfits when the gap is under 0.10 and training F1 is under 0.90; otherwise neither.

| Level (all 36 types, micro F1) | Training contracts | Held-out contracts |
| --- | ---: | ---: |
| Document | 0.916 | 0.799 |
| Sentence | 0.832 | 0.557 |

Sentence units against the lawyers' spans, for a perfect sentence classifier: span recall at most
0.999, character precision
0.952, character recall
0.997.

## Baseline on the same folds

Rule: per clause type, the one word or word pair whose presence best predicts the type at contract level (F1) on the training contracts of each fold; a sentence is flagged when it contains it. Over all 36 types: document micro F1
0.686 (0.676–0.698),
sentence micro F1
0.279 (0.269–0.291).

## Switched off

A clause type is switched off (the model is not used for it, and extract_key_terms does not report it) when its cross-validated document-level F1, pooled over the five held-out folds, is below 0.50. Switched-off types are still scored and listed separately in the results, outside the averages.

Switched off: Exception to a restriction, Most favored nation (best terms), Non-disparagement, Price limits or changes need agreement, Third parties may enforce it, Unlimited use.

## Match rules

- sentence: the server's own sentence splitter; heading-only lines are dropped, as the server does
- positive: a sentence is positive for a clause type when it overlaps a lawyer-marked span of that type by at least one non-space character
- strict: a sentence is positive only when at least half of its non-space characters lie inside lawyer-marked spans of that type (reported beside the main rule)
- flagged: a sentence is flagged when its probability is at or above the clause type's threshold
- document: a contract contains a clause type when the lawyers marked at least one span of it; the model finds it when at least one sentence is flagged
- span_recall: share of lawyer-marked spans that overlap at least one flagged sentence

## Per clause type

| Clause type | Contracts with it | Document P / R / F1 | Sentence P / R / F1 | Span recall | Used |
| --- | ---: | --- | --- | ---: | --- |
| Change of control | 121 | 0.61 / 0.72 / 0.66 | 0.38 / 0.36 / 0.37 | 0.46 | yes |
| Assignment needs consent or is restricted | 374 | 0.96 / 0.95 / 0.96 | 0.68 / 0.77 / 0.72 | 0.83 | yes |
| Termination without breach | 183 | 0.70 / 0.83 / 0.76 | 0.57 / 0.61 / 0.59 | 0.71 | yes |
| Exclusivity | 180 | 0.65 / 0.84 / 0.73 | 0.38 / 0.49 / 0.43 | 0.61 | yes |
| Non-compete | 119 | 0.60 / 0.84 / 0.70 | 0.43 / 0.48 / 0.45 | 0.59 | yes |
| Exception to a restriction | 76 | 0.34 / 0.67 / 0.45 | 0.12 / 0.22 / 0.15 | 0.30 | off |
| No soliciting customers | 34 | 0.43 / 0.62 / 0.51 | 0.35 / 0.39 / 0.37 | 0.48 | yes |
| No soliciting or hiring staff | 59 | 0.87 / 0.92 / 0.89 | 0.81 / 0.71 / 0.75 | 0.80 | yes |
| Non-disparagement | 38 | 0.38 / 0.55 / 0.45 | 0.28 / 0.36 / 0.32 | 0.40 | off |
| Most favored nation (best terms) | 28 | 0.28 / 0.68 / 0.40 | 0.19 / 0.44 / 0.27 | 0.53 | off |
| Right of first refusal, offer or negotiation | 85 | 0.68 / 0.74 / 0.71 | 0.60 / 0.44 / 0.51 | 0.54 | yes |
| Volume limit | 82 | 0.45 / 0.73 / 0.56 | 0.25 / 0.36 / 0.30 | 0.39 | yes |
| Limit on liability | 275 | 0.93 / 0.94 / 0.93 | 0.77 / 0.68 / 0.72 | 0.78 | yes |
| Liability not limited | 111 | 0.67 / 0.78 / 0.72 | 0.57 / 0.46 / 0.51 | 0.63 | yes |
| Liquidated damages or set penalties | 61 | 0.68 / 0.69 / 0.68 | 0.48 / 0.54 / 0.51 | 0.67 | yes |
| Minimum purchase or commitment | 165 | 0.71 / 0.82 / 0.76 | 0.44 / 0.44 / 0.44 | 0.56 | yes |
| Price limits or changes need agreement | 15 | 0.14 / 0.33 / 0.20 | 0.10 / 0.16 / 0.12 | 0.19 | off |
| Revenue or profit sharing | 166 | 0.81 / 0.73 / 0.77 | 0.51 / 0.43 / 0.47 | 0.53 | yes |
| Ownership of work or intellectual property | 124 | 0.81 / 0.81 / 0.81 | 0.58 / 0.50 / 0.54 | 0.57 | yes |
| Jointly owned intellectual property | 46 | 0.78 / 0.83 / 0.80 | 0.54 / 0.47 / 0.50 | 0.54 | yes |
| License granted | 255 | 0.93 / 0.93 / 0.93 | 0.71 / 0.57 / 0.63 | 0.74 | yes |
| License cannot be transferred | 138 | 0.77 / 0.81 / 0.79 | 0.64 / 0.50 / 0.56 | 0.63 | yes |
| License covers the licensor's affiliates | 23 | 0.43 / 0.83 / 0.57 | 0.31 / 0.27 / 0.29 | 0.36 | yes |
| License extends to the licensee's affiliates | 59 | 0.51 / 0.61 / 0.56 | 0.39 / 0.39 / 0.39 | 0.51 | yes |
| Unlimited use | 17 | 0.32 / 0.71 / 0.44 | 0.24 / 0.36 / 0.29 | 0.50 | off |
| Perpetual or irrevocable license | 70 | 0.72 / 0.83 / 0.77 | 0.68 / 0.51 / 0.58 | 0.69 | yes |
| Source code escrow | 13 | 0.69 / 0.85 / 0.76 | 0.54 / 0.26 / 0.35 | 0.48 | yes |
| Insurance required | 166 | 0.89 / 0.95 / 0.92 | 0.71 / 0.81 / 0.76 | 0.88 | yes |
| Warranty period | 75 | 0.79 / 0.80 / 0.79 | 0.37 / 0.35 / 0.36 | 0.47 | yes |
| Records may be inspected or audited | 214 | 0.91 / 0.96 / 0.93 | 0.62 / 0.68 / 0.65 | 0.75 | yes |
| Duties after the agreement ends | 182 | 0.64 / 0.81 / 0.71 | 0.37 / 0.36 / 0.36 | 0.48 | yes |
| Third parties may enforce it | 32 | 0.44 / 0.44 / 0.44 | 0.44 / 0.43 / 0.43 | 0.46 | off |
| Agreement not to sue or challenge | 100 | 0.89 / 0.78 / 0.83 | 0.71 / 0.58 / 0.64 | 0.65 | yes |
| Governing law | 437 | 0.98 / 0.99 / 0.99 | 0.64 / 0.96 / 0.77 | 0.99 | yes |
| Renewal | 176 | 0.86 / 0.87 / 0.86 | 0.67 / 0.71 / 0.69 | 0.80 | yes |
| Notice needed to stop a renewal | 111 | 0.79 / 0.79 / 0.79 | 0.73 / 0.67 / 0.70 | 0.76 | yes |
