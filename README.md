# contract-key-terms-mcp

A small, read-only [MCP](https://modelcontextprotocol.io) server that finds contract clause types in plain-text
agreements and shows why. It wraps a local sentence model trained on CUAD v1, the public contract dataset
labelled by lawyers at The Atticus Project. The model covers CUAD's 36 clause types (assignment, governing law,
termination for convenience, exclusivity, and so on). Everything runs on your machine, and the server makes no
network calls.

Built by Wilshem Pennick, It Just Gets Better, L.L.C.

![Demo: listing clause types, extracting key terms from a sample contract, and explaining one prediction](docs/demo.gif)

*The GIF is rendered from the actual output of `scripts/demo.py` (an MCP client calling three of the tools on a
bundled sample), drawn frame by frame with Pillow by `scripts/render_demo_gif.py`. It is not a screen recording.*

> Not legal advice. Each result quotes what an agreement says and where. It does not say what a clause means,
> whether it is enforceable, or what follows when a clause is not found.

## Tools

| Tool | What it returns |
| --- | --- |
| `list_clause_types()` | The 36 CUAD clause types: key, plain label, CUAD category name, the model's threshold, and whether the model is used for that type |
| `extract_key_terms(path)` | For each clause type, whether the model found it and up to two sentences quoted as written, best first, each with its character offset and score |
| `explain_prediction(path, clause_type)` | The clause type's best sentences in that file, the words and word pairs that raised (and lowered) each score most, and the clause type's highest-weighted terms overall |
| `search_clauses(question, k=5)` | The passages in the allowed folder's files that best match a question by keyword search (BM25), best first, each with its file, character offset and score (see [Search](#search)) |

Six of the 36 types are switched off by the evaluation's rule (see [How well it works](#how-well-it-works)).
`list_clause_types` marks them `model_used: false`, and `extract_key_terms` does not report them.

## How the model works

Each sentence becomes word and word-pair TF-IDF features. Then one logistic regression per clause type gives a
probability. A sentence at or above that type's threshold is reported. `explain_prediction` shows the arithmetic:
each term's contribution is its weight times its TF-IDF value, and the contributions plus the intercept give the
score. The model is read with the Python standard library alone, and its SHA-256 is checked before it loads.

The model holds no CUAD sentence or passage. It holds a vocabulary of 30,000 lower-case words and two-word pairs
that each appear in at least two CUAD sentences (these can include names and numbers from the contracts), how
common each is across those sentences, and, per clause type, weights learned from which sentences the dataset's
lawyers labelled with that type; the labels for document name, parties and dates were not used.

## How well it works

Measured by 5-fold cross-validation on CUAD v1's 510 contracts, split by contract, never by sentence. Each
fold's model was trained on 408 contracts, with its thresholds chosen from those training contracts only, and
then scored on the 102 contracts it had not seen. Near-duplicate contracts were kept in the same fold. The model
that ships was trained the same way on all 510 contracts, with its thresholds from these cross-validated scores,
so these cross-validated figures are the estimate for it. Ranges are 95% bootstrap intervals, resampling contracts.

All 36 clause types (micro averages):

| Level | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Contract: does the contract contain the clause type? | 0.75 (0.74–0.77) | 0.85 (0.84–0.86) | 0.80 (0.79–0.81) |
| Sentence: is the sentence part of a clause the lawyers marked? | 0.55 (0.54–0.56) | 0.56 (0.54–0.58) | 0.56 (0.55–0.57) |

For comparison, a simple baseline on the same folds: for each clause type, the one word or word pair whose
presence best predicted the type in the training contracts (for governing law, "laws"), flagging any sentence
that contains it. Over all 36 types it scores contract-level F1 0.69 (0.68–0.70) and sentence-level F1 0.28
(0.27–0.29).

Six types score below the switch-off rule (below) and are not reported by the server. Over the 30 types it does
report, the figures are higher: contract-level F1 0.82 (0.82–0.83) and sentence-level F1 0.57 (0.56–0.58). Those
30 were chosen using these same cross-validation results, so their figures are somewhat optimistic; the all-36
figures above are not affected by that choice.

How a sentence is matched: sentences are split the way the server splits them. A sentence counts as part of a
clause when it overlaps a passage the CUAD lawyers marked for that type by at least one non-space character. A
stricter rule, at least half of the sentence inside a marked passage, gives the same sentence F1 (0.56). A
sentence is flagged when its score reaches the type's threshold, and a contract counts as found when at least
one sentence is flagged. Results for every type, and how they were computed:
[docs/evaluation/results.md](docs/evaluation/results.md). To reproduce them:
`uv run --group eval python scripts/evaluate_cuad.py --write` (this script downloads CUAD; the server never uses
the network).

Switch-off rule, fixed before the evaluation was run: a clause type is switched off (the model is not used for
it, and extract_key_terms does not report it) when its cross-validated document-level F1, pooled over the five
held-out folds, is below 0.50. Switched-off types are still scored and listed separately in the results, outside
the averages.

The weakest types, all switched off by that rule:

| Clause type | Contract-level F1 | Contracts with it (of 510) | What the figures show |
| --- | --- | --- | --- |
| Price limits or changes need agreement | 0.20 | 15 | Few examples (15), and most contracts it flagged did not have one (precision 0.14) |
| Most favored nation (best terms) | 0.40 | 28 | Most contracts it flagged did not have one (precision 0.28) |
| Unlimited use | 0.44 | 17 | Few examples (17), and most contracts it flagged did not have one (precision 0.32) |
| Third parties may enforce it | 0.44 | 32 | It missed more than half of the contracts that have one, and more than half it flagged did not have one (recall 0.44, precision 0.44) |
| Non-disparagement | 0.45 | 38 | Most contracts it flagged did not have one (precision 0.38) |

"Exception to a restriction" (0.45, 76 contracts) is also switched off.

What the figures don't show: the CUAD lawyers may not have marked every sentence that belongs to a clause, so a
flagged sentence they left unmarked counts against precision. The figures are for U.S. commercial contracts
filed with the SEC. The example scores below come from a contract that is part of CUAD, which the model saw in
training, so they are not evidence of accuracy.


## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```sh
git clone https://github.com/wilshem/contract-key-terms-mcp
cd contract-key-terms-mcp
uv sync
uv run pytest -q          # optional: run the tests
```

The server reads files from one folder only, set when it starts. By default that's the three bundled
sample contracts. Pass `--allow /path/to/folder` (or set `KEY_TERMS_ALLOWED_DIR`) to point it at your own
`.txt` or `.md` files.

To keep folders out even when they sit inside the allowed folder, list them in `KEY_TERMS_DENY_DIRS`, and list
folder names to skip wherever they appear in `KEY_TERMS_DENY_NAMES`. Separate entries with `:` on Linux and
macOS, or `;` on Windows. Common key and credential folders (`~/.ssh`, `~/.aws` and similar) are always refused.
For example:

```sh
KEY_TERMS_DENY_DIRS="~/contracts/private:~/contracts/archive" KEY_TERMS_DENY_NAMES="drafts" \
  uv run contract-key-terms-mcp --allow ~/contracts
```

### Claude Code

```sh
claude mcp add contract-key-terms -- uv run --directory /path/to/contract-key-terms-mcp contract-key-terms-mcp
# or, to read your own folder:
claude mcp add contract-key-terms -- uv run --directory /path/to/contract-key-terms-mcp contract-key-terms-mcp --allow /path/to/contracts
# with deny settings (claude mcp add -e sets the server's environment):
claude mcp add contract-key-terms -e KEY_TERMS_DENY_NAMES=drafts -- uv run --directory /path/to/contract-key-terms-mcp contract-key-terms-mcp --allow /path/to/contracts
```

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings, Developer, Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "contract-key-terms": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/contract-key-terms-mcp", "contract-key-terms-mcp"]
    }
  }
}
```

On Windows, use a Windows path such as `C:\\path\\to\\contract-key-terms-mcp`. If the code lives in WSL, use
`"command": "wsl.exe"` with `"args": ["uv", "run", "--directory", "/path/to/contract-key-terms-mcp",
"contract-key-terms-mcp"]`. If `uv` is not on the WSL path, give its full path instead.

### Search

`search_clauses` splits each file into passages of about 256 words, built from whole sentences where it can (a
longer sentence is cut into 256-word pieces, and a section heading is kept with the text that follows it, which can
take a passage a little over 256 words), and ranks them against your question by keyword search (BM25). It needs
nothing beyond the plain install. Passages are kept in memory only.

### Usage log

The server writes nothing unless you turn its usage log on, with `--log` or `KEY_TERMS_LOG=on`. Each tool call
then adds one line to `~/.contract-key-terms-mcp/usage.jsonl` (or the file in `KEY_TERMS_LOG_FILE`): the time,
the tool's name, how long it took, whether it succeeded, and the error's type if it did not. It never records
document text, questions, paths or other arguments.

## Retrieval results

Keyword search is the shipped method. Vector search was evaluated but is not shipped, because its runtime library
(onnxruntime) tries to make network calls that its documented switch-off does not stop on Linux, which a network
trace we ran in November 2026 showed (name lookups attempted, none could be sent, inside a sandbox with no network).

Both were measured on CUAD v1 with `scripts/evaluate_retrieval.py` (full tables in
[docs/evaluation/retrieval_results.md](docs/evaluation/retrieval_results.md)). That script and the vector search
code are not in this release; they are kept in the tag
[`retrieval-eval-2026-11`](../../tree/retrieval-eval-2026-11) so these figures can be reproduced. Its scripts load
onnxruntime, which tries to make network calls, so they must be run with networking cut off (for example inside
`bwrap --unshare-net`). 102 contracts were used only to choose
the passage size (256 words); the figures below are on the other 408. For each contract and each clause type it
contains, the question was asked of that contract's passages alone. A passage counts as relevant when it overlaps
text the lawyers marked for that type. recall@k is the share of the 3,486 questions with a relevant passage in the
top k.

| Method | Question asked | recall@1 | recall@3 | recall@5 |
| --- | --- | ---: | ---: | ---: |
| Vector | CUAD's question for the type | 0.357 | 0.574 | 0.682 |
| Vector | The type's name | 0.446 | 0.657 | 0.750 |
| Keyword | CUAD's question for the type | 0.426 | 0.661 | 0.754 |
| Keyword | The type's name | 0.497 | 0.691 | 0.768 |

Within one contract, keyword search found a relevant passage more often than vector search, with both kinds of
question. Over all 408 contracts at once (36 questions, one per type), the picture was mixed: with CUAD's questions
vector search did better at recall@1 (0.528 against 0.417), was level at recall@3 and behind at recall@5; with the
type's name it was ahead at all three. 36 questions is too few to rank the two methods. 190 of the 13,443 passages
(1.4%) were longer than the 512 tokens the embedding model reads, so vector search saw only their start.

## Example conversation

The tool results below are real output on the bundled `services_agreement.txt`. They're shortened, and the
quotes are cut at about 110 characters. The assistant's wording is illustrative.

> **You:** Which key terms are in services_agreement.txt?
>
> *Tool call:* `extract_key_terms(path="services_agreement.txt")`
>
> *Result (clauses found):*
> - `anti_assignment` (score 0.9983): "No Party shall have the right to assign its rights or obligations under this
>   Agreement without the consent of ..."
> - `termination_for_convenience` (0.9948): "This Agreement may be terminated by (a) the written agreement of the
>   Parties or (b) by either Party upon 5 day..."
> - `insurance` (0.9781): "(b) the cost of employee benefits for the Services Employees, including 401(k), pension,
>   bonuses and health in..."
> - `governing_law` (0.9997): "This Agreement shall be subject to and governed by the laws of the State of
>   Colorado, excluding any conflicts-..."
>
> **Claude:** The model flagged four clause types: an anti-assignment clause, a termination right that needs no
> breach, a governing-law clause (Colorado), and insurance. The insurance sentence is about the cost of employee
> benefits, including health insurance, not a duty to carry insurance, so check it yourself. Here are the quoted
> sentences ...
>
> **You:** Why did it flag termination for convenience?
>
> *Tool call:* `explain_prediction(path="services_agreement.txt", clause_type="termination_for_convenience")`
>
> *Result:* probability 0.9948 against a threshold of 0.97. Top contributions: "be terminated" (+1.41),
> "terminated" (+1.23), "days written" (+0.98), "notice" (+0.96), "written notice" (+0.93).
>
> **Claude:** The sentence scored 0.9948, above the 0.97 threshold. The terms that pushed it up most were
> "be terminated", "terminated" and "days written" ...

## Limits

- It reads `.txt` and `.md` only, up to 2 MB per file. It doesn't read PDFs or Word files.
- It's a word-based model. It can flag a sentence that only uses a clause's words; for example, a quality
  "minimum of six percent" can score as a minimum commitment. Read every quoted sentence yourself.
- The model was trained on U.S. commercial contracts filed with the SEC. Other kinds of documents may read
  less well.
- Contract text is returned to your assistant as data. A document can contain text written to look like
  instructions, so treat results as untrusted.

## Security

See [SECURITY.md](SECURITY.md). In short: stdio only, no network, read-only, one allowed folder,
paths resolved before they are checked, folders you list refused (compared ignoring case, on every system),
files with more than one hard link refused, and the model's checksum verified before it loads.

## Design notes

- Local and read-only: contracts are often confidential, so the server reads them on your machine, sends
  nothing anywhere and never changes your files. It writes nothing at all unless you turn its usage log on.
- Paths are resolved before they are checked: a link or `..` can make a path that looks inside the allowed
  folder lead outside it, so the check runs on where the path really leads.
- The deny list can only add refusals: settings can name more folders to refuse, but no setting can widen
  what is read beyond the one allowed folder or remove the built-in refusals.

## Licence and data

- Code: Apache License 2.0 ([LICENSE](LICENSE)).
- Model weights, clause type names and sample contracts: from CUAD v1 by The Atticus Project, Inc.,
  https://www.atticusprojectai.org/cuad, licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). The model file is shared under CC BY 4.0. CUAD is provided as
  is, without warranties; see the licence and The Atticus Project's disclaimer
  (https://www.atticusprojectai.org/legal). Full notices: [NOTICE.md](NOTICE.md); sample sources and checksums:
  [samples/SOURCE.md](src/key_terms_mcp/samples/SOURCE.md).

Citation (a courtesy): Hendrycks, Burns, Chen and Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review", NeurIPS 2021.
