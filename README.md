# contract-key-terms-mcp

A small, read-only [MCP](https://modelcontextprotocol.io) server that finds contract clause types in plain-text
agreements and shows why. It wraps a local sentence model trained on CUAD v1, the public contract dataset
labelled by lawyers at The Atticus Project. The model covers CUAD's 36 clause types (assignment, governing law,
termination for convenience, exclusivity, and so on). Everything runs on your machine, and the server makes no
network calls.

![Demo: listing clause types, extracting key terms from a sample contract, and explaining one prediction](docs/demo.gif)

*The GIF is rendered from the actual output of `scripts/demo.py` (an MCP client calling the three tools on a
bundled sample), drawn frame by frame with Pillow by `scripts/render_demo_gif.py`. It is not a screen recording.*

> Not legal advice. Each result quotes what an agreement says and where. It does not say what a clause means,
> whether it is enforceable, or what follows when a clause is not found.

## Tools

| Tool | What it returns |
| --- | --- |
| `list_clause_types()` | The 36 CUAD clause types: key, plain label, CUAD category name, the model's threshold, and whether the model is used for that type |
| `extract_key_terms(path)` | For each clause type, whether the model found it and up to two sentences quoted as written, best first, each with its character offset and score |
| `explain_prediction(path, clause_type)` | The clause type's best sentences in that file, the words and word pairs that raised (and lowered) each score most, and the clause type's highest-weighted terms overall |

For 3 of the 36 types the model's scores were too weak in training. `list_clause_types` marks them
`model_used: false`, and `extract_key_terms` does not report them.

## How the model works

Each sentence becomes word and word-pair TF-IDF features. Then one logistic regression per clause type gives a
probability. A sentence at or above that type's threshold is reported. `explain_prediction` shows the arithmetic:
each term's contribution is its weight times its TF-IDF value, and the contributions plus the intercept give the
score. The model is read with the Python standard library alone, and its SHA-256 is checked before it loads.

The model holds no CUAD sentence or passage. It holds a vocabulary of 30,000 lower-case words and two-word pairs
that each appear in at least two CUAD sentences (these can include names and numbers from the contracts), how
common each is across those sentences, and, per clause type, weights learned from which sentences the dataset's
lawyers labelled with that type; the labels for document name, parties and dates were not used.

This version reports the model alone. No accuracy figure is given for it yet.

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

## Example conversation

The tool results below are real output on the bundled `services_agreement.txt`. They're shortened, and the
quotes are cut at about 110 characters. The assistant's wording is illustrative.

> **You:** Which key terms are in services_agreement.txt?
>
> *Tool call:* `extract_key_terms(path="services_agreement.txt")`
>
> *Result (clauses found):*
> - `anti_assignment` (score 0.9997): "No Party shall have the right to assign its rights or obligations under this
>   Agreement without the consent of ..."
> - `termination_for_convenience` (0.9955): "This Agreement may be terminated by (a) the written agreement of the
>   Parties or (b) by either Party upon 5 day..."
> - `governing_law` (0.9999): "This Agreement shall be subject to and governed by the laws of the State of
>   Colorado, excluding any conflicts-..."
>
> **Claude:** The model found three clause types: an anti-assignment clause, a termination right that needs no
> breach, and a governing-law clause (Colorado). Here are the quoted sentences ...
>
> **You:** Why did it flag termination for convenience?
>
> *Tool call:* `explain_prediction(path="services_agreement.txt", clause_type="termination_for_convenience")`
>
> *Result:* probability 0.9955 against a threshold of 0.98. Top contributions: "be terminated" (+1.80),
> "terminated" (+1.32), "written notice" (+1.28), "days written" (+1.20), "party upon" (+0.76).
>
> **Claude:** The sentence scored 0.9955, above the 0.98 threshold. The terms that pushed it up most were
> "be terminated", "written notice" and "days written" ...

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
paths resolved before they are checked, folders you list refused, and the model's checksum verified before it
loads.

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
