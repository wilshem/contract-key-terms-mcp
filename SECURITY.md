# Security

This tag is an evaluation snapshot, not the release. Vector search here loads onnxruntime, which tries to make
network calls that its documented switch-off does not stop on Linux; run it, and anything else that loads
onnxruntime, only with networking cut off (for example inside `bwrap --unshare-net`). The statements below hold
for keyword search and the other three tools.

What the server does and never does:
- It runs over stdio only and opens no network connection.
- It reads only regular `.txt` and `.md` files, up to 2 MB, inside the one folder set when it starts
  (the bundled samples by default). Paths are resolved (links followed, `..` removed) before they are
  checked, and a file swapped for a link between the check and the open is refused.
- Some folders and files are never read, even inside the allowed folder: `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.config`,
  `~/.netrc`, `~/.docker` and `~/.kube` always, plus any folders you list in `KEY_TERMS_DENY_DIRS`, and any
  folder whose name you list in `KEY_TERMS_DENY_NAMES` (both separated by `:` on Linux and macOS, `;` on
  Windows). Settings can add to this list, never remove from it. They are read once at start.
- The deny check ignores case on every system, so `~/.SSH` counts as `~/.ssh`. It may also refuse a folder
  that differs from a denied one only in case.
- A file with more than one hard link is refused, because another of its names could be in a denied folder
  where no path check can see it. A genuine file that happens to have several hard links is refused too.
- It never writes, moves or deletes your files. It writes nothing at all unless you turn its usage log on
  (`--log` or `KEY_TERMS_LOG=on`); the log holds only the time, tool name, duration and success of each call,
  never document text, questions or paths, and on Linux and macOS is opened without following a link.
- `search_clauses` reads files through the same checks as the other tools; a refused file is skipped and
  counted, never named.
- It checks the SHA-256 of the clause model, and of each embedding model file, before loading it. The embedding
  model is downloaded only by `contract-key-terms-mcp-setup`, which you run yourself; the server never imports
  that command.
- Contract text it returns is quoted as data. Treat it as untrusted: a document can contain text written
  to look like instructions.

To report a security problem, open a private security advisory on this repository (GitHub, Security tab,
"Report a vulnerability"). Please do not open a public issue for it.
