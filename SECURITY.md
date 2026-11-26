# Security

What the server does and never does:
- It runs over stdio only and opens no network connection.
- It reads only regular `.txt` and `.md` files, up to 2 MB, inside the one folder set when it starts
  (the bundled samples by default). Paths are resolved (links followed, `..` removed) before they are
  checked, and a file swapped for a link between the check and the open is refused.
- It never writes, moves or deletes files.
- It checks the model file's SHA-256 before loading it.
- Contract text it returns is quoted as data. Treat it as untrusted: a document can contain text written
  to look like instructions.

To report a security problem, open a private security advisory on this repository (GitHub, Security tab,
"Report a vulnerability"). Please do not open a public issue for it.
