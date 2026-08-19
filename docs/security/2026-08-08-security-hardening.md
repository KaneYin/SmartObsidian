# Security Hardening Record — 2026-08-08

This document records the security review findings, why each issue existed, the
implemented repair, its user-visible effect, and the verification added for it.
It is both a change log and a maintenance guide for the security boundaries that
must not regress.

## Scope and threat model

Weft is currently a local command-line application, not a network service. The
most important boundary is therefore between the current user and content that
can appear inside an Obsidian Vault through sync, shared folders, archives, Git,
or another local account.

The review assumes that:

- a Vault entry, filename, tag, or note body may be less trusted than the user
  running Weft;
- Claude and its response are external to the local trust boundary;
- other accounts may exist on the same machine;
- `.weft/index.json` and `api-log.jsonl` contain sensitive plaintext;
- CLI arguments are currently local, but should remain safe if the CLI is later
  wrapped by another process.

## 1. Vault symlink read escape

### Possible problem

A Vault could contain a file named `leak.md` that was a symbolic link to any
file readable by the current user. The old recursive `*.md` scan selected the
link by its name, and `Path.read_text()` followed the link. The target did not
need a `.md` suffix.

The outside content would then be embedded and copied into `index.json`. If it
ranked for a later question, the raw chunk could also be sent to Claude.

### Why it existed

The parser treated “found below the Vault directory” as equivalent to “contained
by the Vault.” Directory traversal APIs return symlink entries, while ordinary
file reads follow them. No canonical containment or file-type check existed.

### Repair

- `security.vault_root()` resolves and validates the Vault directory.
- `read_vault_text()` rejects every symlink component below that root.
- The resolved target must remain inside the canonical Vault.
- The file is opened with `O_NOFOLLOW` where the operating system provides it,
  and its descriptor must identify a regular file.
- Privacy exclusions run before the secure read, so excluded content is never
  opened merely to decide whether it should be excluded.

### Behavior change and verification

Internal Vault symlinks are now rejected as well as escaping symlinks. This is a
deliberate fail-closed default. The regression test
`test_parse_vault_refuses_markdown_symlink_that_points_outside` creates a real
outside target and verifies that its contents are never returned.

## 2. `_inbox.md` overwrite and symlink write escape

### Possible problem

The old suggestion command always called `write_text()` on `_inbox.md`. This had
two consequences:

1. a normal existing inbox, including unchecked suggestions or user edits, was
   silently truncated on the next run; and
2. if `_inbox.md` was a symlink, its outside target was truncated and replaced
   with generated Markdown.

The second behavior allowed an untrusted Vault to overwrite any file writable by
the current user. A second run with zero new suggestions also erased the review
surface while the ledger continued remembering those suggestions as proposed.

### Why it existed

The initial M2 implementation considered `_inbox.md` generated output and did
not distinguish the path entry from its target. It also wrote an empty-state
document on every run, before accept/reject state existed.

### Repair

- `_inbox.md` is validated before an optional Claude rationale request, avoiding
  an unnecessary external call when the eventual write cannot proceed.
- Symlinks and non-regular targets are always refused.
- A regular existing inbox is preserved by default.
- `--overwrite-inbox` is required to replace an existing regular inbox.
- Zero suggestions leave the inbox untouched.
- The ledger is updated only after the inbox write succeeds.
- Generated Markdown is written as a private file and the destination is
  replaced atomically; explicit replacement replaces the directory entry rather
  than following a target.

### Behavior change and verification

Users should review, move, or delete the old inbox before generating another.
Four inbox tests cover private creation, default preservation, explicit regular
replacement, and refusal to follow a symlink. CLI tests also cover the unchanged
zero-suggestion state.

## 3. Privacy policy and API payload audit

### Possible problem

The architecture required a folder allowlist/redaction boundary and an audit of
every API payload, but the implementation indexed every Markdown file and only
logged the optional suggestion-rationale request. A private note could therefore
enter the plaintext index and later be selected for `ask` without an auditable
record of exactly what left the machine.

### Why it existed

Privacy requirements were documented before implementation. The M0 parser had
no policy object, and API logging was added inside the M2 command instead of at a
shared LLM boundary.

### Repair

Indexing now accepts an immutable `PrivacyPolicy` that is applied before parsing
or embedding:

- `Private/`, `.obsidian/`, `.trash/`, and `.weft/` are excluded by default,
  case-insensitively;
- repeatable `--include PATH` arguments form an allowlist;
- repeatable `--exclude PATH` arguments add exclusions;
- repeatable `--redact REGEX` expressions replace matches with `[REDACTED]`
  before frontmatter parsing, embeddings, and persistence;
- `--include-private` is the explicit escape hatch for the default Private
  exclusion and warns in help text that selected content may leave the machine.

Every new index has a mode-0600 `index.manifest.json` containing the canonical
Vault path and effective policy. `suggest` refuses an older unbound index and
refuses to write suggestions into a different Vault.

The supported CLI wraps both `ask` and `suggest --rationale` LLM calls in
`AuditedLLM`. It records the exact system prompt and user payload before the
network request, so failed attempts are still auditable. Logs are append-only
JSONL with mode `0600`.

### Important consequence

The API audit log intentionally duplicates sensitive outbound content. It is an
accountability mechanism, not a sanitized telemetry log. Backups, support
bundles, and bug reports must treat it like the original notes.

### Verification

Tests prove default exclusion, allowlist behavior, regular-expression redaction,
absence of private/redacted text from persisted metadata, manifest binding,
exact ask payload logging, private permissions, and refusal to append through a
log symlink.

## 4. Numeric validation and resource boundaries

### Possible problem

`argparse(type=int)` accepted negative values. NumPy/Python slicing gives
negative bounds special meaning, so `--k -1` returned almost the entire store
instead of rejecting the request. That could dramatically increase the private
context sent to Claude and increase token cost. Negative suggestion limits also
defeated the attention budget, while `nan` made every cosine comparison behave
unexpectedly.

### Why it existed

The implementation relied on types but did not validate semantic ranges. The
vector store also trusted callers, so bypassing the CLI reproduced the problem.

### Repair

- `--k` accepts only 1–50.
- `--limit` accepts only 0–100.
- `--threshold` must be finite and between -1 and 1.
- `VectorStore.search`, graph retrieval, and `infer_links` repeat the critical
  checks at the library boundary.

CLI and direct-store tests cover negative, oversized, and non-finite values.

## 5. Prompt, Markdown, and terminal injection

### Possible problem

Note bodies, headings, paths, and the question were previously concatenated into
one prose prompt. A synced note could contain instructions telling the model to
ignore the user or reveal other retrieved sources. Note paths and model-written
rationales were also interpolated into Markdown, and model answers or filenames
could contain terminal control characters.

The current agent has no external action tools, so prompt injection mainly
threatened answer integrity and cross-note disclosure. The impact would become
more serious when automatic actions are introduced.

### Why it existed

The initial prompt optimized for readability and testing, not separation of
instructions from untrusted data. The inbox assumed filenames and rationale text
were display-safe.

### Repair

- Ask prompts are structured JSON rather than free-form source concatenation.
- The system prompt explicitly defines source objects as untrusted data and
  rejects instructions embedded in them.
- Suggestion rationale inputs are structured JSON and likewise labeled
  untrusted.
- Dynamic Markdown is escaped, code-span delimiters adapt to embedded backticks,
  and suggested links use the full qualified note path rather than an ambiguous
  filename stem.
- Terminal C0/C1 controls are removed from model answers, source paths, error
  messages, and generated Markdown text.

### Remaining limitation

Prompt injection cannot be completely solved by delimiters or wording. Weft
still sends selected note content to a probabilistic model. Future action-taking
must add capability isolation, explicit approval, output validation, and a
separate authorization layer rather than trusting model text.

## 6. Private local storage

### Possible problem

The observed `.env`, `.weft/`, and generated index files used the process umask
defaults (`0644` files and a `0755` directory). On a multi-user machine, another
local account could read the API key, note text, graph metadata, or exact API
payloads.

### Why it existed

`Path.write_text()`, `open("a")`, and `np.savez(path)` delegated security to the
ambient umask. Git ignore rules prevent accidental commits but do not enforce
filesystem confidentiality.

### Repair

- Security-sensitive generated files use mode `0600`.
- dedicated `.weft/` directories use mode `0700`;
- writes use private temporary files and atomic replacement;
- append-only files use `O_NOFOLLOW`, verify a regular descriptor, and repair
  existing file permissions before appending;
- the existing workspace `.env` and `.weft` artifacts were corrected to
  `0600`/`0700` during this hardening task;
- setup documentation tells users to protect `.env` before sourcing it.

The code intentionally does not change arbitrary existing custom store-parent
directory permissions. It secures newly created parents and `.weft/`, plus every
file it owns.

## 7. Dependency advisories

The 2026-08-08 lockfile audit found two reviewed advisories:

- `torch==2.12.1` was affected by
  [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47),
  a local TorchScript memory-corruption issue fixed in 2.13.0. Weft does not call
  `torch.jit.script`, but the vulnerable package was installed transitively.
- `setuptools==81.0.0` was affected by
  [GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c),
  where Unicode normalization could bypass `MANIFEST.in` exclusions during sdist
  creation on macOS. Weft builds with Hatchling, but the package was present as
  a Torch dependency.

`pyproject.toml` now applies uv transitive constraints of `torch>=2.13.0` and
`setuptools>=83.0.0`. The regenerated lock selects `torch==2.13.0` and
`setuptools==84.0.0`. These are uv project constraints; downstream packaging
outside uv must perform its own lock and advisory scan.

## 8. Documentation drift

The previous agent guidance still said that no source code existed, the README
only described M0, and the tutorial used the old `NotesAssistant/` path. This was
not a direct exploit, but it encouraged unsafe architectural assumptions and
incorrect commands.

README and the CLI tutorial now describe the M0–M2 code, current repository root,
privacy defaults, overwrite behavior, artifacts, permissions, and migration
requirements. The workspace-local `CLAUDE.md` guidance was synchronized too, but
remains untracked under the repository's existing `**/CLAUDE.md` ignore policy.

## Migration checklist

1. Run `chmod 600 .env` if a local environment file exists.
2. Re-run `weft index`; older indexes have no security manifest and cannot be
   used by the hardened `suggest` command.
3. Confirm whether the new default `Private/` exclusion matches the Vault's
   organization. Prefer `--include`/`--exclude`; use `--include-private` only
   after reviewing the external-data consequence.
4. Add any project-specific secrets to repeatable `--redact` patterns.
5. Review or move an existing `_inbox.md`; do not normalize on
   `--overwrite-inbox` as a routine workflow.
6. Protect and retain `api-log.jsonl` according to the sensitivity of the source
   notes, or delete it through the user's normal secure-retention process when
   it is no longer needed.

## Verification performed

The hardened suite was first run in an isolated minimal environment with the
locked versions of pytest, LangGraph, NetworkX, and NumPy:

```bash
UV_CACHE_DIR=/tmp/weft-uv-cache PYTHONPATH=src \
  uv run --isolated --no-project \
  --with pytest==9.1.1 --with langgraph==1.2.8 \
  --with networkx==3.6.1 --with numpy==2.5.1 \
  python -m pytest -p no:cacheprovider
```

It was then run again in a clean isolated environment containing the complete
locked project, including Anthropic, Sentence Transformers, Torch 2.13.0, and
Setuptools 84.0.0:

```bash
UV_CACHE_DIR=/tmp/weft-uv-cache PYTHONPATH=src \
  uv run --isolated --frozen --extra dev \
  python -m pytest -p no:cacheprovider
```

Final result after all compatibility tests: `101 passed`.

The regenerated lock was also queried against OSV:

```bash
UV_CACHE_DIR=/tmp/weft-uv-cache uv audit --locked
```

Result: no known vulnerabilities or adverse project statuses in the 94 audited
third-party packages. `uv audit` currently identifies itself as experimental, so
this result is point-in-time evidence rather than a substitute for continuous
dependency scanning.

The suite includes live filesystem proofs for the original read and write
symlink classes. No real Vault content or API request is used by those tests.

## Remaining known limitations

- There is no local-only LLM backend. `ask` requires Claude whenever retrieval
  returns sources.
- Exact API audit logs are sensitive and have no rotation or retention policy.
- Link inference remains an in-memory O(N²) operation over notes.
- Graph-aware retrieval performs a full vector scan at the current milestone.
- Inbox checkboxes do not yet update accepted/rejected ledger state.
- File confinement substantially narrows symlink attacks, but the application
  is not designed to index a Vault being concurrently modified by a hostile
  local process with the same user privileges.
