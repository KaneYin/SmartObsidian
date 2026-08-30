# 2026-08-30 — Vault-scoped episodic memory

## What changed

`Episode` records (the agent's episodic Q&A log) are now stamped with the
`vault_root` of the index they were logged against, read from
`<store>.manifest.json`. `MemoryStore.episodes()` filters out any episode
whose `vault_root` doesn't match the current store's vault -- including
records with no `vault_root` field at all, which fail closed rather than
being assumed to match.

`weft index` also now refuses to overwrite a store whose manifest names a
different vault unless `--force` is passed (`IndexVaultMismatchError`,
reusing the check already used by `weft suggest`).

## Why

Investigating a real `weft ask` run found that reusing a store path across
two different vaults let a stale, unrelated episode win memory-query recall
by text similarity alone and get pasted into the prompt as context,
producing a confusing, unrepresentative answer. The same missing vault
binding let `weft index` silently clobber an unrelated vault's index with
no warning.

## Regression coverage

- `tests/test_memory.py`: episodes are stamped with vault_root; recall
  excludes episodes from a different vault_root and episodes with a
  missing vault_root field.
- `tests/test_service.py`: `service_index()` raises `IndexVaultMismatchError`
  on a vault-mismatched re-index without `force=True`; `make_memory()`
  reads vault_root from the manifest.
- `tests/test_cli.py`: `weft index` without `--force` exits 1 against a
  mismatched store; `--force` proceeds.

## Known limitation

Semantic memory items (`preference`/`fact`/`decision`/`task`) are
intentionally NOT vault-scoped -- they're documented as durable facts about
the user, meant to persist across vaults. Only the episodic log is scoped.
