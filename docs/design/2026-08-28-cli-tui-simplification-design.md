# Weft CLI and TUI Simplification

## Decision

Expose user intent by default and keep implementation-level RAG controls as
advanced overrides.

The primary interface is:

```bash
weft
weft index <vault>
weft ask "<question>"
weft chat
weft suggest <vault>
```

The first TUI is a dependency-free interactive menu rather than a persistent
full-screen framework. This keeps packaging small while establishing the
service boundary required by a future Textual adapter if real usage needs one.

## Dependency direction

```text
CLI    TUI    REST
  \     |     /
   application services
           |
retrieval, indexing, memory, providers
```

Interfaces render output and collect input. They do not implement retrieval,
indexing, or suggestion workflows. The TUI never invokes the CLI through a
subprocess.

## Retrieval modes

| Mode | k | Hybrid | Graph | Rerank | Rerank pool | Memory query |
| --- | ---: | --- | --- | --- | ---: | --- |
| `fast` | 5 | no | yes | no | 20 | no |
| `balanced` | 8 | yes | yes | no | 20 | yes |
| `best` | 8 | yes | yes | yes | 30 | yes |

These are tuning defaults, not permanent pipeline contracts. Query rewriting
remains off unless explicitly enabled.

Mode selection resolves in this order:

```text
built-in balanced
< .weft/config.toml mode
< WEFT_MODE
< --mode
```

Explicit advanced flags then override individual preset fields. Boolean flags
are tri-state at the parser boundary so “not specified” is distinct from an
explicit enable or disable.

## Application services

The shared service module owns:

- index construction;
- one-shot ask configuration and execution;
- chat-session construction;
- suggestion generation and inbox persistence;
- provider/model/mode configuration.

`RetrievalConfig` is the resolved boundary passed to the existing RAG pipeline.
CLI, TUI, and REST all use the same resolver.

## TUI scope

Bare `weft` supports:

- Ask;
- Chat;
- Index;
- Discover links;
- Settings for mode, provider, model, and store path.

Benchmarking, model downloads, memory administration, REST lifecycle, and
advanced retrieval tuning remain CLI-only. Suggestion review and memory review
are deferred until their primary workflows justify a richer interface.

## Compatibility

Existing advanced flags remain valid. New inverse flags allow presets to be
overridden in both directions, for example:

```bash
weft ask "question" --mode best --k 12 --no-rerank
weft ask "question" --mode fast --hybrid --memory-query
```

No retrieval algorithm was removed or redesigned by this change.
