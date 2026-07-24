# spraypaint

Full-text search for any repo, ranked by **BM25 within scenes** and allocated
across scenes by the **water-filling** rule of the *Split-Attention Synchronised
Agents* calculus. A sibling to `purpose`: where `purpose` locates symbols,
`spraypaint` retrieves ranked passages of file **content**.

It is a token-cheap retrieval primitive for AI agents — `spraypaint ask "..."`
returns a ranked context slice instead of the agent reading whole files.

## Install

```bash
cargo install --path spraypaint --force   # from the graffiti repo root
spraypaint --version
```

## Use: index once, ask many times

```bash
cd /path/to/any/repo
spraypaint index                       # scans the repo -> .spraypaint/index.json
spraypaint ask "how is attention divided across scenes"
spraypaint ask "water filling" --json  # machine output for agents
spraypaint ask "kuramoto" -k 20        # widen the result budget
spraypaint ask "identity" --scenes crates,docs   # restrict to named scenes
```

`ask` returns passages grouped by scene, so a dense area cannot crowd out the
rest. `--flat` ranks globally by score instead.

## Commands

| Command | Purpose |
|---|---|
| `spraypaint index [--root DIR] [--dry-run] [--window N] [--overlap N]` | Build `.spraypaint/index.json`. |
| `spraypaint ask "<query>" [--root DIR] [-k N] [--json] [--flat] [--scenes a,b] [--dry-run]` | Search. `--dry-run` prints diagnostics (price, allocation) and does **not** commit an act. |
| `spraypaint identity [--json]` | The conserved identity fingerprint + χ (Inv 1). |
| `spraypaint count [--json]` | The monotone committed count (Inv 2). |
| `spraypaint scenes [--json]` | Detected/overridden scenes. |
| `spraypaint verify [--json]` | Re-check all four invariants; nonzero exit on breach. |

## Scenes

By default each **top-level directory** is a scene (loose root files form
`(root)`). Override with `.spraypaint/scenes.toml`:

```toml
[scenes]
core = ["crates/core", "crates/s-entropy"]
docs = ["docs", "README.md"]
```

The result budget `k` is divided across scenes by a single price `p*`
(water-filling, Algorithm 1 of the paper): each scene contributes passages while
their relevance clears `p*`; scenes below it drop. A huge dense scene cannot take
all `k` slots unless its passages genuinely out-score the others.

## The four invariants

`spraypaint` is a faithful runtime for the blueprint of
`docs/sources/split-attention-agents.tex`:

1. **Conserved identity** — the index *is* the self-graph; a relabelling-
   invariant fingerprint (+ χ, the min-cut) is recomputed and checked on every
   load.
2. **Never-resetting count** — one committed act per non-dry-run `ask`; never
   decremented, survives re-index and restart.
3. **Search-not-fetch** — every query is a fresh BM25 + water-filling walk; the
   index stores **no answers** (snippets are re-read from disk at query time).
4. **Exclusive phases** — `index` (construction) holds an exclusive lock; `ask`
   (commitment) a shared one; they never overlap.

Run `spraypaint verify` for a one-command conformance certificate.

## Notes

- `.spraypaint/` is a local cache — add it to `.gitignore`.
- Ranking uses global IDF (comparable across scenes); `--scene-idf` is planned as
  an escape hatch for a per-scene reading.
- Hand-rolled BM25 over an inverted index (no external search service);
  deterministic given the same index and query.
```
