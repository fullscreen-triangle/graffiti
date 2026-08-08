# Integrating spraypaint

**Two audiences, one binary.**

1. **buhera OS** embeds the crossfilter *web tool* — `@buhera/spraypaint`, a
   TypeScript module with React + D3 chart components — so a human can search a
   repo interactively and steer the search by dragging the charts.
2. **Any other repo** installs the `spraypaint` **Rust CLI** and calls it from the
   shell, a script, or an AI agent to get a ranked context slice instead of
   reading whole files.

Both paths run the *same* executable. The web tool is a thin, editable view over
`spraypaint … --json`; it never simulates a result. So this document is really
one story told twice: how to talk to the binary from TypeScript (buhera), and how
to talk to it from anywhere else (the CLI).

---

## Part 0 — The binary is the source of truth

Everything below assumes one installed executable:

```
spraypaint index    [--root DIR] [--json] [--dry-run] [--window N] [--overlap N] [--scene-idf]
spraypaint ask <QUERY> [--root DIR] [-k/--budget N=12] [--json] [--dry-run] [--flat] [--scenes a,b]
spraypaint identity [--root DIR] [--json]     # Inv 1: fingerprint + χ
spraypaint count    [--root DIR] [--json]     # Inv 2: never-resetting committed count
spraypaint scenes   [--root DIR] [--json]     # detected / overridden scenes
spraypaint verify   [--root DIR] [--json] [--allow-degenerate]   # re-check all four invariants
spraypaint serve    [--root DIR] [--port N=7373] [--host H] [--open] [--allow-index]
```

- `index` is the **construction** phase (exclusive lock, writes
  `.spraypaint/index.json`, emits no ranked answer).
- `ask` is the **commitment** phase (shared lock, runs a fresh BM25 +
  water-filling walk, increments the committed count).
- `--json` is the machine-facing output every integration should use.
- `--dry-run ask` prints the compiled query, per-scene score heads, `p*`, and the
  allocation **without incrementing the count** — a zero-act read-out (Inv 3).
- `serve` hosts the JSON API and the embedded web UI on loopback. This is the
  supported way to drive `spraypaint` from a browser; see §1.3.
- `verify` exits `0` (all pass), `1` (a real breach), or `2` (**new in 0.2.0** —
  nothing failed, but at least one check was not applicable). Any integration
  that treats "nonzero" as "breach" must be updated: exit 2 means the repository
  is too degenerate to verify — empty, single-document, single-scene, or sharing
  no vocabulary — not that something is wrong with it. `--allow-degenerate` maps
  `2 → 0`. The JSON keeps its top-level `pass` boolean unchanged.

The four invariants the tool guarantees, and why an integrator cares:

| Invariant | What it means for you |
|---|---|
| **1 — Conserved identity** | An index has a stable blake3 `fingerprint` + a χ value (Stoer–Wagner min-cut ≥ floor). Re-indexing the same content — even with files/passages/ids reordered — yields the same fingerprint. You can cache against it, and detect drift. |
| **2 — Never-resetting count** | `count` only ever goes up, +1 per committed `ask`. It survives restart and re-index. It is the repo's search history, not a session counter. |
| **3 — Search-not-fetch** | Snippets are re-read from disk at query time; there is no answer cache. Same `(index, query)` ⇒ identical results; edit a file and results change. Never assume a stored answer. |
| **4 — Exclusive phases** | `index` and `ask` never overlap; you will never read a half-written index. Safe to `index` in CI and `ask` from many callers. |

---

# Part 1 — buhera OS integrates the web tool

The web tool ships as **`@buhera/spraypaint`** (in
`buhera/buhera-os/spraypaint-ts`). It has two layers:

- a **headless core** (`.` and `./runner-node` subpaths) — types, a client, the
  crossfilter→query inversion functions, undo history, and a `SpraypaintSession`
  that owns the whole loop. No React, no DOM.
- **React + D3 components** (`./react` subpath) — the charts that *are* the
  crossfilter surface: `AllocationChart`, `ResultsList`, `IdentityBadge`,
  `SpraypaintPanel`, and the `useSpraypaintSession` hook.

The core is browser-safe. All Node coupling lives in `./runner-node`
(`NodeRunner`, which shells out via `child_process`). That split is the whole
integration trick: **the browser bundle imports the core; the binary runs on the
server; an HTTP route bridges them.**

```
  browser (React charts)  ──fetch──▶  Next.js route  ──NodeRunner──▶  spraypaint --json
        ▲                                                                    │
        └──────────────── AskResult JSON ◀──────────────────────────────────┘
```

> **Two topologies, and which one you want.** The diagram above is for a *hosted*
> site indexing a corpus **you** control on **your** server. It cannot serve a
> user's own repository: a web page has no way to spawn a process on a visitor's
> machine, and no amount of route-writing changes that.
>
> For the far more common case — a user searching their *own* code — the
> direction inverts. The binary hosts the UI instead of a server hosting the
> binary:
>
> ```
>   browser  ──fetch (same origin)──▶  spraypaint serve  ──▶  actions::* ──▶ index
> ```
>
> One executable, `spraypaint serve`, no Node, no route, no CORS, and nothing
> leaves the machine. The web UI in `spraypaint-web/` is compiled into that
> binary and is the reference consumer of the API below. Use §1.1–§1.4 only if
> you are building the hosted variant.

## 1.1 Add the dependency

Because it lives in the same buhera monorepo, wire it as a workspace package
rather than publishing:

```jsonc
// buhera-os/apps/<app>/package.json
{
  "dependencies": {
    "@buhera/spraypaint": "workspace:*",
    "react": ">=18",
    "d3": ">=7"          // only needed by the /react subpath
  }
}
```

Build the module once (it compiles `src → dist`):

```bash
cd buhera-os/spraypaint-ts
npm install
npm run build          # tsc → dist/  (emits ., runner-node, react)
```

`react` and `d3` are **optional peer deps** — they are pulled in only when you
import `@buhera/spraypaint/react`. A pure server integration (Part 1.5) needs
neither.

## 1.2 The canonical artifact: `AskQuery`

Every input surface — a typed prompt, a hand-edited `-k` field, a chart drag —
converges on one runnable object:

```ts
interface AskQuery {
  query: string;            // positional <QUERY>
  budget: number;           // -k / --budget
  scenes: string[] | null;  // --scenes a,b   (null/empty ⇒ all scenes)
  flat: boolean;            // --flat
}
```

It maps 1:1 to `spraypaint ask <query> -k <budget> [--scenes …] [--flat]`. This is
why the charts can be bidirectional: a gesture is *inverted* into a typed
`QueryDiff` against this object, applied, and re-run. An invalid query is
unrepresentable (unlike string-editing a DSL).

## 1.3 The server side — expose the binary over a route

`NodeRunner` runs the local `spraypaint`. Wrap `SpraypaintClient` in an API route
so the browser never needs the binary:

```ts
// app/api/spraypaint/route.ts   (Next.js App Router, server-only)
import { NextRequest, NextResponse } from "next/server";
import { SpraypaintClient } from "@buhera/spraypaint";
import { NodeRunner } from "@buhera/spraypaint/runner-node";
import type { AskQuery } from "@buhera/spraypaint";

const client = new SpraypaintClient(
  new NodeRunner({ bin: "spraypaint" }),   // resolved on PATH
  { root: process.env.SPRAYPAINT_ROOT },    // repo to search
);

export async function POST(req: NextRequest) {
  const { action, query } = (await req.json()) as {
    action: "ask" | "dryRun" | "identity" | "count" | "scenes" | "verify" | "index";
    query?: AskQuery;
  };
  try {
    switch (action) {
      case "ask":      return NextResponse.json(await client.ask(query!));
      case "dryRun":   return NextResponse.json(await client.dryRun(query!));
      case "identity": return NextResponse.json(await client.identity());
      case "count":    return NextResponse.json(await client.count());
      case "scenes":   return NextResponse.json(await client.scenes());
      case "verify":   return NextResponse.json(await client.verify());
      case "index":    await client.index(); return NextResponse.json({ ok: true });
      default:         return NextResponse.json({ error: "unknown action" }, { status: 400 });
    }
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
```

Notes:
- `client.ask()` throws `SpraypaintError` on nonzero exit or non-JSON — the
  `catch` turns that into a 500 with the binary's stderr.
- `client.verify()` deliberately tolerates a nonzero exit and returns the JSON
  verdict (a breach is a valid answer, not a crash).
- Keep the binary and its `.spraypaint/` index on the server. The browser only
  ever sees typed JSON.

## 1.4 The browser side — a `SpraypaintRunner` that speaks HTTP

`SpraypaintClient` depends only on the `SpraypaintRunner` interface
(`run(args) → {stdout, stderr, exitCode}`). To drive it from the browser, give it
a runner that POSTs to the route instead of spawning a process:

```ts
// lib/spraypaint-browser.ts  (client-safe: no node imports)
import { SpraypaintClient, type SpraypaintRunner, type RunOutput } from "@buhera/spraypaint";

class HttpRunner implements SpraypaintRunner {
  async run(args: string[]): Promise<RunOutput> {
    // The React components below don't use the runner directly — they use a
    // SpraypaintClient. If you prefer a raw-args bridge, POST {args} to a route
    // that calls new NodeRunner().run(args). Most apps use the typed client
    // (1.3) instead and never construct args by hand.
    const res = await fetch("/api/spraypaint/raw", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ args }),
    });
    return res.json();
  }
}

export const browserClient = new SpraypaintClient(new HttpRunner());
```

In practice you have two clean choices, pick one:

- **Typed-action route (recommended, 1.3):** the route calls the *methods*
  (`ask`, `identity`, …). The browser calls `fetch("/api/spraypaint", {action})`.
  Simplest; the binary's argv never leaves the server.
- **Raw-args route:** the route calls `new NodeRunner().run(args)` and the
  browser uses the `HttpRunner` above with a real `SpraypaintClient`. Use this if
  you want the *client's* JSON parsing / error typing on the browser side.

## 1.5 Drop in the panel (the whole loop, one component)

For the full crossfilter experience, render `SpraypaintPanel`. It owns a
`SpraypaintSession` via `useSpraypaintSession` and wires all three surfaces:

```tsx
// A client component
"use client";
import { SpraypaintPanel } from "@buhera/spraypaint/react";
import { browserClient } from "@/lib/spraypaint-browser";

export function Search() {
  return (
    <div style={{ height: "80vh" }}>
      <SpraypaintPanel
        client={browserClient}
        initialQuery={{ query: "water-filling attention", budget: 12, scenes: null, flat: false }}
        autoRunOnMount
      />
    </div>
  );
}
```

What the user gets, and what each gesture *does to the binary*:

| Gesture in the chart | Inversion function | Real flag change |
|---|---|---|
| Click a scene's allocation bar | `invertSceneToggle` | add / remove it from `--scenes` |
| Drag the `p*` clearing-price line | `invertPriceDrag` | move `-k` to admit more / fewer passages |
| Budget `+/-` stepper, `-k` field | `invertBudgetStep` / `setBudget` | change `-k` |
| Flat / grouped toggle | `invertFlatToggle` | add / remove `--flat` |
| Type in the prompt, Ctrl+Enter | `editQueryText` | change `<QUERY>` |

Every gesture calls `applyAndRun`, which advances the `AskQuery`, re-invokes the
binary through the route, and redraws. The `IdentityBadge` shows χ (with ✓/✗ vs
floor), the committed count, `p*`, budget, and the fingerprint — the novel
quantities made visible, since they have no pre-existing intuition.

## 1.6 Build a bespoke layout instead

If `SpraypaintPanel` isn't the shape you want, use the hook and compose the
individual charts:

```tsx
"use client";
import { useSpraypaintSession, AllocationChart, ResultsList } from "@buhera/spraypaint/react";
import { browserClient } from "@/lib/spraypaint-browser";

export function CustomSearch() {
  const s = useSpraypaintSession(browserClient, {
    query: "", budget: 12, scenes: null, flat: false,
  });
  const { state, running, run, applyAndRun } = s;
  const onGesture = (diff) => void applyAndRun(diff, "crossfilter");

  return (
    <>
      <button onClick={run} disabled={running}>{running ? "…" : "Run"}</button>
      <AllocationChart query={state.query} result={state.result} stale={state.stale} onGesture={onGesture} />
      <ResultsList     query={state.query} result={state.result} stale={state.stale} onGesture={onGesture} />
    </>
  );
}
```

The hook exposes `state` (query, result, stale), `running`, `error`, and the loop
verbs `run` / `applyGesture` / `applyAndRun` / `undo` / `redo` (+ `canUndo` /
`canRedo`). A UI never mutates a query directly or touches the binary — it calls
these verbs, and every chart draws from `state`. That discipline is what keeps
prompt, editor, and chart-gesture edits peers on one undo timeline.

## 1.7 Headless use (no charts)

The core works without React — for a server task, a test, or a CLI-in-JS:

```ts
import { SpraypaintClient, SpraypaintSession, invertSceneToggle } from "@buhera/spraypaint";
import { NodeRunner } from "@buhera/spraypaint/runner-node";

const client = new SpraypaintClient(new NodeRunner(), { root: "/path/to/repo" });
const session = new SpraypaintSession(client, { query: "min-cut identity", budget: 12, scenes: null, flat: false });

const first = await session.run();                                   // real ask (+1 count)
const diff  = invertSceneToggle(session.state().query, first.allocation, "docs");
if (diff) session.applyGesture(diff, "crossfilter");                 // query changes, marked stale
const next = await session.run();                                    // re-run → new allocation
session.undo();                                                      // step back on the unified timeline
```

## 1.8 Operational checklist for buhera

- **Install the binary on the server** (Part 2.1); `NodeRunner` resolves
  `spraypaint` on `PATH`, or pass `{ bin: "/abs/path/spraypaint" }`.
- **Index before first ask.** Run `spraypaint index` (or `client.index()`) in
  deploy/CI. The construction lock makes this safe to run while readers wait.
- **Set the search root** via `SpraypaintClientOptions.root` or
  `NodeRunner({ cwd })`. Without it the binary walks up to `.git`/`.spraypaint`.
- **Keep `ask` on the server.** The binary and index are server assets; the
  browser sees only typed JSON.
- **Re-index on content change.** Because of Inv 3, stale content = stale
  results; wire a re-index into your content pipeline.

---

# Part 2 — Any other repo uses the Rust CLI

No TypeScript, no buhera. Just the binary, callable from any repo exactly like
`purpose`.

## 2.1 Install

```bash
# from the graffiti checkout
cargo install --path spraypaint --force      # → ~/.cargo/bin/spraypaint
spraypaint --version
```

Ensure `~/.cargo/bin` is on your `PATH`. The binary is self-contained; it needs
nothing from graffiti at runtime.

## 2.2 Index once, ask many times

```bash
cd /path/to/any/repo
spraypaint index                       # writes .spraypaint/index.json  (add to .gitignore)
spraypaint ask "how is auth handled"   # ranked passages, grouped by scene
spraypaint ask "retry backoff" -k 20   # widen the budget
spraypaint ask "parser" --scenes src,tests   # restrict to scenes
spraypaint ask "parser" --flat         # one global ranked order
```

**Scenes** are auto-detected top-level directories at index time; override them
with `.spraypaint/scenes.toml`. `spraypaint scenes` lists them with document and
passage counts. The water-filling rule divides the `-k` budget across scenes by a
single clearing price `p*`, so one huge dense directory cannot monopolise the
results — every relevant scene keeps presence.

Add to the consuming repo's `.gitignore`:

```
.spraypaint/
```

## 2.3 Use it from an AI agent (the intended use)

`spraypaint ask "…" --json` is a **token-cheap retrieval primitive**: instead of
reading whole files into a model's context, the agent gets a ranked context slice.
The JSON is stable and small:

```jsonc
{
  "results": [
    { "path": "src/auth.rs", "scene": "src", "score": 8.31,
      "start_line": 40, "end_line": 79, "snippet": "…re-read from disk…" }
  ],
  "allocation": [ { "scene": "src", "allocated": 8, "available": 22 },
                  { "scene": "tests", "allocated": 4, "available": 9 } ],
  "budget": 12, "price": 3.14, "query_terms": ["auth", "handl"],
  "committed_count": 57, "identity_fingerprint": "b3:e32d98…"
}
```

An agent loop:

```bash
spraypaint ask "where is the retry budget configured" -k 8 --json \
  | jq -r '.results[] | "\(.path):\(.start_line)-\(.end_line)  [\(.scene)]  \(.score)"'
```

The agent reads only the returned `path:line` ranges, not entire files. Because of
Inv 3, the snippet is always the current on-disk content — no stale cache to
distrust.

## 2.4 Use it from shell / scripts

```bash
# top hit path only
spraypaint ask "config loader" -k 1 --json | jq -r '.results[0].path'

# how the budget was split across scenes
spraypaint ask "error handling" --json | jq '.allocation'

# a diagnostic that does NOT count as a committed ask (Inv 3)
spraypaint ask "draft query" --json --dry-run | jq '{price, allocation}'
```

## 2.5 Use it from CI (guard the invariants)

```bash
set -e
spraypaint index

# Distinguish the three outcomes. Collapsing 2 into failure makes CI red on
# repositories that are merely too small to verify; collapsing it into success
# makes CI green on repositories nothing was actually checked against.
spraypaint verify --json > verify.json && rc=0 || rc=$?
case $rc in
  0) echo "invariants verified" ;;
  1) echo "INVARIANT BREACH"; jq -r '.degeneracies[]?' verify.json; exit 1 ;;
  2) echo "not verifiable — degenerate repository:"
     jq -r '.degeneracies[]' verify.json
     # Decide deliberately. `exit 0` accepts it; `exit 1` treats an
     # unverifiable index as a build failure. Do not leave it implicit.
     exit 0 ;;
  *) echo "verify itself failed to run"; exit "$rc" ;;
esac

spraypaint identity --json | jq -r .fingerprint > .spraypaint.fingerprint
```

`verify` re-checks all four invariants and is a one-command certificate — but
only when it exits `0`. Exit `2` says the checks could not be applied: the
repository has no documents, one document, one scene, or no shared vocabulary
across documents, and in those regimes a PASS would be a statement about the
construction rather than about the corpus. `--allow-degenerate` collapses `2` to
`0` if that is the reading you want, and is the shorter form of the `case` above.

Committing the fingerprint lets a later job assert the index is byte-reproducible
for the same content (Inv 1).

## 2.6 Call it from any language (the universal contract)

There is nothing TypeScript-specific about the integration — the contract is
"spawn the process, pass `--json`, parse stdout." Any runtime can do it:

```python
import json, subprocess
def ask(query, k=12, root="."):
    out = subprocess.run(
        ["spraypaint", "ask", query, "-k", str(k), "--json", "--root", root],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)   # -> dict matching the schema in 2.3
```

The TypeScript module in Part 1 is exactly this pattern, typed and with the
crossfilter loop layered on top. If you want charts, use the module; if you want
a retrieval primitive, the CLI + `--json` is the whole API.

---

## Appendix — mapping the two paths

| Concept | buhera web tool (`@buhera/spraypaint`) | Any repo (CLI) |
|---|---|---|
| Runnable artifact | `AskQuery` object | `spraypaint ask …` argv |
| Invoke | `client.ask(query)` (throws on error) | `spraypaint ask … --json` (nonzero exit on error) |
| Diagnostic, no count bump | `session.preview()` / `client.dryRun()` | `spraypaint ask … --dry-run` |
| Identity (Inv 1) | `client.identity()` | `spraypaint identity --json` |
| Committed count (Inv 2) | `client.count()` / `result.committed_count` | `spraypaint count --json` |
| Scenes | `client.scenes()` | `spraypaint scenes --json` |
| Certificate (all invariants) | `client.verify()` | `spraypaint verify --json` |
| Steering | chart gestures → `QueryDiff` → re-run | edit `-k` / `--scenes` / `--flat` flags |

Same binary, same JSON, same four invariants. The web tool just makes the flags
draggable.
