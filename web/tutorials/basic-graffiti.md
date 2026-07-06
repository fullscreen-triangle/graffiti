# Graffiti, in the browser playground

This is a walkthrough of the `.grf` language using the REPL at `/repl`. It
assumes nothing beyond what is already running in this repository --
every snippet below is copy-pasteable directly into a REPL cell.

Read `docs/publications/semantic-causal-propagation/semantic-causal-propagation.tex`
for the theory. This document is only about using the tool.

## What the playground is, and isn't

The `/repl` page is a scratchpad, not the product. It is a TypeScript
implementation of the calculus and the language, running entirely in
your browser tab, with mock catalysts standing in for real search
engines and models. It exists so an idea can be tried in seconds. Ideas
that hold up are re-implemented in Rust as part of Buhera OS itself --
this playground is deliberately the fast, disposable end of that
pipeline, not a preview of the production runtime.

Each REPL cell is a JavaScript function body. The whole `graffiti`
library is available as the `graffiti` object in every cell, and
variables you `let`/`const` in one cell are visible to later cells (a
persistent shared scope, the one thing a REPL needs that a plain
`<textarea>` does not give you for free). A cell body may `await`,
since compiling and running a `.grf` program is asynchronous.

## 1. A seek with no catalysts

The smallest legal Graffiti program is one `seek`. Every `seek` needs a
`not{}` (what the claim is not -- mandatory, since a claim in a medium
with no privileged claim is only ever fixed by negation), a `toward{}`
(the target region), and an `until` clause (when to stop). Paste this
into the first cell and run it (the "run" button, or Ctrl/Cmd+Enter):

```js
const source = `
floor 0.02

project first {
  seek answer
    not{ "unsourced claims" }
    toward{ founding_year_of("Technical University of Munich") }
    until converge
    yield answer
}
`;

const registry = new graffiti.CatalystRegistry();
const { projectResults } = await graffiti.runSource(source, registry);
const answer = projectResults.get("first").get("answer");
answer;
```

You should see a `Claim` object: `{ kind: "Claim", targetClaim: ...,
floor: 0.02, residue: ..., classesAtClosure: [...] }`. No `via{}` chain
was given, so the runtime treats the `toward{}` description itself as
the sole reached claim, and since there is only one claim, the search
converges immediately.

`floor 0.02` is not decoration: `thm:floor` proves every search has a
strictly positive floor, and the type checker rejects a program with no
`floor` declaration. Every returned `Claim` carries this floor and a
`residue >= floor` -- there is no such thing, in this calculus, as an
exact answer of residue zero (`cor:no-sharp`).

## 2. Registering a catalyst

An unadorned `seek` is rarely useful -- real searches dispatch to real
sources. A *catalyst* is anything that takes the current claim and
returns a (possibly different) claim plus a *catalytic power* in
`[0, 1]` (how much of the remaining gap to the target it closed). The
playground ships two reference catalysts so you can wire up a `via{}`
chain without a real search engine or model:

- `createFixtureSearchCatalyst(name, fixtures, power?)` -- a fixed
  lookup table standing in for a search engine or document store.
- `createMockInferenceCatalyst(name, transform, power?)` -- a
  deterministic transform standing in for a call to a local or remote
  model (this is where a real integration would call Ollama or a
  Hugging Face-hosted model instead).

```js
const source = `
floor 0.02

catalyst web_search { namespace: remote input: Region output: Claim }

project lookup {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    via{ web_search(query: launch_date) }
    until converge
    yield launch_date
}
`;

const registry = new graffiti.CatalystRegistry();
registry.register(
  graffiti.createFixtureSearchCatalyst("web_search", {
    launch_date: "1969-07-16",
  }),
);

const { projectResults } = await graffiti.runSource(source, registry);
projectResults.get("lookup").get("launch_date");
```

`web_search` is declared once (`catalyst ... { namespace: ... input:
... output: ... }`) and registered once, on the JavaScript side, with
an implementation. The `.grf` source only ever names catalysts by
signature; it is never told whether a given name is a local file scan,
a remote API call, or a model inference (`thm:namespace-neutral` --
namespace is scheduling metadata, invisible to the calculus).

The `query:` label matters: `createFixtureSearchCatalyst`'s reference
implementation looks up its fixture table by the named argument
`query`; an unnamed (positional) argument or a different label will
miss the fixture and fall through to an `"unresolved:..."` claim
instead. A real catalyst provider is free to read its `args` however
it likes -- this is just how the two reference/mock catalysts do it.

## 3. A coherent multi-catalyst chain

Three independently-sourced catalysts that agree with each other ground
a claim far more robustly than one. `thm:triangle` proves this needs
*at least* three mutually-supporting sources arranged in a cycle --
never one, and never exactly two. Chain catalysts sequentially with
`>>`:

```js
const source = `
floor 0.02

catalyst web_search    { namespace: remote input: Region output: Claim }
catalyst local_notes   { namespace: local  input: Region output: Claim }
catalyst archive_lookup{ namespace: remote input: Region output: Claim }

project budget {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget_overrun
    not{ "post-hoc estimates without primary-source citation" }
    toward{ cost_vs_authorization(launch_date) }
    via{ web_search(query: budget_overrun)
           >> local_notes(query: budget_overrun)
           >> archive_lookup(query: budget_overrun) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, budget_overrun]
    coherence: >= 0.5
  }
}
`;

const registry = new graffiti.CatalystRegistry();
registry.register(
  graffiti.createFixtureSearchCatalyst("web_search", {
    budget_overrun: "budget:on-authorization",
  }),
);
registry.register(
  graffiti.createMockInferenceCatalyst("local_notes", (c) => `${c}:corroborated`),
);
registry.register(
  graffiti.createMockInferenceCatalyst("archive_lookup", (c) => `${c}:corroborated`),
);

const { projectResults } = await graffiti.runSource(source, registry);
const budget = projectResults.get("budget");
[...budget.entries()];
```

Note `until converge otherwise decline` on `budget_overrun`: with three
catalysts in play, closure might be *contested* (the sources disagree),
and `otherwise decline` says to accept that as a normal outcome rather
than erroring. Here all three catalysts are chained with `>>`
(sequential composition -- each catalyst's output feeds the next), so
there is only one propagation and one reached claim: a single-branch
chain converges by construction. To see a genuine disagreement between
independently-sourced catalysts, see the next example, which dispatches
two catalysts in *parallel* (`||`) against the same claim.

## 4. Contested closure (an honest "I don't know")

When independent catalysts resolve to genuinely different claims, the
correct behavior is not to guess -- it is to report the disagreement.
This is `thm:decline`: every search terminates either in convergent
closure (one claim) or contested closure (a `Decline` carrying every
distinct claim-region found).

```js
const source = `
floor 0.02

catalyst source_a { namespace: local input: Region output: Claim }
catalyst source_b { namespace: local input: Region output: Claim }

project disputed {
  seek cause
    not{ "single-source attribution" }
    toward{ primary_cause_of(event) }
    via{ source_a(cause) || source_b(cause) }
    until converge otherwise decline
    yield cause
}
`;

const registry = new graffiti.CatalystRegistry();
registry.register(
  graffiti.createMockInferenceCatalyst("source_a", () => "cause:mechanical-failure"),
);
registry.register(
  graffiti.createMockInferenceCatalyst("source_b", () => "cause:human-error"),
);

const { projectResults } = await graffiti.runSource(source, registry);
const cause = projectResults.get("disputed").get("cause");
cause;
// { kind: "Decline", classes: [["cause:mechanical-failure"], ["cause:human-error"]], ... }
```

`||` dispatches catalysts in parallel branches (as opposed to `>>`,
which chains them sequentially, each fed the previous one's output).
Here the two branches disagree, closure is reached (no further
catalyst is available to resolve the tie), and the `seek` yields a
`Decline` rather than silently picking one side.

## 5. Compiling without running

If you only want to check a `.grf` program's syntax and types --
floor positivity, unknown catalyst references, a coherence warning on
an under-supported `via{}` chain -- use `compile` instead of
`runSource`. It's synchronous and doesn't need a registry populated
with working providers:

```js
const bad = `
project no_floor {
  seek x
    not{ "unsourced claims" }
    toward{ y }
    until converge
    yield x
}
`;

const result = graffiti.compile(bad);
result.ok;       // false
result.diagnostics;
// [{ severity: "error", code: "NO_FLOOR_DECLARED", message: "..." }]
```

Note that `not{}` itself is checked earlier, at parse time, not by the
type checker: a `seek` with no `not{}` clause at all is a `ParseError`
(a thrown exception), not a diagnostic in `compile`'s result -- by
`thm:negation`, an expression asserting no exclusion asserts no
individuation, so the parser refuses to build an AST node for it in
the first place.

## 6. Where to go from here

- `src/graffiti/examples/apollo-budget.ts` is the same coherent-chain
  example above, runnable standalone with `npx tsx` outside the
  browser.
- `src/graffiti/test/` mirrors every theorem in the manuscript with a
  runnable check; `npm run test:graffiti` runs all of them.
- Real catalysts (a real search API, a real Ollama or Hugging Face
  call, a real local filesystem scan) implement the same
  `CatalystDefinition` contract as the mock catalysts used above --
  nothing else in a `.grf` program changes when a mock is swapped for
  the real thing.
- Because this playground is TypeScript, it is intentionally limited
  to what a browser tab can run cheaply: mock or lightweight catalysts,
  small graphs, short chains. Push an idea until it's convincing here,
  then carry it into the Rust implementation, which is where this
  calculus becomes the production Buhera OS search runtime.
