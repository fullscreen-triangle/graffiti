# Graffiti Search DSL — First Sketch

Working name for the language: **`.grf`**. This is a design sketch, not a
spec — modeled directly on the pattern shared by `honjo-dsl.tex`,
`srn-transmission-protocol.tex`, `media-effect-encoder.tex`, and
`domain-grammer-specification.tex` (Turbulance), applied to Kundai's stated
goal: a search DSL for a research-OS module where a script structures a
multi-step, non-casual research process over an unbounded medium (local
machine + internet, no boundary between the two).

## 1. The one primitive: `seek`

Every sibling DSL has exactly one computational primitive, exposed at
different arities/sugars. Graffiti's is `seek`: the act of individuating a
claim-cell from the medium (everything indexable, local or remote) by
negation, at a strictly positive floor.

```
seek NAME
  not   { <boundary> }      -- mandatory: what this is NOT (SRN-style)
  toward{ <region> }        -- the target cell: a claim/answer shape (MEE-style goal)
  via   { <chain> }         -- optional explicit catalyst chain (compose primitives)
  until <admit>              -- convergence criterion (Honjo track-style)
  yield NAME
```

- `not{}` is **mandatory**, exactly as in SRN's glyph. An unbounded medium
  makes "what this is" unspecifiable directly (Individuation-by-Negation,
  `instantiation-of-finite-weighted-graphs.tex` T1); the only tractable move
  is to state what's excluded. A `seek` with no `not{}` is rejected at
  parse time, not just discouraged.
- `toward{}` names a region of claim-space (à la MEE's behaviour cell), not
  a point. "Find the founding date of X" is a `toward` region containing
  every source that would individuate that fact, not a single expected
  string.
- `via{}` is optional — if omitted, the compiler derives the chain
  backward from `toward{}`, exactly as MEE's compiler derives effect
  chains backward from the target behaviour cell (Backward Trajectory
  Completion). If present, the user is naming specific catalysts
  (search engines, local grep, a specific API) the way MEE's `compose(...)`
  lets a user supply explicit primitives alongside `acts_like`.
- `until <admit>` is `converge` (residue crosses the floor — "good enough,
  stop") or `diverge` (explicitly decline — see §4) or an ordinary
  condition, mirroring Honjo's `track ... until converge|diverge|cond`.

## 2. The floor is load-bearing, not decorative

```
floor 0.02   -- program-level; every seek inherits it unless overridden
```

A `seek` can never yield "the exact answer" — only a claim-cell whose
residue (distance from a fully individuated, medium-exhausting answer) is
`<= floor`. This isn't a UX nicety; it's the same accountability-typing
move as Honjo's `Cut` type (`res(v) >= floor(v)`, statically enforced) and
the reason a Graffiti script is honest about degree-of-confidence instead
of pretending to certainty a non-completable medium cannot deliver
(`semantic-uncertainty-propagation.tex`, `theory-of-advertising.tex`
Converse Floor Theorem: zero floor would mean no distinguishable claims and
no terminating searches at all).

## 3. Composition is catalytic, not additive

```
seek origin_year
  not   { fictional, disputed-without-citation }
  toward{ founding_year(subject) }
  via   { web.search("subject founding year")
            >> local.grep(subject)
            >> web.archive_lookup(subject) }
  until converge
  yield origin_year
```

`>>` is catalysis (same operator name/precedence as SRN): each stage closes
a *fraction* of remaining distance to the target cell, composing as
`1 - ∏(1 - κᵢ)` (the multiplicative law shared by every sibling paper and
DSL — Honjo's residue chaining, MEE's effect stacking, SRN's `catalyst-expr`).
This has two concrete, load-bearing consequences for the language:

- **Diminishing returns are visible to the type checker.** A `via` chain
  of near-duplicate catalysts (same source queried five ways) is flagged,
  same as MEE's `SaturationWarning` — diversify catalysts, don't repeat them.
- **Coherence requires ≥3 mutually-supporting catalysts**, not a linear
  chain, to cross the floor with any robustness (Linear Justification
  Failure / Coherence-requires-a-triangle, present in Honjo, MEE, and the
  advertising paper identically). A `via` chain of length 1 or 2 that
  claims `converge` is a compile-time `CoherenceWarning`: two independent
  sources can't outvote their own disagreement; three can.

## 4. Declining is a first-class outcome

```
until converge otherwise decline
```

Per `semantic-uncertainty-propagation.tex`'s two kinds of untranslatability
(shallow vs. deep) — if the catalyst chain's residue plateaus above the
floor rather than descending (a stall, in the scheduling paper's exact
sense — `Δ(n) = 0` over a window), the correct output is to **decline**,
not emit a guess. This is a runtime state (`stalled` → `declined`), not an
error: a `seek` that declines is reported to the project log as "no
sufficient claim found," which is itself useful research output.

## 5. A project is a sequence of committed seeks, not a session

```
project apollo_program {
  floor 0.02

  seek launch_date  not{...} toward{...} until converge yield launch_date
  seek crew_size    not{...} toward{...} until converge yield crew_size

  seek budget_overrun
    not   { post-hoc estimates without primary-source citation }
    toward{ cost(launch_date) vs congressional_authorization(launch_date) }
    via   { local.notes >> web.search(...) >> archive.gov(...) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, crew_size, budget_overrun]
    coherence: >= 0.5
  }
}
```

- Every `seek` is a **cut event**: it advances the project's monotone
  commit count `M` exactly as Honjo's cut count and SRN's partition count
  do. Re-running the same `seek` line is a **new** cut at a higher `M`,
  never a cached replay — this is what makes a project a faithful research
  *history*, not a memoized query cache. If the medium has changed (a page
  updated, a new paper indexed), re-seeking is supposed to notice.
- Later `seek`s can reference earlier `yield`s (`budget_overrun` uses
  `launch_date`) — this is what "structuring a search as a project" means
  operationally: a DAG of individuations where later cells narrow using
  the negation-boundaries established by earlier ones, not a flat list of
  independent queries.
- `goal{}` (Turbulance/MEE naming) is the project-level sufficiency
  threshold — analogous to the Scheduling paper's Sufficiency Stopping:
  individual `seek`s can be released above their own floor once the
  *project's* goal is satisfied, and — per that paper's proof — no single
  `seek` can compute this threshold itself. It has to be supplied top-down
  by the `goal` block, never inferred locally.

## 5b. The stopping criterion: closure, not confidence

`causal-table-propagation.tex` gives the exact mechanism for Kundai's
"reduction of sufficient unknowability" idea (2026-07-01 session), sharper
than a simple floor/confidence threshold:

- Every phrasing/source that resolves to the *same target cell* is a
  different **representation** of the same item (`thm:rep-mobility`) —
  "Peter," "Harvard med student," and "specialist in anatomical processes
  of diseased homo sapiens at an institution of higher learning in Boston"
  all satisfy the same averaging constraint against the same alignment.
  Switching between them commits **no new cut** — no progress, just a
  change of coordinates. A receiver-relative decoder (10-year-old vs.
  doctor vs. lawyer) places them at different literal coordinates
  (`theory-of-advertising.tex` receiver-relativity) but the *cell* is
  identical for all three.
- The correct stop condition is therefore not "residue crossed the floor"
  (too weak on its own) but **admissible-set closure**
  (`cor:gauge`/`thm:path-opacity`): stop a `seek` when every further
  candidate catalyst you could still invoke resolves to a propagation
  *endpoint-indistinguishable* from ones already found — i.e., when no
  further source can still generate a genuinely new path to a different
  cell. This is literally "no other possible paths can be generated,"
  formalized.
- Concretely for the DSL: `until converge` should mean *the admissible
  propagation class has stopped growing new equivalence classes*, not
  merely *residue ≤ floor*. A `seek` can hit the floor on its very first
  source and still be premature if a structurally different second source
  would land in a different cell (a genuine disagreement, not just a
  reformulation) — that's the shallow/deep untranslatability distinction
  from `semantic-uncertainty-propagation.tex` resurfacing here: same-cell
  convergence across independent catalysts is what licenses stopping,
  not any single catalyst's confidence.
- Local steps stay unconstrained the whole time
  (`thm:admissibility`/the "it came from space" remark): a `seek`'s
  intermediate fetches can look arbitrarily bad or off-topic mid-chain
  without invalidating the result, as long as the chain composes to
  something that converges. This is the formal license for `via{}` chains
  that pass through weird intermediate queries on the way to a good answer.

## 6. Open questions to resolve before this is a real spec

1. **What is a "catalyst" concretely?** Honjo's are named framework
   operations (`Individuate`, `Bond`); SRN's are network primitives; MEE's
   are physical-effect primitives. Graffiti's (`web.search`, `local.grep`,
   `archive_lookup`, ...) need a registry contract like Honjo's/MEE's
   plugin trait (signature + provider + namespace) so new sources can be
   added without touching the compiler core.
2. **What does `not{}` actually range over for an unbounded medium?**
   SRN's boundary is a predicate over partition coordinates (a closed
   grammar). Graffiti's medium has no such fixed coordinate system —
   `not{}` will likely need to be closer to Turbulance's evidence/motion
   layer (excluding *classes* of source/claim) than to SRN's literal
   coordinate predicates.
3. **Two-target compilation?** Honjo compiles to both a fast/interactive
   backend (TypeScript/WebGL) and an exact reference backend (Rust), with
   a proven equivalence-up-to-floor. Does Graffiti need an equivalent
   split — e.g. a fast local-index backend vs. an exact/exhaustive
   web-crawl backend — or is one execution model sufficient given it's
   explicitly *not* meant to be fast/casual?
4. **What is the OS-level calling convention?** How does a `.grf` project
   get invoked from the OS's own top-level DSL, and what does it hand
   back — a `Path`/`amalgamation` value (Honjo), a rendered artifact (MEE),
   or something project-specific (a claim ledger)?
