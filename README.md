<h1 align="center">Graffiti</h1>
<p align="center"><em>The future can only be approached from the past</em></p>

<p align="center">
  <img src="./assets/img/National-Park-Museum-Shipka-Buzludzha.jpg alt="Logo" width="500"/>
</p>

Graffiti is a search system built on a single structural premise: search and
recognition are the same operation. To recognise an object is to place a
percept into a region of meaning by ruling out everything the percept is not;
to search for an object is to start from a description of what is wanted and
individuate, from an unbounded medium, a region satisfying that description.
Both terminate at the same kind of object — a bounded region of claim-space,
never a single point — and both are subject to the same lower bound on how
precisely they can resolve their target.

The project has two parts: a formal theory of search as individuation
(`docs/publications/semantic-causal-propagation`), and a domain-specific
language, Graffiti (`.grf`), whose single primitive realises that theory as an
executable script. There is no boundary in the design between search over a
local machine and search over a remote network — both are, prior to
individuation, contents of a single undifferentiated medium.

## The theory

The formal work lives in
[`docs/publications/semantic-causal-propagation/semantic-causal-propagation.tex`](docs/publications/semantic-causal-propagation/semantic-causal-propagation.tex),
a self-contained manuscript. Every object in it is a finite weighted graph or
a construction on one; no metric space, measure, or probability calculus is
introduced beyond a derived normalisation. The central results:

- **The resolution floor.** A search medium spanning both a local machine and
  the unbounded remote network is never exhausted by any finite search
  process. From this single order fact — non-completability, not a
  cardinality claim — it follows as a theorem that every search carries a
  strictly positive floor: no search returns an exact point, only a bounded
  region.
- **Individuation by negation.** In a medium with no privileged claim, a claim
  can be fixed only as the complement of what it is not. Recognition
  (decoding a percept into a claim) and search (projecting a description onto
  a region of the medium) are proved to be inverse readings of a single
  relation, not two different procedures pointed at each other.
- **Representation mobility.** Paraphrases of a fixed claim are equivalent
  representations under an averaging constraint; switching between them costs
  no additional search. "Peter", "a Harvard medical student", and a much
  longer clinical description can individuate the same claim without any of
  the three being a re-verification of the others.
- **Path opacity.** A multi-step search trajectory is admissible if and only
  if it converges on its target; interior steps are otherwise unconstrained
  and may look arbitrarily unlike the eventual result without invalidating
  it.
- **Catalytic composition and coherence.** Independent search steps compose
  multiplicatively, with a precise law of diminishing returns and a
  saturation dichotomy. A claim is robustly grounded against a single
  dissenting source only if it is supported by a strongly connected structure
  of at least three independent catalysts — never a linear chain, never two.
- **Closure.** A search is finished when every further available catalyst
  resolves to a claim-region already reached — a criterion strictly stronger
  than crossing a confidence threshold, and the correct replacement for it. A
  search that cannot close terminates by declining rather than by guessing.

The manuscript gives full proofs, an executable-semantics specification of
the Graffiti language derived directly from the theory, and a numerical
validation suite. `docs/publications/semantic-causal-propagation/validation`
is a self-contained Python implementation of the calculus (an exact
Edmonds–Karp maximum-flow backend with no external dependency) that checks
every theorem against constructed instances; `docs/publications/semantic-causal-propagation/figures`
regenerates the manuscript's eight validation panels directly from that
suite. Both run to completion in under one second.

```bash
# run the full theorem-validation suite (writes JSON results)
cd docs/publications/semantic-causal-propagation/validation
python run_all.py

# regenerate the eight validation figure panels
cd ../figures
python run_all.py
```

## Graffiti, the language

Graffiti scripts (`.grf`) structure a search as a project rather than a
single query. The language's sole primitive is `seek`: a mandatory exclusion
clause (what the result is not), a target claim-region, an optional explicit
chain of catalysts, and a closure-based termination condition. A project is a
directed acyclic graph of `seek` bindings, so later steps can narrow using
the exclusions and results of earlier ones.

```text
project apollo_budget {
  seek launch_date
    not{ "secondary summaries without citation" }
    toward{ launch_date_of("Apollo 11") }
    until converge
    yield launch_date

  seek budget_overrun
    not{ "post-hoc estimates without primary-source citation" }
    toward{ cost_vs_authorization(launch_date) }
    via{ web_search(budget_overrun)
           >> local_notes(budget_overrun)
           >> archive_lookup(budget_overrun) }
    until converge otherwise decline
    yield budget_overrun

  goal {
    claims: [launch_date, budget_overrun]
    coherence: >= 0.5
  }
}
```

Catalysts — the operations a `via` chain composes — are namespace-neutral by
theorem: a local file scan, a remote API call, and an inference from a
locally hosted or externally retrieved machine-learned model are
computationally uniform in the calculus. A runtime orchestrator dispatches
committed `seek` steps against a registry of such catalysts; nothing in the
theory or the language special-cases which kind of resource realises a given
step.

## Repository layout

```
docs/publications/semantic-causal-propagation/
  semantic-causal-propagation.tex   the manuscript
  references.bib
  validation/                       Python validation suite (14 theorem checks)
  figures/                          panel-generation scripts and rendered PNGs

crates/                             prior Rust workspace (see note below)
```

The `crates/` workspace predates the theory above and reflects an earlier
line of work on environmental measurement and atmospheric processing; it is
not yet an implementation of the individuation calculus or the Graffiti
language and should not be read as one. The TypeScript module implementing
the Graffiti compiler and runtime is the current implementation target and
is under active discussion.

## Status

The theory is complete and validated (see the manuscript's Numerical
Validation section and the accompanying suite). The language specification
— lexical grammar, context-free grammar, static type system, and operational
semantics — is written and proved sound in the manuscript. An implementation
of the compiler and orchestration runtime does not yet exist.

## License

MIT. See [LICENSE](LICENSE).

## Author

Kundai Farai Sachikonye, Technical University of Munich.
`kundai.sachikonye@wzw.tum.de`
