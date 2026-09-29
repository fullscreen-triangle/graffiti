# spraypaint specification

`kind: document` · `reader: any agent or tool, and the people who configure them` · `covers:` spraypaint 0.2.0 with coverage verdicts (source tree of 2026-09-30) · `links:` [verification: this document](#verification-this-document)

This specification is made of nodes, not chapters. Each `##` section is one node and can
be read on its own. A node opens with a line of attributes: its `kind` (concept, rule,
procedure, command, interface, artefact, failure, limit, verification), who it is for, and
links to the nodes it depends on. An agent can fetch one node by its anchor, for example
`specifications.md#procedure-agent-protocol`, without reading the rest.

If you read only three nodes, read [concept: purpose](#concept-purpose),
[procedure: agent protocol](#procedure-agent-protocol) and
[rule: what an agent may conclude](#rule-what-an-agent-may-conclude).

Every number, default, flag and field below was taken from the source in `spraypaint/src`,
and the behaviours the source leaves ambiguous were tested against the binary. Where a
statement is a design decision rather than a measured fact, the node says so.

## concept: purpose

`kind: concept` · `reader: anyone` · `links:` [concept: coverage verdict](#concept-coverage-verdict), [rule: choosing spraypaint](#rule-choosing-spraypaint)

spraypaint searches the text files of one directory tree. It answers three questions about
a query:

- which passages of the corpus contain the query's words, ranked;
- how the result budget is shared between the top-level areas of the tree;
- whether the corpus contains what the query is about at all.

It does **not** answer questions, summarise, or rank by meaning. A passage is returned
because it contains words, not because it is true or relevant in any deeper sense. The
third question exists because of that: without it, every search returns its best
look-alike with the same apparent confidence, and a reader cannot tell a real hit from a
passage that merely shares a common word with the query. See
[concept: coverage verdict](#concept-coverage-verdict).

Use is two steps: `spraypaint index` once, then `spraypaint ask "<query>"` as often as
needed. Each ask is a fresh search. The index stores word statistics and line ranges, never
text; the lines shown in a result are read from disk at the moment of the ask.

## rule: choosing spraypaint

`kind: rule` · `reader: agents deciding which tool to call` · `links:` [limit: lexical matching](#limit-lexical-matching)

| You want | Use |
|---|---|
| Every occurrence of an exact string or identifier | `grep`. It is exact, complete and cheaper. |
| Where a symbol is *defined* | A symbol index, or `grep` for the definition syntax. spraypaint shows the lines around a match; it does not know what a definition is. |
| The passages that discuss a topic, ranked, when you know some of the words | `spraypaint ask` |
| Whether this corpus says anything about a topic at all | `spraypaint ask --dry-run`, then read the verdict |
| Facts that live outside the corpus (the web, another store) | Not spraypaint. A `declined` verdict is the signal to look elsewhere. |
| The meaning of a passage, or whether it answers a question | Read the passage. No verdict or score says that. |

spraypaint finds passages by the words in them. It is the wrong tool when you cannot guess
any word the answer would contain; see [procedure: writing a query](#procedure-writing-a-query).

## procedure: agent protocol

`kind: procedure` · `reader: AI agents` · `links:` [procedure: writing a query](#procedure-writing-a-query), [concept: coverage verdict](#concept-coverage-verdict), [concept: committed act](#concept-committed-act), [rule: what an agent may conclude](#rule-what-an-agent-may-conclude)

1. **Check the binary.** `spraypaint --version`. A result without a `coverage` field comes
   from an older build that has no verdict; see [limit: versions](#limit-versions).
2. **Check the index.** `spraypaint scenes`. The error `no index at …` means run
   `spraypaint index` first. Indexing takes an exclusive lock, so asks wait for it.
3. **Write the query** as the words the answer would contain, not as a question. See
   [procedure: writing a query](#procedure-writing-a-query).
4. **Preview it.** `spraypaint ask "<query>" --dry-run --json`. A dry run costs nothing,
   commits nothing, and its JSON carries the verdict *and* the passages.
5. **Read `coverage.verdict` before any result.**
   - `declined`: the returned passages do not contain what the query is about. Read
     `coverage.terms`. A term with `df: 0` is absent from the whole corpus: try its other
     forms once (plural, another language, a synonym). If it is still absent, stop and
     report that the corpus does not contain it, or go to another source. A term with
     `df > 0` and `in_results: false` exists but was not returned: raise `-k` or restrict
     `--scenes`.
   - `partial`: the terms occur, but no single passage holds them all. Read the top
     evidence, then either split the query into two asks or narrow it.
   - `covered`: one passage holds every term. Read its evidence. This is necessary for an
     answer, not proof of one.
6. **Read before you quote.** The evidence is at most five lines. Open the file at
   `evidence_start_line` for the surrounding context before stating what a passage says.
7. **Commit the ask you rely on.** Run the final query without `--dry-run`, so the answer
   you use is recorded as one committed act. Do not commit exploratory asks; the count never
   goes down. See [concept: committed act](#concept-committed-act).
8. **Cite** as `path:evidence_start_line-evidence_end_line`, and quote the matched terms.

Never put credential-shaped words (`token`, `secret`, `password`, `api key`) into an ask on
a tree that may hold secrets; see [limit: disclosure and security](#limit-disclosure-and-security).

For many queries in a row, `spraypaint serve` answers from a cached index, where each CLI
call reloads it; see [limit: scale and cost](#limit-scale-and-cost).

## procedure: writing a query

`kind: procedure` · `reader: anyone writing queries, agents especially` · `links:` [concept: term](#concept-term), [limit: lexical matching](#limit-lexical-matching)

A query is a bag of words. Ordering, grammar and phrasing are ignored.

- **Use the words the answer would contain.** A passage matches only on the exact same
  word after lower-casing. `dentist` does not match `dentists`; `index` does not match
  `indexing`; `colour` does not match `color`.
- **Write keywords, not a question.** These words are removed from queries: `the a an of in
  for on at to is are and or how what which where who why when do does did i find show me
  get tell about`. Every other word is searched, so `can you find the part about scenes`
  searches `can`, `you`, `part`, `scenes`, and the first three add noise.
- **Rare words carry the weight.** Scores and the verdict weight each word by how few
  files contain it. One distinctive word is worth more than several common ones.
- **The verdict treats every term as required.** `covered` needs one passage holding
  every term. Adding variants of a word (`dentist dentists Zahnarzt`) helps *find*
  passages but lowers the verdict, since a passage rarely holds all the variants. Check
  coverage with one form at a time; widen with variants when looking for passages.
- **Identifiers are single words.** `water_fill`, `foo.bar` and `v0.2.0` are one word each.
  `HTTPServer` becomes the one word `httpserver`; camel case is not split. `std::fs`
  becomes `std` and `fs`, and `water-filling` becomes `water` and `filling`. See
  [concept: term](#concept-term).
- **Other scripts work.** Accented letters and non-Latin scripts are ordinary letters:
  `Zahnärztin` and `École` are words, lower-cased to `zahnärztin` and `école`.
- **Paths and file names are not searched.** Only file contents are indexed. Find files by
  name with a file-name search, not with spraypaint.
- **Repeated words count once.** `kuramoto kuramoto` is the query `kuramoto`.
- **Restrict by area** with `--scenes docs,src` when you know where the answer is.

## concept: corpus and root

`kind: concept` · `reader: anyone` · `links:` [command: index](#command-index), [limit: disclosure and security](#limit-disclosure-and-security)

The **root** is the directory that holds `.spraypaint/`. It is found by walking up from
the working directory, or from `--root` if given, to the first directory that contains
`.git` or `.spraypaint`; if none does, the starting directory is the root. `--root` is
therefore a starting point, not a boundary: `--root repo/docs` inside a git repository
indexes the whole repository.

A file enters the corpus when all of these hold:

- its extension is one of `rs py js ts tsx jsx go java c cpp h hpp cs rb php swift kt
  scala toml yaml yml json sh sql` (code), `md txt rst org` (prose) or `tex bib` (LaTeX).
  Files with no extension (`Makefile`, `Dockerfile`) are never indexed;
- no directory on its path is named `.git node_modules target build dist out vendor
  .spraypaint .purpose __pycache__ .venv venv .next .nuxt .cache coverage .idea .vscode`;
- it is not excluded by a `.ignore` file, which is read anywhere, or by a `.gitignore`
  file, which is read **only when the root is inside a git repository**. The global git
  ignore file is not read;
- it is at most 2 MiB;
- it is valid UTF-8. Other files are skipped silently.

**Hidden directories are indexed.** `.github/`, `.claude/` and any other dot-directory not
listed above enter the corpus, including the `.json` and `.toml` files in them. Exclude them
with a `.ignore` file (same syntax as `.gitignore`) when they hold anything private.

## concept: passage

`kind: concept` · `reader: anyone` · `links:` [concept: evidence](#concept-evidence), [command: index](#command-index)

A passage is the unit that is scored. Each file is cut into windows of **40 lines**
overlapping by **10**, so a new window starts every 30 lines. A window's start is moved back
by up to 4 lines to the nearest natural boundary: a blank line, a Markdown heading, a
`\section` or `\subsection`, or a line that opens a definition (`fn`, `pub fn`, `def`,
`class`, `struct`, `enum`, `trait`, `impl`, `function`, `func`, `type`, `interface`,
`module`). Windows with no text are dropped. `index --window N --overlap N` changes the
sizes.

Because windows overlap, one stretch of text can lie in two windows. An ask never returns
both: of two overlapping windows of the same file, only the higher-scoring one is kept, and
this happens *before* the budget is shared out.

Forty lines suit scoring but are too long to read as a result. What an ask shows is the
passage's [evidence](#concept-evidence).

## concept: term

`kind: concept` · `reader: anyone` · `links:` [procedure: writing a query](#procedure-writing-a-query)

A term is a word as the index sees it. Text is split into words by the Unicode word
boundary rules (UAX #29), then lower-cased. Consequences, all tested:

| Text | Terms |
|---|---|
| `water_fill`, `foo.bar`, `v0.2.0`, `3.14`, `don't` | one term each, unchanged |
| `HTTPServer` | `httpserver` |
| `std::fs`, `water-filling`, `e-mail` | split at the punctuation: `std` `fs`, `water` `filling`, `e` `mail` |
| `Zahnärztin`, `ÉCOLE` | `zahnärztin`, `école` |

There is no stemming, no plural reduction and no synonym list. Words longer than 64 bytes
are dropped. A query is split the same way as the index, and then the stopwords in
[procedure: writing a query](#procedure-writing-a-query) are removed. The stopword list
applies to queries only; indexed text keeps every word.

## concept: scoring

`kind: concept` · `reader: anyone, maintainers` · `links:` [concept: allocation](#concept-allocation)

Each passage is scored against the query by Okapi BM25 with `k1 = 1.2` and `b = 0.75`.

- **IDF is global and counted in files:** a term's weight is
  `ln((N − df + 0.5) / (df + 0.5) + 1)`, where `N` is the number of indexed files and `df`
  the number of files containing the term. Global IDF puts every scene's scores on one
  scale, which the allocation needs.
- **Length normalisation is per scene:** a passage's length is compared with the average
  passage length of its own scene, so a scene of long files is not penalised.
- A passage scores above zero only if it contains at least one query term.

A score means something only within one query. It is not a probability, and scores from
different queries or corpora cannot be compared. Judge a result by the
[verdict](#concept-coverage-verdict) and its evidence, not by its score.

## concept: scene

`kind: concept` · `reader: anyone` · `links:` [concept: allocation](#concept-allocation), [artefact: index directory](#artefact-index-directory)

A scene is one area of the corpus. By default each top-level directory of the root is a
scene, named after the directory, and the files directly in the root form the scene
`(root)`.

`.spraypaint/scenes.toml` overrides this:

```toml
[scenes]
core = ["crates/core", "crates/s-entropy"]
docs = ["docs", "README.md"]
```

Each entry maps a scene name to path prefixes relative to the root. A file joins the
first scene, in alphabetical order of scene names, that has a prefix equal to its path or
containing it. A file matching no prefix joins `(unscoped)`: the override labels the tree,
it does not exclude anything. Scenes are fixed at index time, so re-run `spraypaint index`
after editing the file. `spraypaint scenes` lists them.

## concept: allocation

`kind: concept` · `reader: anyone reading results, maintainers` · `links:` [concept: scene](#concept-scene), [concept: scoring](#concept-scoring)

The budget `-k` (default 12) is the total number of passages an ask may return, across all
scenes. It is shared by **water-filling**: one threshold `p*`, the clearing price, is found
such that the passages scoring above `p*` are returned, in whichever scene they lie, and
their count fits the budget. A dense scene gets many slots only when its passages
out-score the others'. It cannot fill the budget with its tenth-best passage while another
scene's best is left out.

How to read it:

- `p* = 0` means the budget was **not** used up: every scoring passage was returned, and
  raising `-k` would add nothing.
- `p* > 0` means passages were left out. Raising `-k` lowers `p*` and returns more.
- Per scene, `allocated` is how many passages it received and `available` is how many of
  its passages scored above zero, after overlapping windows were dropped. `available` is
  not the scene's size; `spraypaint scenes` gives that.
- `p*` is an output of the allocation. Nothing accepts a price as input.

Results are listed by scene, then by descending score. `--flat` lists the same passages in
one descending order; it changes the presentation, not which passages are returned.

The design is Algorithm 1 of `docs/sources/split-attention-agents.tex`.

## concept: coverage verdict

`kind: concept` · `reader: everyone reading a result` · `links:` [procedure: agent protocol](#procedure-agent-protocol), [rule: what an agent may conclude](#rule-what-an-agent-may-conclude)

Every ask ends with a verdict on whether the corpus covers the query, computed from the
index alone, with no model involved:

| Verdict | Condition |
|---|---|
| `covered` | One returned passage contains every query term. |
| `partial` | Returned passages together carry at least half the query's weight, but no single passage contains every term. |
| `declined` | Returned passages together carry less than half the query's weight, or the query has no searchable terms. |

A term's **weight** is its IDF (see [concept: scoring](#concept-scoring)). A term the corpus
has never seen (`df = 0`) gets the highest weight possible, because the absent word is
usually the one the query is about. `weight_share` is the weight of the terms found in at
least one returned passage, divided by the weight of all the query's terms.

For each term the verdict reports `df`, its weight and `in_results`. The reason names the
missing terms, heaviest first, in two groups:

- **not in the corpus** (`df = 0`): no indexed file contains the word. Another source, or
  another form of the word, is needed.
- **in the corpus but not in any returned passage** (`df > 0`, `in_results: false`): the
  word exists but fell outside the budget or the selected scenes. Raising `-k` or changing
  `--scenes` may help.

Worked example. Asked `suitable dentist in Greifswald` over a mailbox where a colleague
writes often about the city and no one mentions a dentist, the top passage is the
colleague's mail, and the verdict is:

```
declined — returned passages carry 19% of the query's weight; not in the corpus: suitable, dentist
```

The passage is real and so is the match. The verdict says the match is on the wrong word.

What the verdict does **not** say:

- `covered` does not mean answered. The words occur together; whether the passage answers
  the question is for the reader to judge.
- `declined` does not mean the fact is false or does not exist. It means these words are not
  in what this corpus returned.
- The threshold of one half (`decline_below` in JSON) is a convention, not a derived bound.
  `weight_share` is reported so a caller can apply its own.

## concept: evidence

`kind: concept` · `reader: anyone reading results` · `links:` [concept: passage](#concept-passage)

The evidence of a passage is the run of at most **5** lines within it where the query terms
are densest. Every 5-line window of the passage is scored by the total weight of the
distinct query terms it contains. The best window wins, with ties going to the window with
more matching lines and then to the earliest. The winner is centred on its matching lines,
and blank lines are trimmed from both ends. Lines longer than 200 characters are cut and
end in `…`.

Evidence is read from disk at ask time. If the file changed after indexing so that no line
still matches, the evidence falls back to the passage's first non-empty line. It then shows
where the passage is, not why it matched; `matched_terms`, which comes from the index,
still says what it was scored on. Re-index after edits.

## concept: committed act

`kind: concept` · `reader: agents, auditors` · `links:` [concept: invariants](#concept-invariants), [command: count](#command-count)

Every ask without `--dry-run` is a **committed act**: after the search and before the
result is printed, the counter in `.spraypaint/count` goes up by one. The counter never goes
down. No command resets or decrements it, it survives re-indexing and restarts, and
concurrent asks are serialised so no increment is lost.

A dry run (`--dry-run`, or `/api/dry-run`) commits nothing. Its JSON still carries the
verdict and the passages, so exploring costs nothing. The count therefore records answers
that were delivered, not searches that were tried. Commit the ask whose result you use, and
preview everything else.

## concept: invariants

`kind: concept` · `reader: maintainers, auditors` · `links:` [command: verify](#command-verify)

spraypaint implements four invariants from the Split-Attention Synchronised Agents
calculus (`docs/sources/split-attention-agents.tex`). `spraypaint verify` checks them.

1. **Conserved identity.** The index is a graph: one vertex per file (keyed by its content
   hash), and an edge between every pair of files, weighted by the words they share plus a
   fixed floor of `1e-6`. Its fingerprint is a blake3 digest that does not change when files
   are reordered, renumbered or regrouped. Every load recomputes the fingerprint and refuses
   an index whose stored fingerprint differs. χ (chi) is the graph's minimum cut. Because
   every edge carries the floor, `χ ≥ floor` holds by construction and says nothing about
   the corpus.
2. **Never-resetting count.** See [concept: committed act](#concept-committed-act).
3. **Search, not fetch.** The index stores term counts and line ranges, never text. Every
   ask is a new search, and every line shown is read from disk.
4. **Exclusive phases.** Building the index holds an exclusive lock on
   `.spraypaint/.lock`; asks hold a shared lock. An ask never sees a half-written index, and
   an index build waits for running asks.

## command: index

`kind: command` · `reader: anyone` · `links:` [concept: corpus and root](#concept-corpus-and-root), [concept: passage](#concept-passage)

`spraypaint index [--root DIR] [--json] [--dry-run] [--window N] [--overlap N]`

Walks the corpus, cuts files into passages, and writes `.spraypaint/index.json` by writing
a temporary file and renaming it. It holds the exclusive phase lock for the whole build.
Progress goes to stderr and the summary to stdout.

- `--dry-run` reports how many files would be indexed and writes nothing.
- `--window N` (default 40) and `--overlap N` (default 10) set the passage size in lines.
- `--json` prints `{root, documents, passages, scenes, identity_fingerprint}`, or
  `{root, would_index}` for a dry run.

Re-run it after files change. There is no incremental update; each run rebuilds the whole
index.

## command: ask

`kind: command` · `reader: anyone` · `links:` [artefact: ask output](#artefact-ask-output), [concept: coverage verdict](#concept-coverage-verdict), [concept: allocation](#concept-allocation)

`spraypaint ask "<query>" [--root DIR] [-k N] [--json] [--dry-run] [--flat] [--scenes a,b]`

| Flag | Default | Effect |
|---|---|---|
| `-k N`, `--budget N` | 12 | total passages to return across all scenes. Use `N ≥ 1`; `0` returns nothing. |
| `--json` | off | machine output; see [artefact: ask output](#artefact-ask-output) |
| `--dry-run` | off | commit nothing. Human output shows the verdict and the allocation but no passages; JSON output includes the passages. |
| `--flat` | off | list the results in one descending order instead of by scene |
| `--scenes a,b` | all | search only these scenes, by exact name |

It loads the index (verifying its fingerprint), searches, and, unless `--dry-run` is given,
commits one act before printing. Quote a query that contains spaces.

## command: identity

`kind: command` · `reader: auditors` · `links:` [concept: invariants](#concept-invariants)

`spraypaint identity [--root DIR] [--json]` prints the fingerprint, χ, the floor, and the
vertex and edge counts of the index graph. The JSON keys are `fingerprint`,
`char_invariant`, `floor`, `n_vertices` and `n_edges`. The fingerprint changes when the
corpus content changes, so it identifies which index an answer came from.

## command: count

`kind: command` · `reader: agents, auditors` · `links:` [concept: committed act](#concept-committed-act)

`spraypaint count [--root DIR] [--json]` prints `committed acts: N`, or
`{"committed_count": N}`. It reads 0 when nothing has been committed yet.

## command: scenes

`kind: command` · `reader: anyone` · `links:` [concept: scene](#concept-scene)

`spraypaint scenes [--root DIR] [--json]` lists each scene with its file and passage counts.
The JSON form is an array of `{name, documents, passages}`. Use the names with
`ask --scenes`. It fails with `no index at …` when there is no index, which makes it a
cheap way to check that one exists.

## command: verify

`kind: command` · `reader: maintainers, CI` · `links:` [concept: invariants](#concept-invariants), [artefact: verify output](#artefact-verify-output)

`spraypaint verify [--root DIR] [--json] [--allow-degenerate]` re-checks the four
invariants. Each check is `PASS`, `FAIL`, or `N/A` (it could not run, or could only pass
vacuously).

| Exit | Meaning |
|---|---|
| 0 | every check passed, none was N/A, and the corpus is not degenerate |
| 1 | at least one check failed: an invariant is breached |
| 2 | nothing failed, but a check was N/A or the corpus is degenerate |

`--allow-degenerate` turns exit 2 into 0. It never hides a failure.

The checks, by stable name, are `fingerprint`, `chi_floor`, `stored_fields` and
`schema_version` (invariant 1), `count_readable` (2), `no_answer_cache` (3) and
`lock_operational` (4). A corpus is **degenerate**, and its passes are not evidence, when it
has no files, one file, no two files sharing any word, or fewer than two scenes. A fresh
index with nothing committed yet also exits 2, because `count_readable` has nothing to
check.

## command: serve

`kind: command` · `reader: people running the UI, tools calling the HTTP API` · `links:` [interface: HTTP API](#interface-http-api), [procedure: pairing a hosted UI](#procedure-pairing-a-hosted-ui), [limit: disclosure and security](#limit-disclosure-and-security)

`spraypaint serve [--root DIR] [--port N] [--host IP] [--open] [--allow-index] [--no-cache] [--pair ORIGIN] [--i-know-this-is-public]`

Serves the embedded web UI at `/` and the JSON API under `/api/`, with 4 worker threads.

| Flag | Default | Effect |
|---|---|---|
| `--port N` | 7373 | port to bind. A busy port is an error; the server never picks another. |
| `--host IP` | 127.0.0.1 | address to bind. A non-loopback address needs `--i-know-this-is-public`. |
| `--open` | off | open the UI in a browser |
| `--allow-index` | off | allow `POST /api/index` to rebuild the index |
| `--no-cache` | off | reload `index.json` on every request instead of caching it |
| `--pair ORIGIN` | none | let a hosted copy of the UI at ORIGIN call the API with a printed token |

The index is cached by the modification time and length of `index.json`, and re-verified
whenever either changes. A build made without the UI still serves the API, and `/` then
returns a plain-text notice.

## interface: HTTP API

`kind: interface` · `reader: programs, agents` · `links:` [command: serve](#command-serve), [artefact: ask output](#artefact-ask-output)

All responses are JSON. The ask endpoints produce byte-for-byte the same JSON as
`spraypaint ask --json`.

| Method and path | Does | Commits |
|---|---|---|
| `GET /api/health` | `{ok, version, root, has_index, allow_index, indexing}`. Takes no lock, so it answers during a build. | no |
| `GET` or `POST /api/ask` | one committed ask | **yes** |
| `GET` or `POST /api/dry-run` | the same search, with no commit | no |
| `GET /api/identity` | as `identity --json` | no |
| `GET /api/count` | `{committed_count}` | no |
| `GET /api/scenes` | as `scenes --json` | no |
| `GET /api/verify` | as `verify --json`; always HTTP 200, even when a check fails | no |
| `POST /api/index` | start a background build; 202 `{status: "running"}`. Needs `--allow-index`. | no |
| `GET /api/index/status` | `{status, running, detail}`; `status` is `idle`, `running`, `done` or `failed` | no |

**Ask parameters.** As a query string: `q` (or `query`; required, not empty), `k` (or
`budget`; default 12, raised to at least 1) and `scenes` (comma-separated). Or as a JSON
body: `{"query": "...", "budget": 12, "scenes": ["docs"]}`. There is no `flat` parameter;
the order is up to the client.

**Errors** carry `{"error": {"code", "message"}}`, with these stable codes:

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `bad_request` | missing or malformed parameter or body |
| 401 | `unauthorized` | paired origin without a valid token |
| 403 | `index_disabled` | `POST /api/index` without `--allow-index` |
| 404 | `no_index` | no index yet |
| 404 | `not_found` | no such endpoint |
| 405 | `method_not_allowed` | not GET or POST |
| 409 | `identity_mismatch` | stored fingerprint does not match the content; re-index |
| 409 | `schema_mismatch` | index written by an incompatible version; re-index |
| 409 | `index_running` | a build is already running |
| 413 | `bad_request` | body larger than 64 KiB |
| 500 | `internal` | anything else |

Outside the API, the server answers 403 in plain text to a cross-origin request from any
origin it is not paired with, and 421 to a request whose `Host` is not the loopback address
it is bound to. Both guard against a web page reading the local index.

## procedure: pairing a hosted UI

`kind: procedure` · `reader: people using a deployed copy of the UI` · `links:` [command: serve](#command-serve)

A web page served from elsewhere cannot reach a local server unless it is paired with it.

1. Run `spraypaint serve --pair https://your-ui.example`. The origin must match the page's
   origin exactly: scheme, host and port, with no path.
2. The server prints a token. It exists only in that process's memory, is never written to
   disk, and dies with the server.
3. Paste the token into the page. The page sends it as `Authorization: Bearer <token>` on
   every API call. A request from the paired origin without it gets 401.

Pairing admits one origin. Every other origin is still refused, and same-origin use of the
embedded UI needs no token. Never paste the token anywhere but the paired page, and never
write it down.

## artefact: ask output

`kind: artefact` · `reader: programs, agents` · `links:` [command: ask](#command-ask), [concept: coverage verdict](#concept-coverage-verdict)

**JSON** (`ask --json`, `ask --dry-run --json`, `/api/ask`, `/api/dry-run`):

| Field | Contents |
|---|---|
| `dry_run` | `true` for a preview, `false` for a committed ask. Branch on this field, not on how the result was requested. |
| `coverage.verdict` | `covered`, `partial` or `declined` |
| `coverage.reason` | one sentence naming the terms responsible |
| `coverage.weight_share` | 0–1: the share of the query's weight found in returned passages |
| `coverage.decline_below` | the threshold for `declined`, currently 0.5 |
| `coverage.terms[]` | per query term, in query order: `term`, `df` (files containing it; 0 = never seen), `weight`, `in_results` |
| `query_terms` | the terms searched, after stopword removal and de-duplication |
| `budget` | `k` as requested |
| `price` | `p*`, the clearing price; see [concept: allocation](#concept-allocation) |
| `committed_count` | the count after this act. **Absent** on a dry run, not zero. |
| `identity_fingerprint` | the fingerprint of the index that answered |
| `allocation[]` | per scene: `scene`, `allocated`, `available`, `best_score`, `median_score` (both over every scoring passage of the scene; `null` if none scored) |
| `results[]` | per passage: `scene`, `path` (relative, `/`-separated), `start_line` and `end_line` (the passage, 1-based inclusive), `score`, `snippet` (the evidence lines, joined by `\n`), `evidence_start_line`, `evidence_end_line`, `matched_terms` |

**Human** (`ask` without `--json`):

```
declined — returned passages carry 19% of the query's weight; not in the corpus: suitable, dentist
  suitable             weight  2.08  not in corpus
  dentist              weight  2.08  not in corpus
  greifswald           weight  0.98  in 1 doc(s), returned

1 passage(s)  |  price p*=0.0000  |  committed act #2  |  b3:7249f01b8cb

[mail]
  mail/boss.md:1-3  (score 1.660)  matched: greifswald
      1 │ # Greifswald
      2 │
      3 │ Where to register in Greifswald, where to get a bike repaired in Greifswald, …
```

The verdict comes first, and the per-term table follows it only when the verdict is not
`covered`. Then come a header line (passage count, `p*`, the committed act, the start of the
fingerprint), scene headings in brackets, and for each passage its location, score and
matched terms above its numbered evidence lines. `--dry-run` without `--json` prints the
verdict, the query terms, `p*` and each scene's `allocated/available`, but no passages.

## artefact: verify output

`kind: artefact` · `reader: programs, CI` · `links:` [command: verify](#command-verify)

`verify --json` and `/api/verify`:

| Field | Contents |
|---|---|
| `pass` | `true` only when `overall` is `PASS` |
| `overall` | `PASS`, `FAIL` or `N/A` |
| `degeneracies[]` | one sentence per degenerate condition found |
| `inv1_identity`, `inv2_count`, `inv3_search_not_fetch`, `inv4_phases` | each `{status, pass, checks[]}`, with checks `{name, status, detail}`; `/api/verify` adds `title` |

An invariant's status is its worst check. Read `detail` verbatim: it states, for example,
that `chi_floor` holds by construction and what fraction of edges sit at the floor.

## artefact: index directory

`kind: artefact` · `reader: anyone` · `links:` [concept: corpus and root](#concept-corpus-and-root)

`<root>/.spraypaint/` holds:

| File | Contents |
|---|---|
| `index.json` | the index: schema version, root, build time, scenes, each file's content hash and each passage's line range, length and term counts, the vocabulary, document frequencies, and the identity block. No file text. |
| `count` | the committed count, one integer |
| `.lock` | the phase lock (invariant 4) |
| `.count.lock` | serialises counter increments |
| `scenes.toml` | optional scene override, written by you |

Everything except `scenes.toml` is a cache. Keep it out of version control, and keep the
scene list reviewable:

```gitignore
.spraypaint/*
!.spraypaint/scenes.toml
```

`index.json` is pretty-printed JSON. On a corpus of 257 files and 2,883 passages it is
21.7 MB.

## rule: what an agent may conclude

`kind: rule` · `reader: AI agents` · `links:` [concept: coverage verdict](#concept-coverage-verdict), [limit: lexical matching](#limit-lexical-matching)

1. A result means only that the passage contains the matched terms. Quote the evidence, and
   do not turn a match into a claim about what the passage asserts until you have read it.
2. `covered` means the words occur together, not that the question is answered.
3. `declined` means the corpus does not contain what the query names, **in those words**.
   Before reporting absence, try the word's other forms. Even then, report "this corpus
   does not mention X", never "X does not exist" or "X is false".
4. `not returned` is not absence. The term is in the corpus; widen `-k` or the scenes.
5. A higher score is a better lexical match within this query, nothing more.
6. An empty result never proves that something is not in the files. Paths, file names,
   unindexed extensions, ignored files and files over 2 MiB are invisible. Fall back to
   `grep` before concluding anything.
7. When the answer matters later, name the index you searched by its
   `identity_fingerprint`; it changes when the corpus changes.

## failure: errors and exits

`kind: failure` · `reader: anyone` · `links:` [command: verify](#command-verify)

A failing command prints `Error:` and a cause to stderr and exits 1 (`verify` has its own
codes). The causes:

| Message begins | Cause | Fix |
|---|---|---|
| `no index at …` | no `index.json` at the root | `spraypaint index` |
| `index schema version mismatch` | written by an incompatible version | `spraypaint index` |
| `identity fingerprint mismatch` | the index was edited or corrupted | `spraypaint index` |
| `path does not exist` | bad `--root` | fix the path |
| `refusing to bind` | non-loopback `--host` without `--i-know-this-is-public` | bind loopback, or accept the exposure |
| `cannot bind` | the port is in use | pass another `--port` |
| `--pair …:` | the origin given to `--pair` is malformed | give `scheme://host[:port]` |

An ask that finds nothing is not an error. It exits 0 with a `declined` verdict and
`no matching passages.`

## limit: lexical matching

`kind: limit` · `reader: anyone` · `links:` [procedure: writing a query](#procedure-writing-a-query), [concept: term](#concept-term)

spraypaint matches words, not meanings.

- A passage that says the same thing in different words is not found. `car` does not find
  `automobile`, and `rank` does not find `ranking`.
- A passage that uses the query's words about something else is found, and may be ranked
  first. The verdict catches the case where the *specific* words are missing, not the case
  where the right words are used in another sense.
- IDF is global, so a word common across the whole corpus scores low even inside the one
  scene where it is decisive.
- There is no phrase search: `water filling` matches `water` and `filling` anywhere in the
  same 40 lines, in any order.

For an identifier, spraypaint shows the lines around it, but `grep` gives the exact line
faster. For a concept, it finds the passages that use your words for it and misses those
that use other words.

## limit: scale and cost

`kind: limit` · `reader: agents planning many queries, people with large trees` · `links:` [command: serve](#command-serve)

Every CLI ask loads the whole `index.json`, recomputes its fingerprint, scores every
passage, and reads the files it returns. On a corpus of 257 files, 2,883 passages and
27,055 distinct words (a 21.7 MB index), one ask took 3.8 seconds with a release build,
most of it spent loading and verifying the index.

- For many queries, run `spraypaint serve` once and call `/api/dry-run` and `/api/ask`. The
  server caches the verified index between requests, so each ask skips the reload.
- Each `spraypaint index` rebuilds the index in full; there is no incremental mode.
- Scoring time grows linearly with the number of passages.

These figures are one measurement on one machine, not a benchmark.

## limit: disclosure and security

`kind: limit` · `reader: anyone, agents especially` · `links:` [concept: corpus and root](#concept-corpus-and-root), [command: serve](#command-serve)

spraypaint prints file contents. Whatever is indexed can appear in a result.

- Hidden directories and `.json`, `.toml` and `.yaml` files are indexed, so local settings
  files that hold keys or tokens are in the corpus unless excluded. Exclude them with a
  `.ignore` file, and keep secrets outside any indexed tree.
- Do not send credential-shaped words through an ask on such a tree. A dry run's JSON also
  carries passages, so the same care applies to dry runs whose output a program reads.
- `serve` has no authentication of its own. It is safe on loopback because of its origin
  and `Host` checks. Binding any other address publishes the tree's content to whoever can
  reach the port, and lets them raise a count that can never be lowered. The refusal to
  bind without `--i-know-this-is-public` is deliberate.
- A pairing token grants API access to its origin for as long as the server runs.

## limit: versions

`kind: limit` · `reader: agents meeting an unknown build` · `links:` [procedure: agent protocol](#procedure-agent-protocol)

Several builds are in circulation, and the version number alone does not settle what a
build can do. Detect features instead:

| Feature | How to detect it | If absent |
|---|---|---|
| Coverage verdict, evidence lines, `matched_terms`, overlap removal | a `coverage` field in `ask --json` | the snippet is the first non-empty line of a 40-line window and there is no verdict: treat every result as a possible look-alike |
| `verify` exit 2, `--allow-degenerate`, `overall`, `degeneracies[]` | `spraypaint --version` reports 0.2.0 or later | `verify` exits 0 on a degenerate corpus |
| `ask --dry-run --json` honoured | the output parses as JSON | the dry run printed human text |

This document describes the source tree of 2026-09-30: version 0.2.0, including the
coverage verdict.

## verification: this document

`kind: verification` · `reader: anyone` · `links:` [procedure: writing a query](#procedure-writing-a-query)

This file was checked with the tool it describes. It was placed alone in an empty git
repository, indexed, and asked the questions an agent would bring to it, each written as
keywords (`ask --dry-run --json -k 3`). A question passes when its top result's evidence
lies inside the node that answers it. Two questions about things this document does not
discuss were asked as well; they pass when the verdict is `declined`.

Results at the time of writing (spraypaint source of 2026-09-30):

| Query | Top result's node | Verdict |
|---|---|---|
| `stopwords question words keywords` | procedure: writing a query | partial |
| `clearing price raising budget` | concept: allocation | covered |
| `exit degenerate allow` | command: verify | covered |
| `hidden directories gitignore ignore` | concept: corpus and root | covered |
| `stemming camel UAX` | concept: term | partial |
| `pairing token bearer origin` | procedure: pairing a hosted UI | covered |
| `counter decrement committed` | concept: committed act | declined |
| `windows overlap boundary lines` | concept: passage | covered |
| `declined weight_share threshold` | concept: coverage verdict | covered |
| `unauthorized schema_mismatch codes` | interface: HTTP API | partial |
| `evidence centred trimmed` | concept: evidence | covered |
| `cached many queries seconds` | limit: scale and cost | partial |
| `credential secrets settings keys` | limit: disclosure and security | covered |
| `fingerprint relabelling cut floor` | concept: invariants | declined |
| `scenes.toml unscoped prefixes` | concept: scene | covered |
| `detect feature builds version` | limit: versions | covered |
| `absence exist false report` | rule: what an agent may conclude | covered |
| `kubernetes helm chart` (off-topic) | — | declined |
| `invoice refund payment` (off-topic) | — | declined |

All 17 questions reached the answering node, and both off-topic questions were declined.

- **One question failed on the first draft.** `cached many queries seconds` reached the
  agent protocol instead of the scale node, because the scale node said "keeps the index"
  and "3.8 s" rather than "caches" and "seconds". The node was reworded. That is the
  procedure this tool supports for a document: a node that a fair question cannot reach
  does not say, in its reader's words, what it is about.
- **Two correct hits were declined, and were left that way.** `decrement` is absent (the
  text says "decrements"), and so is `relabelling` (the text says "reordered, renumbered or
  regrouped"). The right passage ranked first anyway, on the other terms. These show the
  verdict judging words, not meaning, exactly as
  [limit: lexical matching](#limit-lexical-matching) and rule 3 of
  [rule: what an agent may conclude](#rule-what-an-agent-may-conclude) say. Rewording the
  text to make them pass would have hidden the limit instead of recording it.

- **This node must be removed before the check is repeated.** It quotes every query, so
  when it is indexed it becomes the best match for all of them, off-topic ones included:
  with it left in, 8 of the 17 questions and both off-topic questions landed here, and the
  off-topic questions read `covered`. The same thing happens in any corpus that holds notes
  of past queries; the notes will match the queries they record.

To repeat the check, copy this file, cut at the heading of this node, into a directory
containing an empty `.git`. Run `spraypaint index --root <dir>`, then ask each query above
with `spraypaint ask "<query>" --dry-run --json -k 3 --root <dir>`. The top result's
`evidence_start_line` should fall inside the node named in the table.
