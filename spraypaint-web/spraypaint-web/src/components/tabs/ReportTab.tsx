"use client";

// A report on the search that actually ran, and on the index it ran against.
//
// The version this replaces reported on a `.grf` execution — seeks, catalyst
// chains, composite power κ, closure steps — none of which the binary computes
// or the API returns. It rendered through `dangerouslySetInnerHTML` over a
// hand-rolled regex chain, which was harmless only because every value in it
// was generated locally. The moment real snippets flow in from disk, a file in
// the user's own repo containing `<img onerror=…>` becomes script execution in
// their browser. So: React nodes, no `innerHTML`, anywhere.
//
// KaTeX is gone with it. Nothing here is an equation now, and the CDN <link>
// it needed would have rendered unstyled offline — which is the only way this
// UI is ever served.

import { useMemo } from "react";

import type { AskQuery, AskResponse, Identity, SceneInfo, VerifyResponse } from "@/lib/api";
import { toCommand } from "@/lib/gestures";

function Section({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <section className="mb-7">
      <h2 className="mb-2 border-b border-neutral-800 pb-1 text-sm font-semibold text-neutral-200">
        <span className="mr-2 text-neutral-600">{n}.</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: React.ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            {head.map((h) => (
              <th
                key={h}
                className="border-b border-neutral-800 px-2 py-1 text-left font-medium uppercase tracking-wider text-[10px] text-neutral-500"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-neutral-900">
              {r.map((c, j) => (
                <td key={j} className="px-2 py-1 font-mono tabular-nums text-neutral-300">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const DASH = <span className="text-neutral-600">—</span>;

export default function ReportTab({
  result,
  query,
  identity,
  scenes,
  verify,
  count,
}: {
  result: AskResponse | null;
  query: AskQuery;
  identity: Identity | null;
  scenes: SceneInfo[];
  verify: VerifyResponse | null;
  count: number | null;
}) {
  const totals = useMemo(() => new Map(scenes.map((s) => [s.name, s.passages])), [scenes]);

  if (!result && !identity) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-500">
        No index loaded.
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-8 py-6">
        <h1 className="mb-1 text-lg font-semibold text-neutral-100">Search report</h1>
        <p className="mb-6 font-mono text-[11px] text-neutral-600">$ {toCommand(query)}</p>

        <Section n={1} title="Index">
          {identity ? (
            <Table
              head={["field", "value", "note"]}
              rows={[
                [
                  "fingerprint",
                  <span key="f" title={identity.fingerprint}>
                    {identity.fingerprint.slice(0, 24)}…
                  </span>,
                  <span key="n" className="font-sans text-neutral-500">
                    blake3 over the canonicalised self-graph
                  </span>,
                ],
                [
                  "χ (min-cut)",
                  identity.char_invariant.toExponential(6),
                  <span key="n" className="font-sans text-neutral-500">
                    minimum cut of the document self-graph
                  </span>,
                ],
                [
                  "floor",
                  identity.floor.toExponential(1),
                  // The distinction the old UI never drew, and the reason χ can
                  // look healthy on a corpus with no shared vocabulary at all.
                  <span key="n" className="font-sans text-neutral-500">
                    construction parameter added to every edge — not a measured bound
                  </span>,
                ],
                ["vertices", identity.n_vertices, ""],
                ["edges", identity.n_edges, ""],
                [
                  "committed acts",
                  count ?? DASH,
                  <span key="n" className="font-sans text-neutral-500">
                    monotone; no decrement path
                  </span>,
                ],
              ]}
            />
          ) : (
            <p className="text-xs text-neutral-500">Identity unavailable.</p>
          )}
        </Section>

        <Section n={2} title="Allocation">
          {result ? (
            <>
              <p className="mb-2 text-xs leading-relaxed text-neutral-400">
                The budget <span className="font-mono">k={result.budget}</span> is allocated
                across scenes by water-filling. A scene receives passages while its scores
                clear <span className="font-mono">p*</span>; scenes below it receive nothing.
                {" "}
                <span className="font-mono">p*</span> is an{" "}
                <span className="text-neutral-200">output</span> of that computation — found
                by bisection — not a value any endpoint accepts.
              </p>
              <p className="mb-3 font-mono text-xs text-amber-500">
                p* = {result.price.toFixed(6)}
              </p>
              <Table
                head={["scene", "allocated", "scoring", "indexed", "best", "median"]}
                rows={result.allocation.map((a) => [
                  <span key="s" className="text-sky-400">
                    {a.scene}
                  </span>,
                  a.allocated,
                  a.available,
                  totals.get(a.scene) ?? DASH,
                  a.best_score === null ? DASH : a.best_score.toFixed(4),
                  a.median_score === null ? DASH : a.median_score.toFixed(4),
                ])}
              />
              <p className="mt-2 text-[11px] leading-relaxed text-neutral-500">
                <span className="font-mono">scoring</span> counts passages scoring above zero
                for this query; <span className="font-mono">indexed</span> is the scene&apos;s
                total. They are different numbers.{" "}
                <span className="font-mono">best</span>/<span className="font-mono">median</span>{" "}
                are computed over every scoring passage before truncation, so they describe
                the scene rather than the budget.
              </p>
            </>
          ) : (
            <p className="text-xs text-neutral-500">No query has been run.</p>
          )}
        </Section>

        <Section n={3} title="Results">
          {result ? (
            <>
              <p className="mb-3 text-xs text-neutral-400">
                {result.results.length} passage{result.results.length === 1 ? "" : "s"} returned
                for terms{" "}
                <span className="font-mono text-neutral-300">
                  {result.query_terms.length ? result.query_terms.join(", ") : "(none survived stopword filtering)"}
                </span>
                {result.dry_run ? (
                  <span className="ml-2 text-amber-500">· preview, not committed</span>
                ) : (
                  <span className="ml-2 text-emerald-500">
                    · committed act #{result.committed_count}
                  </span>
                )}
                .
              </p>
              {result.results.length > 0 && (
                <Table
                  head={["scene", "path", "lines", "score"]}
                  rows={result.results.map((r) => [
                    r.scene,
                    <span key="p" className="text-sky-400">
                      {r.path}
                    </span>,
                    `${r.start_line}–${r.end_line}`,
                    r.score.toFixed(4),
                  ])}
                />
              )}
              <p className="mt-2 text-[11px] leading-relaxed text-neutral-500">
                Snippets are re-read from disk at query time and are not listed here — no
                answer is stored in the index (Inv 3).
              </p>
            </>
          ) : (
            <p className="text-xs text-neutral-500">No query has been run.</p>
          )}
        </Section>

        <Section n={4} title="Invariant conformance">
          {verify ? (
            <>
              <Table
                head={["invariant", "status", "checks"]}
                rows={[
                  ["Inv 1 · conserved identity", verify.inv1_identity.status, verify.inv1_identity.checks.length],
                  ["Inv 2 · never-resetting count", verify.inv2_count.status, verify.inv2_count.checks.length],
                  ["Inv 3 · search, not fetch", verify.inv3_search_not_fetch.status, verify.inv3_search_not_fetch.checks.length],
                  ["Inv 4 · exclusive phases", verify.inv4_phases.status, verify.inv4_phases.checks.length],
                ].map(([a, b, c]) => [
                  <span key="a" className="font-sans">
                    {a}
                  </span>,
                  <span
                    key="b"
                    className={
                      b === "pass"
                        ? "text-emerald-400"
                        : b === "fail"
                          ? "text-red-400"
                          : "text-amber-400"
                    }
                  >
                    {b}
                  </span>,
                  c,
                ])}
              />
              {verify.degeneracies.length > 0 && (
                <div className="mt-3 rounded border border-amber-900/50 bg-amber-950/20 px-3 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-500">
                    degenerate regimes
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-amber-200/70">
                    A pass under these conditions is not evidence — the check had nothing to
                    discriminate against.
                  </p>
                  <ul className="mt-1 space-y-0.5">
                    {verify.degeneracies.map((d) => (
                      <li key={d} className="font-mono text-[11px] text-amber-200/80">
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <p className="text-xs text-neutral-500">No verification report.</p>
          )}
        </Section>

        <Section n={5} title="References">
          <ol className="list-decimal space-y-1 pl-5 text-[11px] leading-relaxed text-neutral-500">
            <li>
              Sachikonye, K. F. (2026). <em>Semantic Causal Propagation: An
              Individuation-Theoretic Calculus of Boundary-Free Search.</em> Technical
              University of Munich.
            </li>
            <li>
              Stoer, M. &amp; Wagner, F. (1997). A simple min-cut algorithm.{" "}
              <em>Journal of the ACM</em>, 44(4), 585–591.
            </li>
            <li>
              Robertson, S. &amp; Zaragoza, H. (2009). The probabilistic relevance framework:
              BM25 and beyond. <em>Foundations and Trends in Information Retrieval</em>, 3(4),
              333–389.
            </li>
          </ol>
        </Section>
      </div>
    </div>
  );
}
