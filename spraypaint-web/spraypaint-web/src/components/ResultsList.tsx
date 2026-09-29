"use client";

// The passages the binary returned, under the verdict on them.
//
// The verdict comes first because a result list read without it invites the
// reader to take the best look-alike as the answer. Asked for a dentist, a
// corpus with no dentist in it returns whoever writes most about the city; the
// passage is real, the match is real, and it answers nothing. The binary says
// so in `coverage`, and this is where the reader sees it.

import type { ReactNode } from "react";

import type { AskQuery, AskResponse, AskResult, Coverage } from "@/lib/api";

const VERDICT_STYLE = {
  covered: "border-emerald-800/60 bg-emerald-950/30 text-emerald-300",
  partial: "border-amber-800/60 bg-amber-950/30 text-amber-300",
  declined: "border-rose-800/60 bg-rose-950/30 text-rose-300",
} as const;

function VerdictBanner({ coverage }: { coverage: Coverage }) {
  return (
    <div className={`border-b px-4 py-2.5 text-xs ${VERDICT_STYLE[coverage.verdict]}`}>
      <div className="flex items-baseline gap-2">
        <span className="font-semibold uppercase tracking-wider">{coverage.verdict}</span>
        <span className="text-neutral-300">{coverage.reason}</span>
      </div>
      {coverage.verdict !== "covered" && coverage.terms.length > 0 && (
        // Which word is missing is the whole message of a non-covered verdict,
        // so every term is listed, not just the missing ones.
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {coverage.terms.map((t) => (
            <span
              key={t.term}
              title={`weight ${t.weight.toFixed(2)} · in ${t.df} document(s)`}
              className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
                t.in_results
                  ? "border-neutral-700 text-neutral-300"
                  : t.df === 0
                    ? "border-rose-800/70 text-rose-300 line-through decoration-rose-500/60"
                    : "border-amber-800/70 text-amber-300"
              }`}
            >
              {t.term}
              <span className="ml-1 text-neutral-500">
                {t.in_results ? "returned" : t.df === 0 ? "not in corpus" : "not returned"}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Mark whole-word, case-insensitive occurrences of the matched terms. The
// binary tokenises on Unicode word boundaries, so a term is matched only where
// it is not glued to further letters or digits — `rank` must not light up
// inside `ranking`, which is a different term to the index.
function highlight(line: string, terms: string[]): ReactNode[] {
  if (terms.length === 0 || line.length === 0) return [line];
  const re = new RegExp(
    `(?<![\\p{L}\\p{N}_])(${terms.map(escapeRegExp).join("|")})(?![\\p{L}\\p{N}_])`,
    "giu"
  );
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of line.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > last) out.push(line.slice(last, i));
    out.push(
      <mark key={i} className="rounded-sm bg-sky-500/25 px-0.5 text-sky-200">
        {m[0]}
      </mark>
    );
    last = i + m[0].length;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

function Row({ r, top }: { r: AskResult; top: number }) {
  const terms = r.matched_terms ?? [];
  const lines = r.snippet.split("\n");
  // Older binaries send one first-line snippet and no range; number it from the
  // passage start rather than invent a location for it.
  const first = r.evidence_start_line ?? r.start_line;
  const width = String(first + lines.length - 1).length;

  return (
    <div className="group border-b border-neutral-800 px-4 py-3 hover:bg-neutral-900/60">
      <div className="flex items-baseline gap-2 font-mono text-xs">
        <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-neutral-400">
          {r.scene}
        </span>
        <span className="text-sky-400">{r.path}</span>
        <span className="text-neutral-500">
          :{r.start_line}-{r.end_line}
        </span>
        <span
          className="ml-auto flex items-center gap-1.5 tabular-nums text-neutral-500"
          title={`BM25 ${r.score.toFixed(3)} — shown relative to the top result, since the raw score has no meaning across queries`}
        >
          <span className="inline-block h-1 w-12 overflow-hidden rounded bg-neutral-800">
            <span
              className="block h-full bg-sky-600"
              style={{ width: `${top > 0 ? (r.score / top) * 100 : 0}%` }}
            />
          </span>
          {top > 0 ? `${Math.round((r.score / top) * 100)}%` : "—"}
        </span>
      </div>
      {terms.length > 0 && (
        <div className="mt-1 font-mono text-[10px] text-neutral-500">
          matched: <span className="text-neutral-400">{terms.join(", ")}</span>
        </div>
      )}
      {/* File contents re-read from disk at query time, rendered as text and
          React nodes only — never HTML. This is the one place arbitrary repo
          content reaches the page. */}
      <pre className="mt-1.5 overflow-x-auto font-mono text-xs leading-relaxed text-neutral-300">
        {r.snippet ? (
          lines.map((line, i) => (
            <div key={i} className="flex">
              <span className="mr-3 shrink-0 select-none text-right text-neutral-600" style={{ width: `${width}ch` }}>
                {first + i}
              </span>
              <span className="whitespace-pre-wrap break-words">{highlight(line, terms)}</span>
            </div>
          ))
        ) : (
          <span className="italic text-neutral-600">(file unreadable since indexing)</span>
        )}
      </pre>
    </div>
  );
}

export default function ResultsList({
  result,
  query,
  stale,
}: {
  result: AskResponse | null;
  query: AskQuery;
  stale: boolean;
}) {
  if (!result) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-sm text-neutral-500">
        Enter a query and press Run.
      </div>
    );
  }

  if (result.results.length === 0) {
    return (
      <div className="flex h-full flex-col">
        {result.coverage && <VerdictBanner coverage={result.coverage} />}
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center text-sm text-neutral-500">
          <div>No passages matched.</div>
          {result.query_terms.length === 0 ? (
            <div className="text-xs">
              Every word was a stopword — the binary had no terms left to search for.
            </div>
          ) : (
            <div className="font-mono text-xs">terms: {result.query_terms.join(", ")}</div>
          )}
        </div>
      </div>
    );
  }

  // `flat` re-sorts globally. It is applied here rather than server-side because
  // the binary returns the same passages either way — only the presentation
  // changes, so re-requesting would spend a round trip for identical bytes.
  const rows = query.flat
    ? [...result.results].sort((a, b) => b.score - a.score)
    : result.results;
  const top = Math.max(...result.results.map((r) => r.score));

  const grouped: [string, AskResult[]][] = [];
  if (!query.flat) {
    for (const r of rows) {
      const last = grouped[grouped.length - 1];
      if (last && last[0] === r.scene) last[1].push(r);
      else grouped.push([r.scene, [r]]);
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="sticky top-0 z-10 bg-neutral-950/95 backdrop-blur">
        {result.coverage && <VerdictBanner coverage={result.coverage} />}
        <div className="flex items-center gap-3 border-b border-neutral-800 px-4 py-2 text-xs">
          <span className="text-neutral-300">
            {result.results.length} passage{result.results.length === 1 ? "" : "s"}
          </span>
          {result.dry_run ? (
            // A preview must never be mistaken for a committed act. This is the
            // only visual difference, and it is the whole point of the dry run.
            <span className="rounded border border-amber-700/60 bg-amber-950/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-400">
              preview · not committed
            </span>
          ) : (
            <span className="rounded border border-emerald-800/60 bg-emerald-950/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-emerald-400">
              committed act #{result.committed_count}
            </span>
          )}
          {stale && (
            <span className="ml-auto text-[10px] uppercase tracking-wide text-neutral-500">
              query changed since this ran
            </span>
          )}
        </div>
      </div>

      {query.flat
        ? rows.map((r, i) => <Row key={`${r.path}:${r.start_line}:${i}`} r={r} top={top} />)
        : grouped.map(([scene, items]) => (
            <div key={scene}>
              <div className="border-b border-neutral-800 bg-neutral-900/40 px-4 py-1 font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                {scene} · {items.length}
              </div>
              {items.map((r, i) => (
                <Row key={`${r.path}:${r.start_line}:${i}`} r={r} top={top} />
              ))}
            </div>
          ))}
    </div>
  );
}
