"use client";

// The passages the binary returned. Nothing like this existed before, and it is
// the main thing `spraypaint ask` actually produces.

import type { AskQuery, AskResponse, AskResult } from "@/lib/api";

function Row({ r }: { r: AskResult }) {
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
        <span className="ml-auto tabular-nums text-neutral-500">{r.score.toFixed(3)}</span>
      </div>
      {/* Snippets are file contents re-read from disk at query time. Rendered as
          text, never as HTML — this is the one place arbitrary repo content
          reaches the page. */}
      <pre className="mt-1.5 overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-neutral-300">
        {r.snippet || <span className="italic text-neutral-600">(empty line)</span>}
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
      <div className="flex h-full flex-col items-center justify-center gap-2 p-8 text-center text-sm text-neutral-500">
        <div>No passages matched.</div>
        {result.query_terms.length === 0 ? (
          <div className="text-xs">
            Every word was a stopword — the binary had no terms left to search for.
          </div>
        ) : (
          <div className="font-mono text-xs">terms: {result.query_terms.join(", ")}</div>
        )}
      </div>
    );
  }

  // `flat` re-sorts globally. It is applied here rather than server-side because
  // the binary returns the same passages either way — only the presentation
  // changes, so re-requesting would spend a round trip for identical bytes.
  const rows = query.flat
    ? [...result.results].sort((a, b) => b.score - a.score)
    : result.results;

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
      <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-neutral-800 bg-neutral-950/95 px-4 py-2 text-xs backdrop-blur">
        <span className="text-neutral-300">
          {result.results.length} passage{result.results.length === 1 ? "" : "s"}
        </span>
        <span className="text-neutral-600">·</span>
        <span className="font-mono text-neutral-400">
          p*={result.price.toFixed(4)}
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

      {query.flat
        ? rows.map((r, i) => <Row key={`${r.path}:${r.start_line}:${i}`} r={r} />)
        : grouped.map(([scene, items]) => (
            <div key={scene}>
              <div className="border-b border-neutral-800 bg-neutral-900/40 px-4 py-1 font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                {scene} · {items.length}
              </div>
              {items.map((r, i) => (
                <Row key={`${r.path}:${r.start_line}:${i}`} r={r} />
              ))}
            </div>
          ))}
    </div>
  );
}
