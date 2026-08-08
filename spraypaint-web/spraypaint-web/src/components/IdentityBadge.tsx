"use client";

// The conserved identity of the index (Inv 1), plus the two numbers a search
// produces (`p*`, budget) and the one it increments (committed count).
//
// The old InvariantTab buried χ inside a detail string and never displayed the
// floor at all — which is precisely the number that makes χ interpretable. FLOOR
// is added to every edge at construction, so χ ≥ FLOOR holds by arithmetic; a χ
// sitting *at* the floor means the graph carries no corpus signal, and a reader
// cannot tell that without seeing both.

import type { AskResponse, Identity } from "@/lib/api";

function Cell({
  label,
  value,
  title,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5" title={title}>
      <span className="text-[10px] uppercase tracking-wider text-neutral-500">{label}</span>
      <span className={`text-xs text-neutral-200 ${mono ? "font-mono tabular-nums" : ""}`}>
        {value}
      </span>
    </div>
  );
}

export default function IdentityBadge({
  identity,
  result,
  count,
}: {
  identity: Identity | null;
  result: AskResponse | null;
  count: number | null;
}) {
  if (!identity) {
    return (
      <div className="border-b border-neutral-800 px-4 py-2 text-xs text-neutral-500">
        No index loaded.
      </div>
    );
  }

  // χ at the floor is not a failure, but it is the case where χ says nothing
  // about the corpus — worth marking, not hiding.
  const atFloor = identity.char_invariant <= identity.floor * (1 + 1e-9);

  return (
    <div className="flex flex-wrap items-start gap-x-6 gap-y-3 border-b border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <Cell
        label="fingerprint"
        value={identity.fingerprint.slice(0, 16)}
        title={`${identity.fingerprint}\n\nblake3 over the canonicalised self-graph. Matches \`spraypaint identity --json\`.`}
      />
      <Cell
        label="χ (min-cut)"
        title="Characteristic invariant: the minimum cut of the document self-graph."
        value={
          <span className={atFloor ? "text-amber-400" : undefined}>
            {identity.char_invariant.toExponential(4)}
          </span>
        }
      />
      <Cell
        label="floor"
        title="A construction parameter added to every edge, not a measured bound. χ ≥ floor holds by arithmetic."
        value={
          <span className="text-neutral-400">
            {identity.floor.toExponential(1)}
            {atFloor && <span className="ml-1 text-amber-500">χ at floor</span>}
          </span>
        }
      />
      <Cell label="vertices" value={identity.n_vertices} />
      <Cell label="edges" value={identity.n_edges} />

      <div className="ml-auto flex items-start gap-x-6">
        <Cell
          label="committed acts"
          title="Monotone counter (Inv 2). Only an explicit Run increments it; previews do not."
          value={count === null ? <span className="text-neutral-600">—</span> : count}
        />
        <Cell
          label="p* (clearing price)"
          title="An OUTPUT of water-filling, found by bisection — not a settable input. No endpoint accepts a price."
          value={
            result ? (
              result.price.toFixed(4)
            ) : (
              <span className="text-neutral-600">—</span>
            )
          }
        />
        <Cell
          label="budget k"
          value={result ? result.budget : <span className="text-neutral-600">—</span>}
        />
      </div>
    </div>
  );
}
