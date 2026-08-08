"use client";

// The four invariants, rendered verbatim from `/api/verify`.
//
// The version this replaces had three problems that all pointed the same way —
// it showed a healthier system than existed:
//
//   * Two states only (green dot / red dot). The binary reports **three**:
//     pass, fail, and n/a. Collapsing n/a into either one is the whole reason
//     the exit-2 contract exists — a degenerate index that verified nothing
//     rendered as a clean pass.
//   * `"monotone ✓"` was a hardcoded string, not a computed check.
//   * The count sparkline plotted `[1..runCount]` — a linear ramp generated in
//     `engine.ts`. `.spraypaint/count` is a single integer; no history is
//     stored anywhere, so no history can be drawn.
//
// Every string below comes from the binary. This component adds no judgement.

import type { CheckStatus, VerifyInvariant, VerifyResponse } from "@/lib/api";

const STATUS_STYLE: Record<CheckStatus, { dot: string; text: string; label: string }> = {
  pass: { dot: "bg-emerald-500", text: "text-emerald-400", label: "pass" },
  fail: { dot: "bg-red-500", text: "text-red-400", label: "fail" },
  "n/a": { dot: "bg-amber-500", text: "text-amber-400", label: "n/a" },
};

function Dot({ status }: { status: CheckStatus }) {
  return <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${STATUS_STYLE[status].dot}`} />;
}

function InvariantCard({ title, inv }: { title: string; inv: VerifyInvariant }) {
  const s = STATUS_STYLE[inv.status];
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/40">
      <div className="flex items-center gap-2 border-b border-neutral-800 px-3 py-2">
        <Dot status={inv.status} />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-300">
          {inv.title ?? title}
        </span>
        <span className={`ml-auto text-[10px] uppercase tracking-wider ${s.text}`}>
          {s.label}
        </span>
      </div>
      <div className="divide-y divide-neutral-800/60">
        {inv.checks.length === 0 ? (
          <div className="px-3 py-2 text-xs italic text-neutral-600">no sub-checks reported</div>
        ) : (
          inv.checks.map((c) => (
            <div key={c.name} className="flex gap-2 px-3 py-2">
              <Dot status={c.status} />
              <div className="min-w-0 flex-1">
                <div className="font-mono text-[11px] text-neutral-300">{c.name}</div>
                {/* Verbatim from the binary. This is where `chi_floor` says the
                    inequality holds by construction, and where a tampered
                    stored field names itself. Rewording it here would be the
                    UI second-guessing the check. */}
                <div className="mt-0.5 whitespace-pre-wrap break-words text-[11px] leading-relaxed text-neutral-500">
                  {c.detail}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default function InvariantTab({
  verify,
  onRefresh,
  busy,
}: {
  verify: VerifyResponse | null;
  onRefresh: () => void;
  busy?: boolean;
}) {
  if (!verify) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-neutral-500">
        <span>No verification report.</span>
        <button
          onClick={onRefresh}
          className="rounded border border-neutral-700 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          Run verify
        </button>
      </div>
    );
  }

  const s = STATUS_STYLE[verify.overall];
  // The exit contract, spelled out. A reader who sees an amber banner should be
  // able to predict what `spraypaint verify` would return in their shell.
  const exitCode = verify.overall === "fail" ? 1 : verify.overall === "n/a" ? 2 : 0;

  return (
    <div className="h-full overflow-y-auto">
      <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-neutral-800 bg-neutral-950/95 px-4 py-2 backdrop-blur">
        <Dot status={verify.overall} />
        <span className={`text-xs font-semibold uppercase tracking-wider ${s.text}`}>
          {verify.overall === "n/a" ? "not applicable" : verify.overall}
        </span>
        <span
          className="font-mono text-[11px] text-neutral-500"
          title="What `spraypaint verify` would exit with. 2 means nothing failed, but at least one check could not be applied — use --allow-degenerate to map it to 0."
        >
          exit {exitCode}
        </span>
        <button
          onClick={onRefresh}
          disabled={busy}
          className="ml-auto rounded border border-neutral-700 px-2 py-0.5 text-[11px] text-neutral-300 hover:bg-neutral-800 disabled:text-neutral-600"
        >
          {busy ? "…" : "Re-verify"}
        </button>
      </div>

      {verify.degeneracies.length > 0 && (
        // A degenerate index is not a failure, but a PASS on one is not
        // evidence either — this is the panel's most important line.
        <div className="border-b border-amber-900/50 bg-amber-950/20 px-4 py-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-500">
            degenerate regimes — checks below verify less than they appear to
          </div>
          <ul className="mt-1 space-y-0.5">
            {verify.degeneracies.map((d) => (
              <li key={d} className="font-mono text-[11px] text-amber-200/80">
                {d}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid gap-3 p-4 lg:grid-cols-2">
        <InvariantCard title="Inv 1 · Conserved identity" inv={verify.inv1_identity} />
        <InvariantCard title="Inv 2 · Never-resetting count" inv={verify.inv2_count} />
        <InvariantCard title="Inv 3 · Search, not fetch" inv={verify.inv3_search_not_fetch} />
        <InvariantCard title="Inv 4 · Exclusive phases" inv={verify.inv4_phases} />
      </div>
    </div>
  );
}
