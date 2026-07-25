"use client";

import React from "react";
import { ExecutionState } from "@/lib/engine";

interface Props {
  state: ExecutionState | null;
}

function StatusCard({
  label,
  value,
  detail,
  ok,
}: {
  label: string;
  value: string;
  detail: string;
  ok: boolean;
}) {
  return (
    <div className="bg-[#252526] border border-[#333333] rounded p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[12px] text-[#858585] uppercase tracking-wider font-semibold">
          {label}
        </span>
        <span
          className={`w-3 h-3 rounded-full ${ok ? "bg-[#89d185]" : "bg-[#f14c4c]"}`}
        />
      </div>
      <span className="text-[18px] font-mono text-[#cccccc] truncate">{value}</span>
      <span className="text-[11px] text-[#858585]">{detail}</span>
    </div>
  );
}

export default function InvariantTab({ state }: Props) {
  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-[#858585] text-sm">
        Run a script to see invariant status
      </div>
    );
  }

  const inv = state.invariants;

  return (
    <div className="h-full overflow-auto p-4">
      <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
        <StatusCard
          label="Inv 1 · Conserved Identity"
          value={inv.identityHash}
          detail={`χ (min-cut) = ${inv.chi.toFixed(4)} — ${inv.identityValid ? "hash matches on reload" : "MISMATCH"}`}
          ok={inv.identityValid}
        />
        <StatusCard
          label="Inv 2 · Never-Resetting Count"
          value={`M = ${inv.committedCount}`}
          detail={`${inv.countHistory.length} committed acts — monotone ✓`}
          ok={inv.committedCount > 0}
        />
        <StatusCard
          label="Inv 3 · Search-Not-Fetch"
          value={inv.searchNotFetch ? "enforced" : "VIOLATED"}
          detail="No cached answers returned — passages re-read from disk at query time"
          ok={inv.searchNotFetch}
        />
        <StatusCard
          label="Inv 4 · Exclusive Phases"
          value={inv.lastPhase}
          detail={`Index (exclusive) and ask (shared) locks never overlap — ${inv.exclusivePhases ? "no overlap detected" : "OVERLAP"}`}
          ok={inv.exclusivePhases}
        />
      </div>

      {/* count sparkline */}
      <div className="mt-6 max-w-2xl mx-auto bg-[#252526] border border-[#333333] rounded p-4">
        <div className="text-[11px] text-[#858585] uppercase tracking-wider font-semibold mb-2">
          Committed count history
        </div>
        <div className="flex items-end gap-[2px] h-[40px]">
          {inv.countHistory.map((val, i) => (
            <div
              key={i}
              className="bg-[#4fc1ff] rounded-sm"
              style={{
                width: Math.max(4, 200 / inv.countHistory.length),
                height: `${(val / Math.max(...inv.countHistory)) * 100}%`,
                opacity: 0.7,
              }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-1 text-[10px] text-[#858585]">
          <span>run 1</span>
          <span>run {inv.countHistory.length}</span>
        </div>
      </div>
    </div>
  );
}
