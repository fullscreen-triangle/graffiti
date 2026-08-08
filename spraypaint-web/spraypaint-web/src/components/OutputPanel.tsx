"use client";

// Three tabs, down from eight. The five removed ones (Agent Graph, Trajectory,
// Catalysts, Coherence, Closure) drew `.grf`-calculus quantities the binary does
// not compute and no endpoint returns.

import { useState } from "react";

import type { AskQuery, AskResponse, Identity, SceneInfo, VerifyResponse } from "@/lib/api";
import InvariantTab from "./tabs/InvariantTab";
import ReportTab from "./tabs/ReportTab";
import SceneAllocationTab from "./tabs/SceneAllocationTab";

const TABS = [
  { id: "scenes", label: "Allocation" },
  { id: "invariants", label: "Invariants" },
  { id: "report", label: "Report" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function OutputPanel({
  result,
  query,
  identity,
  scenes,
  verify,
  count,
  stale,
  running,
  onBudget,
  onReverify,
}: {
  result: AskResponse | null;
  query: AskQuery;
  identity: Identity | null;
  scenes: SceneInfo[];
  verify: VerifyResponse | null;
  count: number | null;
  stale: boolean;
  running: boolean;
  onBudget: (k: number) => void;
  onReverify: () => void;
}) {
  const [activeTab, setActiveTab] = useState<TabId>("scenes");

  return (
    <div className="flex h-full flex-col border-l border-neutral-800 bg-neutral-950">
      <div className="flex h-[35px] shrink-0 items-center border-b border-neutral-800 bg-neutral-900/60">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`h-full border-b-2 px-3 text-xs transition-colors ${
              activeTab === tab.id
                ? "border-sky-600 bg-neutral-950 text-neutral-200"
                : "border-transparent text-neutral-500 hover:text-neutral-300"
            }`}
          >
            {tab.label}
            {/* Invariants and Report describe the index, not the pending query,
                so a changed query does not make them stale. */}
            {stale && tab.id === "scenes" && (
              <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-amber-500" />
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden">
        {activeTab === "scenes" && (
          <SceneAllocationTab
            result={result}
            scenes={scenes}
            budget={query.budget}
            stale={stale}
            onBudget={onBudget}
          />
        )}
        {activeTab === "invariants" && (
          <InvariantTab verify={verify} onRefresh={onReverify} busy={running} />
        )}
        {activeTab === "report" && (
          <ReportTab
            result={result}
            query={query}
            identity={identity}
            scenes={scenes}
            verify={verify}
            count={count}
          />
        )}
      </div>
    </div>
  );
}
