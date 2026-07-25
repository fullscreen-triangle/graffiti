"use client";

import React, { useState } from "react";
import { ExecutionState, ScriptDiff } from "@/lib/engine";
import ForceGraphTab from "./tabs/ForceGraphTab";
import TrajectoryTab from "./tabs/TrajectoryTab";
import CatalystTab from "./tabs/CatalystTab";
import CoherenceTab from "./tabs/CoherenceTab";
import SceneAllocationTab from "./tabs/SceneAllocationTab";
import ClosureTab from "./tabs/ClosureTab";
import InvariantTab from "./tabs/InvariantTab";
import ReportTab from "./tabs/ReportTab";

interface Props {
  state: ExecutionState | null;
  onCrossfilter: (diff: ScriptDiff) => void;
}

const TABS = [
  { id: "graph", label: "Agent Graph" },
  { id: "trajectory", label: "Trajectory" },
  { id: "catalyst", label: "Catalysts" },
  { id: "coherence", label: "Coherence" },
  { id: "scenes", label: "Scenes" },
  { id: "closure", label: "Closure" },
  { id: "invariants", label: "Invariants" },
  { id: "report", label: "Report" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function OutputPanel({ state, onCrossfilter }: Props) {
  const [activeTab, setActiveTab] = useState<TabId>("graph");

  const stale = state?.stale ?? false;

  return (
    <div className="h-full flex flex-col bg-[#1e1e1e] border-l border-[#1e1e1e]">
      {/* tab bar */}
      <div className="flex items-center bg-[#252526] border-b border-[#1e1e1e] h-[35px] shrink-0 overflow-x-auto scrollbar-thin">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 h-full text-[12px] whitespace-nowrap border-b-2 transition-colors ${
              activeTab === tab.id
                ? "text-[#cccccc] border-[#007acc] bg-[#1e1e1e]"
                : "text-[#858585] border-transparent hover:text-[#cccccc]"
            }`}
          >
            {tab.label}
            {stale && tab.id !== "invariants" && tab.id !== "report" && (
              <span className="ml-1 w-1.5 h-1.5 rounded-full bg-[#cca700] inline-block" />
            )}
          </button>
        ))}
      </div>

      {/* tab content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "graph" && <ForceGraphTab state={state} />}
        {activeTab === "trajectory" && (
          <TrajectoryTab state={state} onCrossfilter={onCrossfilter} stale={stale} />
        )}
        {activeTab === "catalyst" && (
          <CatalystTab state={state} onCrossfilter={onCrossfilter} stale={stale} />
        )}
        {activeTab === "coherence" && (
          <CoherenceTab state={state} onCrossfilter={onCrossfilter} stale={stale} />
        )}
        {activeTab === "scenes" && (
          <SceneAllocationTab state={state} onCrossfilter={onCrossfilter} stale={stale} />
        )}
        {activeTab === "closure" && (
          <ClosureTab state={state} onCrossfilter={onCrossfilter} stale={stale} />
        )}
        {activeTab === "invariants" && <InvariantTab state={state} />}
        {activeTab === "report" && <ReportTab state={state} />}
      </div>
    </div>
  );
}
