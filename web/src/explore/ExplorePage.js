// Top-level explore view: parameter controls that re-run a fixed .grf
// demo program (writable), feeding a crossfilter instance over the
// resulting ContactRow log, whose linked views (bar charts + network
// graph) redraw together on any filter or re-run.

import React, { useState, useEffect, useCallback, useRef } from "react";
import { CrossfilterProvider } from "./CrossfilterContext";
import { BarGroupChart } from "./BarGroupChart";
import { ContactNetworkView } from "./ContactNetworkView";
import { ParamControls } from "./ParamControls";
import { runExploreProgram, DEFAULT_PARAMS } from "./runExploreProgram";

export default function ExplorePage() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [rows, setRows] = useState([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [outcomeSummary, setOutcomeSummary] = useState(null);
  const runIdRef = useRef(0);

  const runNow = useCallback(async (nextParams) => {
    const runId = ++runIdRef.current;
    setRunning(true);
    setError(null);
    try {
      const { rows: newRows, results } = await runExploreProgram(nextParams);
      if (runId !== runIdRef.current) return; // a newer run superseded this one
      setRows(newRows);
      const summary = [...results.entries()]
        .map(([name, v]) => `${name}: ${v.kind}${v.kind === "Claim" ? ` -> ${v.targetClaim}` : ""}`)
        .join("  |  ");
      setOutcomeSummary(summary);
    } catch (err) {
      if (runId !== runIdRef.current) return;
      setError(String(err && err.message ? err.message : err));
    } finally {
      if (runId === runIdRef.current) setRunning(false);
    }
  }, []);

  useEffect(() => {
    runNow(params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleParamsChange = (next) => {
    setParams(next);
    runNow(next);
  };

  return (
    <div className="min-h-screen w-full bg-black px-6 py-10 sm:px-3 text-white">
      <div className="mx-auto max-w-6xl flex flex-col gap-6">
        <div>
          <h1 className="text-lg font-mono text-white/80">explore</h1>
          <p className="text-xs text-white/40 font-mono max-w-2xl">
            Move a parameter; the .grf program re-runs for real and every view below redraws from the
            new committed-contact log. Click a bar to filter; every other view (including the network)
            reflects the shared filter, crossfilter-style.
          </p>
        </div>

        <div className="flex gap-6 flex-wrap">
          <ParamControls params={params} onChange={handleParamsChange} running={running} />
          <div className="flex flex-col gap-1 justify-center font-mono text-xs text-white/60 max-w-md">
            <span className="text-white/40">outcome</span>
            {error ? <span className="text-red-400">{error}</span> : <span>{outcomeSummary}</span>}
          </div>
        </div>

        <CrossfilterProvider rows={rows}>
          <div className="flex gap-6 flex-wrap">
            <BarGroupChart title="contacts by seek" dimKey="seekName" groupKey="bySeek" />
            <BarGroupChart title="contacts by catalyst" dimKey="catalystName" groupKey="byCatalyst" />
            <BarGroupChart title="contacts by power (binned)" dimKey="power" groupKey="powerHistogram" />
          </div>
          <ContactNetworkView />
        </CrossfilterProvider>
      </div>
    </div>
  );
}
