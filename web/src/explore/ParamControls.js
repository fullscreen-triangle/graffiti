// The writable half of the crossfilter: these controls don't filter an
// existing dataset, they change real inputs to the Graffiti interpreter
// (floor, per-catalyst power) and trigger a fresh run. The resulting
// ContactRow log replaces the crossfilter's rows entirely, and every
// linked view redraws from the new data.

import React from "react";

function Slider({ label, value, min, max, step, onChange, disabled }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-white/60 font-mono">
      <span>
        {label}: <span className="text-white/90">{value.toFixed(3)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-sky-400"
      />
    </label>
  );
}

export function ParamControls({ params, onChange, running }) {
  const set = (key) => (value) => onChange({ ...params, [key]: value });

  return (
    <div className="flex flex-col gap-3 border border-white/10 rounded-md p-3 min-w-[220px]">
      <span className="text-xs text-white/40 font-mono">
        parameters {running ? "(re-running...)" : ""}
      </span>
      <Slider label="floor" value={params.floor} min={0.005} max={0.2} step={0.005} onChange={set("floor")} disabled={running} />
      <Slider
        label="web_search power"
        value={params.webSearchPower}
        min={0}
        max={1}
        step={0.05}
        onChange={set("webSearchPower")}
        disabled={running}
      />
      <Slider
        label="local_notes power"
        value={params.localNotesPower}
        min={0}
        max={1}
        step={0.05}
        onChange={set("localNotesPower")}
        disabled={running}
      />
      <Slider
        label="archive_lookup power"
        value={params.archiveLookupPower}
        min={0}
        max={1}
        step={0.05}
        onChange={set("archiveLookupPower")}
        disabled={running}
      />
    </div>
  );
}
