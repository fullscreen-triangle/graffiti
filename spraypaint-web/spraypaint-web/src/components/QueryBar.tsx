"use client";

// The control surface. Replaces the `.grf` code editor, which authored a script
// language the binary does not accept — there is no parser for it anywhere in
// the Rust tree, so nothing typed there could ever have been executed.
//
// Every control here emits a `Gesture`, so each is a typed value the session
// validates rather than a text edit that has to be parsed back out.

import { useEffect, useRef, useState } from "react";

import type { AskQuery, SceneInfo } from "@/lib/api";
import { MAX_BUDGET, MIN_BUDGET, toCommand, type Gesture } from "@/lib/gestures";

export default function QueryBar({
  query,
  scenes,
  running,
  canUndo,
  canRedo,
  onGesture,
  onRun,
  onUndo,
  onRedo,
}: {
  query: AskQuery;
  scenes: SceneInfo[];
  running: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onGesture: (g: Gesture, opts?: { debounce?: boolean }) => void;
  onRun: () => void;
  onUndo: () => void;
  onRedo: () => void;
}) {
  // The text field is uncontrolled-ish: local state keeps typing responsive
  // while the debounced preview fires from the session.
  const [text, setText] = useState(query.query);
  const inputRef = useRef<HTMLInputElement>(null);

  // Undo/redo and gesture-driven changes must reach the field. Syncing only on
  // an actual difference avoids clobbering the caret while the user types.
  useEffect(() => {
    if (query.query !== text) setText(query.query);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.query]);

  return (
    <div className="flex flex-col gap-3 border-b border-neutral-800 px-4 py-3">
      <div className="flex items-center gap-2">
        <input
          ref={inputRef}
          value={text}
          placeholder="Search this repository…"
          onChange={(e) => {
            setText(e.target.value);
            // Debounced: free text is continuous input, and every keystroke
            // would otherwise be a request.
            onGesture({ kind: "set-query", query: e.target.value }, { debounce: true });
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") onRun();
          }}
          className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-950 px-3 py-1.5 font-mono text-sm text-neutral-100 outline-none placeholder:text-neutral-600 focus:border-sky-600"
        />
        <button
          onClick={onRun}
          disabled={running || !text.trim()}
          title="Commit a search act — this increments the monotone count."
          className="rounded bg-sky-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
        >
          {running ? "Running…" : "Run"}
        </button>
        <button
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo"
          className="rounded border border-neutral-700 px-2 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:border-neutral-800 disabled:text-neutral-600"
        >
          ↶
        </button>
        <button
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo"
          className="rounded border border-neutral-700 px-2 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:border-neutral-800 disabled:text-neutral-600"
        >
          ↷
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <label className="flex items-center gap-2 text-xs text-neutral-400">
          <span className="uppercase tracking-wider text-neutral-500">budget k</span>
          <input
            type="range"
            min={MIN_BUDGET}
            max={MAX_BUDGET}
            value={query.budget}
            onChange={(e) =>
              onGesture(
                { kind: "set-budget", budget: Number(e.target.value) },
                { debounce: true }
              )
            }
            className="w-40 accent-sky-500"
          />
          <span className="w-8 font-mono tabular-nums text-neutral-200">{query.budget}</span>
        </label>

        <label className="flex items-center gap-2 text-xs text-neutral-400">
          <input
            type="checkbox"
            checked={query.flat}
            onChange={() => onGesture({ kind: "toggle-flat" })}
            className="accent-sky-500"
          />
          <span title="Rank globally instead of grouping by scene. Presentation only — the same passages come back either way.">
            flat ranking
          </span>
        </label>

        {scenes.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">scenes</span>
            <button
              onClick={() => onGesture({ kind: "set-scenes", scenes: [] })}
              className={`rounded border px-1.5 py-0.5 text-[11px] ${
                query.scenes.length === 0
                  ? "border-sky-600 bg-sky-950/60 text-sky-300"
                  : "border-neutral-700 text-neutral-400 hover:bg-neutral-800"
              }`}
              title="Search every scene — the CLI's default."
            >
              all
            </button>
            {scenes.map((s) => {
              const on = query.scenes.includes(s.name);
              return (
                <button
                  key={s.name}
                  onClick={() => onGesture({ kind: "toggle-scene", scene: s.name })}
                  title={`${s.documents} document(s), ${s.passages} passage(s) in the index`}
                  className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${
                    on
                      ? "border-sky-600 bg-sky-950/60 text-sky-300"
                      : "border-neutral-700 text-neutral-400 hover:bg-neutral-800"
                  }`}
                >
                  {s.name}
                  <span className="ml-1 text-neutral-600">{s.passages}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* The UI claims to be a view over the binary, so it shows the command it
          stands for. Copy-pasteable, and the fastest way to check that a gesture
          did what the user thought. */}
      <div className="overflow-x-auto font-mono text-[11px] text-neutral-600">
        <span className="select-none text-neutral-700">$ </span>
        {toCommand(query)}
      </div>
    </div>
  );
}
