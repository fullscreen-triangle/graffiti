"use client";

// The shell. Two panes: results on the left, charts on the right.
//
// The three-column IDE this replaces (file explorer → `.grf` editor → charts)
// existed to author a script language nothing parses. There is no file tree to
// browse — the corpus is whatever `spraypaint index` walked — and no document to
// edit. A query is four values, so `QueryBar` holds them and the space goes to
// the passages, which are what the tool actually produces.
//
// The 800ms `setTimeout` that used to wrap "execution" is gone too: real
// requests take as long as they take, and faking a delay on top of a real one
// only makes the tool feel slower than it is.

import { useCallback, useEffect, useMemo, useState } from "react";

import IdentityBadge from "./IdentityBadge";
import OutputPanel from "./OutputPanel";
import PairingScreen from "./PairingScreen";
import QueryBar from "./QueryBar";
import ResultsList from "./ResultsList";
import { ApiError, makeApi, type IndexStatus } from "@/lib/api";
import { useConnection } from "@/hooks/useConnection";
import { useSpraypaintSession } from "@/hooks/useSpraypaintSession";

export default function IDE() {
  const conn = useConnection();

  // Hooks cannot be called conditionally, so the session is always constructed —
  // it simply addresses a server that is not there until pairing succeeds. The
  // early returns below happen after every hook has run.
  const s = useSpraypaintSession(conn.connection);
  const api = useMemo(() => makeApi(conn.connection), [conn.connection]);
  const [split, setSplit] = useState(0.52);
  const [indexJob, setIndexJob] = useState<IndexStatus | null>(null);

  const onBudget = useCallback(
    (budget: number) => s.gesture({ kind: "set-budget", budget }, { debounce: true }),
    [s]
  );

  const onReverify = useCallback(async () => {
    try {
      s.setVerify(await api.verify());
    } catch {
      // `/api/verify` answers 200 even for a breach, so a throw here means the
      // server is unreachable — `refreshContext` reports that condition.
      void s.refreshContext();
    }
  }, [s, api]);

  // Poll only while a build is actually running.
  useEffect(() => {
    if (!indexJob?.running) return;
    const t = setInterval(async () => {
      try {
        const st = await api.indexStatus();
        setIndexJob(st);
        if (!st.running) void s.refreshContext();
      } catch {
        setIndexJob(null);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [indexJob?.running, s, api]);

  const startIndex = useCallback(async () => {
    try {
      // The 202 body carries only `{status}` — not `running` or `detail`. Read
      // the status endpoint for the full shape rather than filling in the two
      // missing fields with guesses.
      await api.startIndex();
      setIndexJob(await api.indexStatus());
    } catch (e) {
      setIndexJob({
        status: "failed",
        running: false,
        detail: e instanceof ApiError ? e.message : String(e),
      });
    }
  }, [api]);

  // Column drag.
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    if (!dragging) return;
    const move = (e: MouseEvent) =>
      setSplit(Math.max(0.25, Math.min(0.75, e.clientX / window.innerWidth)));
    const up = () => setDragging(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragging]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      if (e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        s.undo();
      } else if (e.key === "y" || (e.key === "z" && e.shiftKey)) {
        e.preventDefault();
        s.redo();
      } else if (e.key === "Enter") {
        e.preventDefault();
        void s.run();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [s]);

  const noIndex = s.health !== null && !s.health.has_index;
  const unreachable = s.error?.code === "unreachable";

  // Every hook above has now run, so these returns are safe.
  if (conn.status === "probing") {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-950 text-xs text-neutral-600">
        looking for a spraypaint server…
      </div>
    );
  }
  if (conn.status === "unpaired") {
    return (
      <PairingScreen
        onConnect={conn.connect}
        connecting={conn.connecting}
        error={conn.error}
      />
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-neutral-950 text-neutral-200">
      <div className="flex h-[30px] shrink-0 items-center gap-3 border-b border-neutral-800 bg-neutral-900 px-3 text-xs">
        <span className="font-semibold tracking-wide text-sky-400">spraypaint</span>
        <span className="text-neutral-600">
          {s.health?.root ?? "—"}
        </span>
        {s.health?.version && (
          <span className="font-mono text-[10px] text-neutral-600">v{s.health.version}</span>
        )}
        <span className="ml-auto flex items-center gap-2 text-[11px] text-neutral-500">
          {conn.status === "paired" && (
            <>
              <span
                title="This page is driving a spraypaint binary on your machine."
                className="rounded border border-emerald-800 px-1.5 py-px font-mono text-[10px] text-emerald-400"
              >
                paired · {conn.connection.baseUrl.replace(/^https?:\/\//, "")}
              </span>
              <button
                onClick={conn.disconnect}
                className="text-neutral-600 underline-offset-2 hover:text-neutral-400 hover:underline"
              >
                disconnect
              </button>
            </>
          )}
          {s.running && <span className="text-sky-400">running…</span>}
          {s.stale && !s.running && (
            <span className="text-amber-500">query changed — results are from a previous run</span>
          )}
        </span>
      </div>

      {unreachable && (
        <div className="border-b border-red-900/60 bg-red-950/30 px-4 py-2 text-xs text-red-300">
          {s.error?.message}
        </div>
      )}

      {noIndex && (
        <div className="flex items-center gap-3 border-b border-amber-900/50 bg-amber-950/20 px-4 py-2 text-xs text-amber-200">
          <span>No index in this repository.</span>
          {s.health?.allow_index ? (
            <button
              onClick={startIndex}
              disabled={indexJob?.running}
              className="rounded border border-amber-700 px-2 py-0.5 text-[11px] hover:bg-amber-900/40 disabled:text-amber-700"
            >
              {indexJob?.running ? "Indexing…" : "Build index"}
            </button>
          ) : (
            // Indexing takes a blocking exclusive lock, so the server refuses it
            // unless explicitly permitted. Naming the flag is the whole message.
            <span className="font-mono text-[11px] text-amber-400/80">
              run <span className="text-amber-200">spraypaint index</span>, or restart the
              server with <span className="text-amber-200">--allow-index</span>
            </span>
          )}
          {indexJob?.detail && (
            <span className="font-mono text-[11px] text-amber-400/70">{indexJob.detail}</span>
          )}
        </div>
      )}

      {s.error && !unreachable && !noIndex && (
        <div className="border-b border-red-900/60 bg-red-950/30 px-4 py-2 text-xs text-red-300">
          <span className="font-mono text-red-400">{s.error.code}</span> — {s.error.message}
        </div>
      )}

      <IdentityBadge identity={s.identity} result={s.result} count={s.count} />

      <QueryBar
        query={s.query}
        scenes={s.scenes}
        running={s.running}
        canUndo={s.canUndo}
        canRedo={s.canRedo}
        onGesture={s.gesture}
        onRun={() => void s.run()}
        onUndo={s.undo}
        onRedo={s.redo}
      />

      <div className="flex min-h-0 flex-1">
        <div style={{ width: `${split * 100}%` }} className="min-w-0">
          <ResultsList result={s.result} query={s.query} stale={s.stale} />
        </div>
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          className="w-[3px] shrink-0 cursor-col-resize bg-neutral-800 transition-colors hover:bg-sky-600"
        />
        <div style={{ width: `${(1 - split) * 100}%` }} className="min-w-0">
          <OutputPanel
            result={s.result}
            query={s.query}
            identity={s.identity}
            scenes={s.scenes}
            verify={s.verify}
            count={s.count}
            stale={s.stale}
            running={s.running}
            onBudget={onBudget}
            onReverify={onReverify}
          />
        </div>
      </div>

      <div className="flex h-[22px] shrink-0 items-center gap-4 bg-sky-800 px-3 text-[11px] text-white">
        <span>{s.scenes.length} scene{s.scenes.length === 1 ? "" : "s"}</span>
        <span>
          {s.scenes.reduce((n, x) => n + x.passages, 0)} passages indexed
        </span>
        {s.count !== null && <span>M = {s.count}</span>}
        <span className="ml-auto">
          {/* The distinction the whole session hook exists to preserve. */}
          {s.result === null
            ? "ready"
            : s.result.dry_run
              ? "preview — not committed"
              : "committed"}
        </span>
      </div>
    </div>
  );
}
