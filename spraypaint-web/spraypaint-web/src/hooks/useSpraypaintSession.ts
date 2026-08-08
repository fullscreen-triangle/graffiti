"use client";

// ── The session: one query, one result, one undo history ──
//
// The single rule this hook exists to enforce:
//
//   **Interactive changes call `/api/dry-run`. Only an explicit Run calls
//   `/api/ask`.**
//
// The committed count is monotone and has no decrement path (`count.rs`). If a
// budget drag previewed through `/api/ask`, every intermediate value on the way
// from k=12 to k=40 would be permanently recorded as a search act, and the
// number the invariant panel displays would become meaningless within a minute
// of use. `dryRun` returns the same shape with `committed_count` absent, so the
// UI can show a full preview without asserting an act that did not happen.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  api,
  ApiError,
  DEFAULT_QUERY,
  type AskQuery,
  type AskResponse,
  type Health,
  type Identity,
  type SceneInfo,
  type VerifyResponse,
} from "@/lib/api";
import { applyGesture, describeGesture, isRunnable, sameRequest, type Gesture } from "@/lib/gestures";
import { UndoStack } from "@/lib/undo";

/** How long free text sits still before a preview fires. */
const PREVIEW_DEBOUNCE_MS = 400;

export interface SessionState {
  query: AskQuery;
  /** Last response, preview or committed. `dry_run` distinguishes them. */
  result: AskResponse | null;
  /** True when `result` describes a query other than the current one. */
  stale: boolean;
  running: boolean;
  error: ApiError | null;
  health: Health | null;
  identity: Identity | null;
  scenes: SceneInfo[];
  verify: VerifyResponse | null;
  /** Committed count, from the last commit or an explicit `/api/count` read. */
  count: number | null;
  canUndo: boolean;
  canRedo: boolean;
}

export function useSpraypaintSession() {
  const [query, setQuery] = useState<AskQuery>(DEFAULT_QUERY);
  const [result, setResult] = useState<AskResponse | null>(null);
  /** The query `result` was produced from. The basis for staleness. */
  const [executedQuery, setExecutedQuery] = useState<AskQuery | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [scenes, setScenes] = useState<SceneInfo[]>([]);
  const [verify, setVerify] = useState<VerifyResponse | null>(null);
  const [count, setCount] = useState<number | null>(null);

  // `canUndo`/`canRedo` are mirrored into state deliberately. The stack itself
  // lives in a ref because mutating it must not re-render, but a ref's value
  // never reaches the render pass — which is why the old toolbar buttons never
  // disabled. These two booleans are the bridge.
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const undoRef = useRef<UndoStack<AskQuery>>(new UndoStack<AskQuery>());

  // Monotone request id. An older in-flight preview that resolves after a newer
  // one must not overwrite it — without this, typing fast leaves the results
  // showing whichever request the network happened to finish last.
  const seqRef = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const syncUndoFlags = useCallback(() => {
    setCanUndo(undoRef.current.canUndo());
    setCanRedo(undoRef.current.canRedo());
  }, []);

  // Seed the undo stack so the initial query is a restorable state rather than
  // a value with nothing behind it.
  useEffect(() => {
    undoRef.current.push({
      value: DEFAULT_QUERY,
      timestamp: Date.now(),
      source: "init",
      description: "session start",
    });
    syncUndoFlags();
  }, [syncUndoFlags]);

  /** Load everything that does not depend on a query. */
  const refreshContext = useCallback(async () => {
    try {
      const h = await api.health();
      setHealth(h);
      if (!h.has_index) {
        // Nothing else will succeed, and each would throw `no_index`. Report the
        // one condition that actually explains it.
        setIdentity(null);
        setScenes([]);
        setVerify(null);
        return;
      }
    } catch (e) {
      if (e instanceof ApiError) setError(e);
      return;
    }

    // Independent reads, so failures are per-endpoint rather than all-or-nothing:
    // a broken `verify` should not blank out the scene list.
    const [id, sc, vf, ct] = await Promise.allSettled([
      api.identity(),
      api.scenes(),
      api.verify(),
      api.count(),
    ]);
    if (id.status === "fulfilled") setIdentity(id.value);
    if (sc.status === "fulfilled") setScenes(sc.value);
    if (vf.status === "fulfilled") setVerify(vf.value);
    if (ct.status === "fulfilled") setCount(ct.value.committed_count);
  }, []);

  useEffect(() => {
    void refreshContext();
  }, [refreshContext]);

  /**
   * Execute a query. `commit` decides which endpoint — and that is the whole
   * difference between "showing you what would happen" and "recording that it
   * did".
   */
  const execute = useCallback(async (q: AskQuery, commit: boolean) => {
    if (!isRunnable(q)) {
      setResult(null);
      setExecutedQuery(null);
      setError(null);
      return;
    }
    const seq = ++seqRef.current;
    setRunning(true);
    setError(null);
    try {
      const res = commit ? await api.ask(q) : await api.dryRun(q);
      if (seq !== seqRef.current) return; // superseded
      setResult(res);
      setExecutedQuery(q);
      if (typeof res.committed_count === "number") setCount(res.committed_count);
    } catch (e) {
      if (seq !== seqRef.current) return;
      setError(e instanceof ApiError ? e : new ApiError("internal", String(e), 0));
      setResult(null);
      setExecutedQuery(null);
    } finally {
      if (seq === seqRef.current) setRunning(false);
    }
  }, []);

  /** Explicit Run. The only path that increments the count. */
  const run = useCallback(async () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    await execute(query, true);
  }, [execute, query]);

  /** Preview now, without debounce. For discrete gestures — a click is already
   *  a deliberate act and does not need waiting out. */
  const preview = useCallback(
    async (q: AskQuery = query) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      await execute(q, false);
    },
    [execute, query]
  );

  /** Preview after the user stops changing things. For free text and drags. */
  const previewDebounced = useCallback(
    (q: AskQuery) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => void execute(q, false), PREVIEW_DEBOUNCE_MS);
    },
    [execute]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const known = scenes.map((s) => s.name);

  /**
   * Apply a gesture: update the query, record undo, and preview.
   *
   * `debounce` should be true for continuous input (typing, dragging) and false
   * for discrete input (a scene chip click). It only affects *when* the preview
   * fires, never whether the count moves — a preview is a dry run either way.
   */
  const gesture = useCallback(
    (g: Gesture, opts?: { debounce?: boolean; preview?: boolean }) => {
      const next = applyGesture(query, g, known);
      if (sameRequest(next, query) && next.flat === query.flat) return; // no-op
      setQuery(next);
      undoRef.current.push({
        value: next,
        timestamp: Date.now(),
        source: g.kind === "set-query" ? "editor" : "gesture",
        description: describeGesture(g),
      });
      syncUndoFlags();

      // A `flat` toggle is pure presentation — `askBody` does not send it — so
      // re-running would spend a request to receive identical bytes.
      if (opts?.preview === false || g.kind === "toggle-flat") return;
      if (opts?.debounce) previewDebounced(next);
      else void preview(next);
    },
    [query, known, preview, previewDebounced, syncUndoFlags]
  );

  /** Gesture then commit, in one step. For "click a scene and Run". */
  const applyAndRun = useCallback(
    async (g: Gesture) => {
      const next = applyGesture(query, g, known);
      setQuery(next);
      undoRef.current.push({
        value: next,
        timestamp: Date.now(),
        source: "gesture",
        description: describeGesture(g),
      });
      syncUndoFlags();
      await execute(next, true);
    },
    [query, known, execute, syncUndoFlags]
  );

  // Undo restores a query and *previews* it. It does not re-commit: undoing a
  // search should not record a second search act.
  const undo = useCallback(() => {
    const entry = undoRef.current.undo();
    syncUndoFlags();
    if (!entry) return;
    setQuery(entry.value);
    void preview(entry.value);
  }, [preview, syncUndoFlags]);

  const redo = useCallback(() => {
    const entry = undoRef.current.redo();
    syncUndoFlags();
    if (!entry) return;
    setQuery(entry.value);
    void preview(entry.value);
  }, [preview, syncUndoFlags]);

  const state: SessionState = {
    query,
    result,
    // Staleness compares against the query that actually produced `result`,
    // recorded at execution time. Deriving it from the payload instead would be
    // guesswork: the response carries `budget` and the *stemmed* `query_terms`,
    // but neither the raw query string nor the scene filter, so two different
    // requests can produce indistinguishable payloads. `flat` is excluded by
    // `sameRequest` — it changes only on-screen ordering.
    stale: result !== null && executedQuery !== null && !sameRequest(query, executedQuery),
    running,
    error,
    health,
    identity,
    scenes,
    verify,
    count,
    canUndo,
    canRedo,
  };

  return {
    ...state,
    run,
    preview,
    gesture,
    applyAndRun,
    undo,
    redo,
    refreshContext,
    setVerify,
    startIndex: api.startIndex,
    indexStatus: api.indexStatus,
  };
}

