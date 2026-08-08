// ── Gestures: every user interaction as a typed value ──
//
// This replaces `applyDiff`, which was `script.replace(oldText, newText)` — an
// un-anchored, first-occurrence, plain-string patch against a `.grf` script.
// That could silently rewrite a comment, or match nothing and no-op invisibly,
// and there was no way to tell the two apart.
//
// Here a gesture is a small tagged object and `applyGesture` is pure and total:
// it maps a valid `AskQuery` to a valid `AskQuery` for every input. There is no
// parse step, no ambiguous match, and no silent failure — an invalid query is
// unrepresentable rather than merely unlikely.
//
// Undo therefore stores whole `AskQuery` values, so undo is exact rather than a
// reverse-replace that may or may not find its own text again.

import type { AskQuery } from "./api";

/** Hard ceiling on the budget slider.
 *
 * Not a property of the binary — `ask` accepts any usize and water-filling
 * simply allocates everything available. It is a UI guard: each allocated
 * passage is re-read from disk and rendered, so an accidental drag to 10,000
 * would produce a page nobody wants and a request nobody meant to make. */
export const MAX_BUDGET = 200;
export const MIN_BUDGET = 1;

export type Gesture =
  | { kind: "set-query"; query: string }
  | { kind: "set-budget"; budget: number }
  | { kind: "toggle-scene"; scene: string }
  /** Replace the whole scene selection at once (e.g. "all" / "none" buttons). */
  | { kind: "set-scenes"; scenes: string[] }
  | { kind: "toggle-flat" }
  /** Restore a previous query wholesale — used by undo/redo and history. */
  | { kind: "reset"; query: AskQuery };

export function clampBudget(n: number): number {
  if (!Number.isFinite(n)) return MIN_BUDGET;
  return Math.min(MAX_BUDGET, Math.max(MIN_BUDGET, Math.round(n)));
}

/**
 * Apply a gesture. Pure, total, and never mutates its input.
 *
 * `known` is the live scene list from `/api/scenes`. Scene toggles are validated
 * against it, so a stale name — from a restored session, or a scene that
 * vanished when the repo was re-indexed — cannot enter the query and be sent to
 * a binary that would silently return nothing for it. When `known` is not yet
 * loaded the toggle is accepted as-is; rejecting everything before the first
 * fetch resolves would make the UI feel broken on load.
 */
export function applyGesture(q: AskQuery, g: Gesture, known?: string[]): AskQuery {
  switch (g.kind) {
    case "set-query":
      return { ...q, query: g.query };

    case "set-budget":
      return { ...q, budget: clampBudget(g.budget) };

    case "toggle-scene": {
      if (known && known.length > 0 && !known.includes(g.scene)) return q;
      const on = q.scenes.includes(g.scene);
      const scenes = on
        ? q.scenes.filter((s) => s !== g.scene)
        : [...q.scenes, g.scene].sort();
      return { ...q, scenes };
    }

    case "set-scenes": {
      const filtered =
        known && known.length > 0 ? g.scenes.filter((s) => known.includes(s)) : g.scenes;
      // Deduplicated and sorted so that two selections with the same members are
      // `equalQuery`-identical regardless of the order they were clicked in.
      return { ...q, scenes: [...new Set(filtered)].sort() };
    }

    case "toggle-flat":
      return { ...q, flat: !q.flat };

    case "reset":
      return normalizeQuery(g.query);
  }
}

/** Coerce anything query-shaped into a valid `AskQuery`. */
export function normalizeQuery(q: AskQuery): AskQuery {
  return {
    query: q.query ?? "",
    budget: clampBudget(q.budget),
    scenes: [...new Set(q.scenes ?? [])].sort(),
    flat: Boolean(q.flat),
  };
}

/**
 * Do two queries denote the same request?
 *
 * `flat` is excluded on purpose: it is a presentation choice made in the
 * browser and is never sent to the binary (`askBody` in `api.ts` omits it), so
 * toggling it does not make a displayed result stale. Including it here would
 * mark perfectly current results as needing a re-run.
 */
export function sameRequest(a: AskQuery, b: AskQuery): boolean {
  return (
    a.query.trim() === b.query.trim() &&
    a.budget === b.budget &&
    a.scenes.length === b.scenes.length &&
    a.scenes.every((s, i) => s === b.scenes[i])
  );
}

/** Is this query worth sending at all? Mirrors the server's own 400 on empty `q`. */
export function isRunnable(q: AskQuery): boolean {
  return q.query.trim().length > 0;
}

/** Short human label for a gesture — used as the undo entry's description. */
export function describeGesture(g: Gesture): string {
  switch (g.kind) {
    case "set-query":
      return g.query.trim() ? `query "${g.query.trim()}"` : "cleared query";
    case "set-budget":
      return `budget k=${clampBudget(g.budget)}`;
    case "toggle-scene":
      return `scene ${g.scene}`;
    case "set-scenes":
      return g.scenes.length ? `scenes: ${g.scenes.join(", ")}` : "all scenes";
    case "toggle-flat":
      return "flat ranking";
    case "reset":
      return "restored";
  }
}

/**
 * The equivalent CLI invocation, for display.
 *
 * The UI's claim is that it is a view over the binary, so it should be able to
 * show you the command it stands for. Arguments are quoted when they contain
 * anything a shell would treat specially — this is a display aid, and a line
 * the user may well copy and paste.
 */
export function toCommand(q: AskQuery): string {
  const quote = (s: string) => (/^[A-Za-z0-9_.,\/-]+$/.test(s) ? s : `"${s.replace(/"/g, '\\"')}"`);
  const parts = ["spraypaint", "ask", quote(q.query || "")];
  if (q.budget !== 12) parts.push("-k", String(q.budget));
  if (q.scenes.length) parts.push("--scenes", quote(q.scenes.join(",")));
  if (q.flat) parts.push("--flat");
  return parts.join(" ");
}
