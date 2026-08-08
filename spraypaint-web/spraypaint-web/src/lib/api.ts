// ── Typed client for the `spraypaint serve` JSON API ──
//
// Every type here mirrors bytes the Rust binary actually emits (`output.rs`,
// `serve/api.rs`). Nothing in this file invents a value: if the binary does not
// report a number, no number appears. That is the project's standing rule —
// the web tool is a view over `spraypaint … --json`, and it never simulates a
// result.
//
// Requests are same-origin and relative. The UI is served by the binary itself,
// so there is no host to configure, no CORS, and no cross-origin request that
// the server's Host check would (correctly) reject with 421.

/** A query, in the form the CLI takes it. Maps 1:1 to argv. */
export interface AskQuery {
  query: string;
  /** Water-filling budget A — the `-k` flag. Always >= 1. */
  budget: number;
  /** Empty means "all scenes", matching the CLI's `--scenes` default. */
  scenes: string[];
  /** Rank globally rather than grouping by scene (presentation only). */
  flat: boolean;
}

export const DEFAULT_QUERY: AskQuery = {
  query: "",
  budget: 12,
  scenes: [],
  flat: false,
};

/**
 * Per-scene allocation and score distribution.
 *
 * Three counts that are easy to conflate and must not be:
 *
 *  - `allocated` — passages water-filling gave this scene, capped by budget.
 *  - `available` — passages scoring **above zero for this query**. Not the
 *    scene's size; a different query gives a different number.
 *  - a scene's total passage count is `SceneInfo.passages`, from `/api/scenes`.
 *
 * `best_score`/`median_score` are computed in `ask.rs` over every scoring
 * passage *before* truncation, so they describe the scene rather than the
 * returned slice. `null` when the scene scored nothing.
 */
export interface AskAllocation {
  scene: string;
  allocated: number;
  available: number;
  best_score: number | null;
  median_score: number | null;
}

/** One returned passage. `snippet` is re-read from disk at query time (Inv 3). */
export interface AskResult {
  scene: string;
  path: string;
  start_line: number;
  end_line: number;
  score: number;
  snippet: string;
}

/**
 * A committed answer, or a dry-run preview.
 *
 * The two differ in exactly two fields, and both differences are load-bearing:
 * `dry_run` is `true` on a preview, and `committed_count` is **absent** rather
 * than zero, because a preview does not touch the monotone counter (Inv 3).
 * Treating a missing count as 0 would display a count that is almost certainly
 * wrong, so it is typed as optional and rendered as "—".
 */
export interface AskResponse {
  dry_run?: boolean;
  query_terms: string[];
  budget: number;
  /** Clearing price p* — an **output** of water-filling, never an input. */
  price: number;
  committed_count?: number;
  identity_fingerprint: string;
  allocation: AskAllocation[];
  results: AskResult[];
}

/** The conserved identity block (Inv 1). */
export interface Identity {
  fingerprint: string;
  /** Min-cut of the self-graph. */
  char_invariant: number;
  /** Construction parameter: the weight every edge carries. Not a measurement. */
  floor: number;
  n_vertices: number;
  n_edges: number;
}

/** A scene as the index knows it — `passages` here is the true total. */
export interface SceneInfo {
  name: string;
  documents: number;
  passages: number;
}

/** Three-state check outcome. `n/a` is distinct from pass: it means untested. */
export type CheckStatus = "pass" | "fail" | "n/a";

export interface VerifyCheck {
  name: string;
  status: CheckStatus;
  detail: string;
}

export interface VerifyInvariant {
  title?: string;
  status: CheckStatus;
  pass: boolean;
  checks: VerifyCheck[];
}

export interface VerifyResponse {
  /** Strict: true only if every check passed and none was n/a. */
  pass: boolean;
  overall: CheckStatus;
  /** Regimes where a PASS would not be evidence. Empty is the good case. */
  degeneracies: string[];
  inv1_identity: VerifyInvariant;
  inv2_count: VerifyInvariant;
  inv3_search_not_fetch: VerifyInvariant;
  inv4_phases: VerifyInvariant;
}

export interface Health {
  ok: boolean;
  version: string;
  root: string;
  has_index: boolean;
  allow_index: boolean;
  indexing: boolean;
}

export interface IndexStatus {
  status: "idle" | "running" | "done" | "failed";
  running: boolean;
  detail: string | null;
}

/** An error the server reported, carrying its stable code. */
export class ApiError extends Error {
  constructor(
    /** Stable code: no_index, identity_mismatch, index_disabled, … */
    readonly code: string,
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** Is this the "you have not indexed yet" case the UI has a screen for? */
  get isNoIndex(): boolean {
    return this.code === "no_index";
  }
}

/**
 * One request. Errors become `ApiError` with the server's own code and message.
 *
 * A non-JSON body on a failure is not swallowed: if the server (or something in
 * front of it) returns HTML, the status line becomes the message rather than a
 * misleading JSON parse error.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (e) {
    // fetch rejects only on a transport failure — the server is not running,
    // or the page was opened from a file:// URL rather than served.
    throw new ApiError(
      "unreachable",
      `Cannot reach the spraypaint server. Is \`spraypaint serve\` still running? (${
        e instanceof Error ? e.message : String(e)
      })`,
      0
    );
  }

  const text = await res.text();
  if (!res.ok) {
    try {
      const body = JSON.parse(text) as { error?: { code: string; message: string } };
      if (body.error) throw new ApiError(body.error.code, body.error.message, res.status);
    } catch (e) {
      if (e instanceof ApiError) throw e;
    }
    throw new ApiError("http_error", `HTTP ${res.status}: ${text.slice(0, 300)}`, res.status);
  }

  return JSON.parse(text) as T;
}

function askBody(q: AskQuery): string {
  // `flat` is deliberately not sent: it is a presentation choice made in the
  // browser, and the server has no flag for it. Sending it would imply the
  // result set changes, which it does not.
  return JSON.stringify({
    query: q.query,
    budget: Math.max(1, Math.floor(q.budget)),
    scenes: q.scenes,
  });
}

export const api = {
  health: () => request<Health>("/api/health"),

  /**
   * **Commits a search act** — increments the monotone count (Inv 2).
   * Only an explicit Run may call this. Everything interactive uses `dryRun`.
   */
  ask: (q: AskQuery) =>
    request<AskResponse>("/api/ask", { method: "POST", body: askBody(q) }),

  /**
   * Diagnostics with no answer committed. The count has no decrement path, so
   * previewing through `ask` would inflate it permanently on every gesture.
   */
  dryRun: (q: AskQuery) =>
    request<AskResponse>("/api/dry-run", { method: "POST", body: askBody(q) }),

  identity: () => request<Identity>("/api/identity"),
  count: () => request<{ committed_count: number }>("/api/count"),
  scenes: () => request<SceneInfo[]>("/api/scenes"),

  /** Always HTTP 200, including on a breach — `pass:false` is a real answer. */
  verify: () => request<VerifyResponse>("/api/verify"),

  startIndex: () => request<{ status: string }>("/api/index", { method: "POST" }),
  indexStatus: () => request<IndexStatus>("/api/index/status"),
};
