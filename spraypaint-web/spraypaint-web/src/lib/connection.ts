// ── Where the API lives, and what may talk to it ──
//
// There are two topologies, and this module is the only place that knows which
// one is in play:
//
//  1. **Served by the binary.** `spraypaint serve` embeds this UI, so the page
//     origin *is* the API origin. Requests are relative, same-origin, no token.
//     This is the path with no browser caveats at all.
//
//  2. **Paired.** The page is hosted (Vercel), the binary runs on the visitor's
//     machine, and `spraypaint serve --pair <this-origin>` authorises exactly
//     that origin with a token printed on their terminal. Requests are then
//     absolute, cross-origin, and bearer-authenticated.
//
// Mode 1 is detected, not configured: if `/api/health` answers on our own
// origin, we are being served by the binary and nothing else is needed.

const STORAGE_KEY = "spraypaint.connection.v1";

/** Where the binary listens by default (`serve`'s `--port` default is 7373). */
export const DEFAULT_SERVER_URL = "http://127.0.0.1:7373";

export interface Connection {
  /** Absolute base URL, or "" for same-origin. */
  baseUrl: string;
  /** Bearer token. Empty in same-origin mode, where none is required. */
  token: string;
}

export const SAME_ORIGIN: Connection = { baseUrl: "", token: "" };

/** Is this connection the served-by-the-binary case? */
export function isSameOrigin(c: Connection): boolean {
  return c.baseUrl === "";
}

/**
 * Normalise a server URL typed by a human.
 *
 * Accepts `127.0.0.1:7373` and `localhost:7373` without a scheme, since that is
 * what people type, and the banner prints a full URL anyway. A trailing slash is
 * dropped so `baseUrl + "/api/health"` never produces a double slash.
 */
export function normaliseServerUrl(raw: string): { url: string } | { error: string } {
  const s = raw.trim().replace(/\/+$/, "");
  if (!s) return { error: "Enter the server URL printed by `spraypaint serve`." };

  const withScheme = /^https?:\/\//i.test(s) ? s : `http://${s}`;
  let u: URL;
  try {
    u = new URL(withScheme);
  } catch {
    return { error: `Not a URL: ${raw}` };
  }
  if (u.pathname !== "/" || u.search || u.hash) {
    return { error: "Use just the host and port, e.g. http://127.0.0.1:7373" };
  }
  return { url: `${u.protocol}//${u.host}` };
}

/**
 * Does this URL point at the local machine?
 *
 * Used only to decide whether to send `targetAddressSpace: "local"` and to
 * decide which browser guidance to show — never as a security control. The
 * server does its own checking, and a browser lying about this would gain
 * nothing since the token is still required.
 */
export function isLoopbackUrl(url: string): boolean {
  try {
    const h = new URL(url).hostname.toLowerCase();
    return (
      h === "localhost" ||
      h === "[::1]" ||
      h === "::1" ||
      h.endsWith(".localhost") ||
      /^127\./.test(h)
    );
  } catch {
    return false;
  }
}

/** Tokens are 40 lowercase hex characters — 20 bytes from `Token::mint`. */
export function looksLikeToken(t: string): boolean {
  return /^[0-9a-f]{40}$/.test(t.trim());
}

// ── Browser capability ────────────────────────────────────────────────────
//
// Reaching `http://127.0.0.1` from an `https://` page is not uniformly allowed,
// and the two live restrictions are commonly confused, so both are named here:
//
//  * **Mixed content is NOT the obstacle.** Loopback has been exempt from mixed
//    content blocking since Chrome 53; `http://127.0.0.1` from an HTTPS page is
//    a "potentially trustworthy" origin by spec.
//
//  * **Local Network Access is.** From Chrome 142, a public HTTPS origin
//    reaching loopback triggers a permission prompt. Granted, it works; denied
//    or dismissed, `fetch` rejects with a bare `TypeError: Failed to fetch`,
//    indistinguishable from the server being down. Hence the explicit hint.
//
//  * **Safari cannot do this at all.** WebKit departs from the Mixed Content
//    spec here and forbids loopback subresources from HTTPS documents outright.
//    There is no server-side fix and no flag a user can set — the honest answer
//    is to run `spraypaint serve --open` and use the UI the binary serves.

export type BrowserSupport = "supported" | "prompt-required" | "unsupported";

export interface BrowserVerdict {
  support: BrowserSupport;
  /** One sentence, shown to the user. Empty when nothing needs saying. */
  note: string;
}

/**
 * What this browser will do with a paired connection to loopback.
 *
 * User-agent sniffing, which is normally the wrong tool — but the property being
 * detected is a *refusal*, and a refusal is not feature-detectable in advance:
 * the only way to observe it is to make the request and get an error that looks
 * exactly like a dead server. Sniffing here buys an accurate message, never a
 * behaviour change, so a wrong guess costs only wording.
 */
export function browserVerdict(pageIsHttps: boolean, ua: string): BrowserVerdict {
  if (!pageIsHttps) {
    // An http:// page (localhost dev, or the binary's own UI) has none of these
    // restrictions.
    return { support: "supported", note: "" };
  }

  const isChromium = /Chrome|Chromium|Edg\//.test(ua) && !/OPR\//.test(ua);
  // Safari is the UA that claims WebKit without claiming Chrome.
  const isSafari = /Safari/.test(ua) && !/Chrome|Chromium|Edg\/|OPR\//.test(ua);

  if (isSafari) {
    return {
      support: "unsupported",
      note:
        "Safari blocks HTTPS pages from reaching http://127.0.0.1 and does not " +
        "offer a permission prompt, so pairing cannot work here. Run " +
        "`spraypaint serve --open` and use the interface the binary serves, or " +
        "pair from Chrome, Edge, or Firefox.",
    };
  }
  if (isChromium) {
    return {
      support: "prompt-required",
      note:
        "Chrome will ask permission to reach your local network the first time " +
        "you connect. Choose Allow — if you dismiss it, the connection fails " +
        "with the same message as a server that is not running.",
    };
  }
  return { support: "supported", note: "" };
}

// ── Persistence ───────────────────────────────────────────────────────────

/**
 * Load a stored connection.
 *
 * The token is stored in `sessionStorage`, not `localStorage`: it authorises
 * reading the content of any indexed file in the user's repo, and it dies with
 * the server process anyway, so persisting it beyond the tab would only widen
 * the window in which a stale secret sits on disk for no benefit.
 */
export function loadConnection(): Connection | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Connection>;
    if (typeof parsed.baseUrl !== "string" || typeof parsed.token !== "string") return null;
    return { baseUrl: parsed.baseUrl, token: parsed.token };
  } catch {
    return null;
  }
}

export function saveConnection(c: Connection): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(c));
  } catch {
    // Private mode, or storage disabled. The connection still works for this
    // page load; it simply will not survive a reload.
  }
}

export function clearConnection(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* as above */
  }
}
