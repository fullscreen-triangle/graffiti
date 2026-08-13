"use client";

// ── The pairing screen ──
//
// Shown only when this page is hosted and no local binary is connected. Its job
// is to get the user from "a web page that can do nothing" to "a web page
// driving the binary on my machine", and to be honest about what that grants.
//
// The command is shown with this page's *actual* origin substituted in, because
// the server matches the origin exactly — a copied command carrying the wrong
// host produces a 403 that looks, from here, exactly like a bad token.

import { useEffect, useMemo, useState } from "react";

import {
  type Connection,
  DEFAULT_SERVER_URL,
  browserVerdict,
  looksLikeToken,
  normaliseServerUrl,
} from "@/lib/connection";

interface Props {
  onConnect: (c: Connection) => Promise<boolean>;
  connecting: boolean;
  /** Transport-level failure from the last attempt, if any. */
  error: string | null;
}

export default function PairingScreen({ onConnect, connecting, error }: Props) {
  const [url, setUrl] = useState(DEFAULT_SERVER_URL);
  const [token, setToken] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const [origin, setOrigin] = useState("");
  const [copied, setCopied] = useState(false);

  // Read from `window` in an effect, not during render: this component is
  // statically exported, so render also happens at build time where there is no
  // window and the origin is not knowable.
  useEffect(() => {
    setOrigin(window.location.origin);
  }, []);

  const verdict = useMemo(
    () =>
      typeof window === "undefined"
        ? { support: "supported" as const, note: "" }
        : browserVerdict(window.location.protocol === "https:", navigator.userAgent),
    []
  );

  const command = origin ? `spraypaint serve --pair ${origin}` : "spraypaint serve --pair <this page's URL>";

  const submit = async () => {
    setLocalError(null);
    const parsed = normaliseServerUrl(url);
    if ("error" in parsed) {
      setLocalError(parsed.error);
      return;
    }
    const t = token.trim();
    if (!t) {
      setLocalError("Paste the token the server printed.");
      return;
    }
    // A shape check, not a security check — the server decides. It exists to
    // catch the common paste error (a partial selection, or the whole banner
    // line) before it becomes an indistinguishable 401.
    if (!looksLikeToken(t)) {
      setLocalError(
        "That does not look like a token. Expect 40 hexadecimal characters, on its own line under “Token”."
      );
      return;
    }
    await onConnect({ baseUrl: parsed.url, token: t });
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-950 px-6 py-12 text-neutral-200">
      <div className="w-full max-w-2xl">
        <h1 className="text-lg font-semibold tracking-wide text-sky-400">spraypaint</h1>
        <p className="mt-2 text-sm text-neutral-400">
          This page is the interface. The search itself runs on your machine — your code is
          never uploaded, and this page can only reach a server you explicitly pair with it.
        </p>

        {verdict.support === "unsupported" ? (
          <div className="mt-6 rounded border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-300">
            {verdict.note}
          </div>
        ) : (
          <>
            <ol className="mt-8 space-y-6 text-sm">
              <li>
                <div className="text-neutral-300">
                  <span className="mr-2 font-mono text-neutral-500">1.</span>
                  Install the binary and index a repository.
                </div>
                <pre className="mt-2 overflow-x-auto rounded border border-neutral-800 bg-neutral-900 p-3 font-mono text-xs text-neutral-300">
                  {"cargo install --path spraypaint\ncd /your/repo && spraypaint index"}
                </pre>
              </li>

              <li>
                <div className="text-neutral-300">
                  <span className="mr-2 font-mono text-neutral-500">2.</span>
                  Start the server, authorising this page by name.
                </div>
                <div className="mt-2 flex items-start gap-2">
                  <pre className="min-w-0 flex-1 overflow-x-auto rounded border border-neutral-800 bg-neutral-900 p-3 font-mono text-xs text-sky-300">
                    {command}
                  </pre>
                  <button
                    onClick={() => {
                      void navigator.clipboard?.writeText(command).then(() => {
                        setCopied(true);
                        setTimeout(() => setCopied(false), 1500);
                      });
                    }}
                    disabled={!origin}
                    className="shrink-0 rounded border border-neutral-700 px-2 py-1 text-[11px] text-neutral-300 hover:bg-neutral-800 disabled:opacity-40"
                  >
                    {copied ? "copied" : "copy"}
                  </button>
                </div>
                <p className="mt-2 text-xs text-neutral-500">
                  It prints a token. That token authorises this page to read the content of any
                  indexed file in that repository and to increment the committed count, so paste
                  it here and nowhere else. It exists only in that process — stopping the server
                  revokes it.
                </p>
              </li>

              <li>
                <div className="text-neutral-300">
                  <span className="mr-2 font-mono text-neutral-500">3.</span>
                  Connect.
                </div>
                <div className="mt-3 space-y-3">
                  <label className="block">
                    <span className="text-xs text-neutral-500">Server URL</span>
                    <input
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      spellCheck={false}
                      className="mt-1 w-full rounded border border-neutral-800 bg-neutral-900 px-3 py-2 font-mono text-xs text-neutral-200 outline-none focus:border-sky-600"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-neutral-500">Token</span>
                    <input
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") void submit();
                      }}
                      spellCheck={false}
                      autoComplete="off"
                      placeholder="40 hex characters"
                      className="mt-1 w-full rounded border border-neutral-800 bg-neutral-900 px-3 py-2 font-mono text-xs text-neutral-200 outline-none focus:border-sky-600"
                    />
                  </label>
                  <button
                    onClick={() => void submit()}
                    disabled={connecting}
                    className="rounded bg-sky-700 px-4 py-2 text-xs font-medium text-white hover:bg-sky-600 disabled:opacity-50"
                  >
                    {connecting ? "Connecting…" : "Connect"}
                  </button>
                </div>
              </li>
            </ol>

            {verdict.note && verdict.support === "prompt-required" && (
              <div className="mt-6 rounded border border-amber-900/50 bg-amber-950/20 p-3 text-xs text-amber-200">
                {verdict.note}
              </div>
            )}

            {(localError || error) && (
              <div className="mt-4 rounded border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-300">
                {localError ?? error}
              </div>
            )}
          </>
        )}

        <p className="mt-8 border-t border-neutral-900 pt-4 text-xs text-neutral-600">
          Prefer no pairing at all? <span className="font-mono text-neutral-500">spraypaint serve --open</span>{" "}
          serves this same interface from the binary itself, on localhost, with no token and no
          browser restrictions.
        </p>
      </div>
    </div>
  );
}
