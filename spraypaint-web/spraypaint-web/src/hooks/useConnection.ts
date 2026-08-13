"use client";

// ── Deciding which server this page talks to ──
//
// On mount we probe our *own* origin for `/api/health`. If it answers, the
// binary is serving this page and there is nothing to configure — no URL, no
// token, no browser caveats. That probe is why the pairing screen never appears
// for the `spraypaint serve --open` path, which is the majority case.
//
// If it does not answer, this page is hosted (Vercel) and the only way it can do
// anything is a paired local binary. `status` then drives the pairing screen.

import { useCallback, useEffect, useState } from "react";

import { makeApi } from "@/lib/api";
import {
  type Connection,
  SAME_ORIGIN,
  clearConnection,
  isSameOrigin,
  loadConnection,
  saveConnection,
} from "@/lib/connection";

export type ConnectionStatus =
  /** The same-origin probe has not finished. */
  | "probing"
  /** Served by the binary. Nothing to configure. */
  | "same-origin"
  /** Hosted page with a working paired connection. */
  | "paired"
  /** Hosted page, no usable connection yet. Show the pairing screen. */
  | "unpaired";

export interface ConnectionState {
  status: ConnectionStatus;
  connection: Connection;
  /** Why the last connect attempt failed. Cleared on success. */
  error: string | null;
  connecting: boolean;
}

export function useConnection() {
  const [status, setStatus] = useState<ConnectionStatus>("probing");
  const [connection, setConnection] = useState<Connection>(SAME_ORIGIN);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  /**
   * Try a connection by calling `/api/health` through it.
   *
   * `health` is the right probe precisely because it takes no lock and loads no
   * index: it answers on a repo with nothing indexed, so a successful pair is
   * not conditional on the user having run `spraypaint index` first.
   */
  const attempt = useCallback(async (c: Connection): Promise<string | null> => {
    try {
      await makeApi(c).health();
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      // 1. Are we served by the binary?
      if ((await attempt(SAME_ORIGIN)) === null) {
        if (cancelled) return;
        setConnection(SAME_ORIGIN);
        setStatus("same-origin");
        return;
      }
      if (cancelled) return;

      // 2. Hosted. Restore a connection from this tab, if one is stored and
      //    still works — the server may have been stopped since, which revokes
      //    the token, so a stored value is a hypothesis rather than a fact.
      const stored = loadConnection();
      if (stored && !isSameOrigin(stored)) {
        if ((await attempt(stored)) === null) {
          if (cancelled) return;
          setConnection(stored);
          setStatus("paired");
          return;
        }
        clearConnection();
      }
      if (cancelled) return;
      setStatus("unpaired");
    })();
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  /** Connect with a URL and token entered on the pairing screen. */
  const connect = useCallback(
    async (c: Connection): Promise<boolean> => {
      setConnecting(true);
      setError(null);
      const failure = await attempt(c);
      setConnecting(false);
      if (failure !== null) {
        setError(failure);
        return false;
      }
      saveConnection(c);
      setConnection(c);
      setStatus("paired");
      return true;
    },
    [attempt]
  );

  const disconnect = useCallback(() => {
    clearConnection();
    setConnection(SAME_ORIGIN);
    setStatus("unpaired");
    setError(null);
  }, []);

  return { status, connection, error, connecting, connect, disconnect };
}
