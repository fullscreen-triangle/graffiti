// Shared crossfilter instance + a manual redraw bus, following the
// lavoisier web/src/components/experiment/cf/CrossfilterContext.js
// pattern: one provider owns the crossfilter pack; any number of leaf
// chart components subscribe a redraw callback on mount and call
// redrawAll() after they change a filter, so every other chart re-reads
// its own group and re-renders without a virtual-DOM diff of the chart
// body.

import React, { createContext, useContext, useMemo, useRef, useCallback, useState } from "react";
import { buildCrossfilterPack } from "./crossfilterPack";

const Ctx = createContext(null);

export function CrossfilterProvider({ rows, children }) {
  const pack = useMemo(() => buildCrossfilterPack(rows), [rows]);
  const subscribers = useRef(new Set());
  const [, setTick] = useState(0);

  const subscribe = useCallback((fn) => {
    subscribers.current.add(fn);
    return () => subscribers.current.delete(fn);
  }, []);

  const redrawAll = useCallback(() => {
    setTick((t) => t + 1);
    for (const fn of subscribers.current) fn();
  }, []);

  const value = useMemo(() => ({ pack, subscribe, redrawAll }), [pack, subscribe, redrawAll]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useCrossfilter() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useCrossfilter must be used inside a CrossfilterProvider");
  return ctx;
}

export function useChartRedraw(redrawFn) {
  const { subscribe } = useCrossfilter();
  React.useEffect(() => subscribe(redrawFn), [subscribe, redrawFn]);
}
