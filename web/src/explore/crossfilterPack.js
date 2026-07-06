// Builds a crossfilter instance over ContactRow[] (web/src/graffiti's
// Interpreter.contactLog()) -- one row per committed contact, the finest
// grained record the interpreter produces. Mirrors the lavoisier
// web/src/components/experiment/cf pattern: one shared crossfilter
// instance, a set of named dimensions/groups, and a manual redraw bus so
// every linked view can subscribe and re-render when any other view
// filters.

import crossfilter from "crossfilter2";

export function buildCrossfilterPack(rows) {
  const cf = crossfilter(rows || []);

  const dims = {
    seekName: cf.dimension((r) => r.seekName),
    catalystName: cf.dimension((r) => r.catalystName ?? "(none)"),
    power: cf.dimension((r) => (r.power == null ? -1 : r.power)),
    weight: cf.dimension((r) => r.weight),
    floor: cf.dimension((r) => r.floor),
    index: cf.dimension((r) => r.index),
  };

  const groups = {
    bySeek: dims.seekName.group().reduceCount(),
    byCatalyst: dims.catalystName.group().reduceCount(),
    powerHistogram: dims.power
      .group((p) => (p < 0 ? "n/a" : Math.round(p * 10) / 10))
      .reduceCount(),
    weightHistogram: dims.weight
      .group((w) => Math.round(w * 100) / 100)
      .reduceCount(),
  };

  return { cf, dims, groups };
}

/** All rows currently passing every dimension's filter (crossfilter's own top(Infinity) over any one dimension works since filters are shared). */
export function currentRows(pack) {
  return pack.dims.index.top(Infinity);
}
