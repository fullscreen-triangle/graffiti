/** Test 05: Convergence Admissibility and Path Opacity (Theorem 6.3, 6.4). */

import { Checker } from "./harness";
import { Rng, randomContactGraph } from "../core/graph";
import { endpointInvariants, isConvergent, randomInteriorVariant } from "../core/propagation";

export function run() {
  const c = new Checker("05", "Theorem 6.3/6.4 (Convergence Admissibility, Path Opacity)");
  const rng = new Rng(19);

  const nClaims = 14;
  const floor = 0.2;
  const graph = randomContactGraph({ nClaims, floor, edgeProb: 0.35, rng });
  const seed = graph.claims[0]!;
  const target = graph.claims[graph.claims.length - 1]!;

  const variants = Array.from({ length: 40 }, () => {
    const interiorLen = rng.int(0, Math.min(6, nClaims - 2));
    return randomInteriorVariant(rng, seed, target, graph.claims, interiorLen);
  });

  for (const prop of variants) {
    c.assert(isConvergent(prop, target), "propagation terminating at target is admissible");
  }

  const invariants = variants.map((p) => endpointInvariants(graph, p));
  const reference = invariants[0]!;
  for (const inv of invariants.slice(1)) {
    c.assert(inv.seed === reference.seed, "endpoint seed invariant across interiors");
    c.assert(inv.target === reference.target, "endpoint target invariant across interiors");
    c.assertClose(inv.targetMinCut, reference.targetMinCut, 1e-9, "endpoint min-cut invariant across interiors");
  }

  return c.result();
}
