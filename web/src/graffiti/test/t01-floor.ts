/** Test 01: Resolution Floor (Theorem 3.2). */

import { Checker } from "./harness";
import { Rng, randomContactGraph } from "../core/graph";

export function run() {
  const c = new Checker("01", "Theorem 3.2 (Resolution Floor)");
  const rng = new Rng(42);

  for (let trial = 0; trial < 60; trial++) {
    const nClaims = rng.int(3, 10);
    const floor = 0.05 + rng.next() * 1.5;
    const graph = randomContactGraph({ nClaims, floor, edgeProb: 0.3, rng });
    for (const claim of graph.claims) {
      const sigma = graph.separationCost(claim);
      c.assert(sigma >= floor - 1e-9, `sigma(${claim})=${sigma} >= floor=${floor}`);
    }
  }

  return c.result();
}
