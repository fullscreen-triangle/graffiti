/** Test 06: Multiplicative Composition Law (Theorem 7.3). */

import { Checker } from "./harness";
import { Rng } from "../core/graph";
import { compositePower, residualAfterChain } from "../core/catalysis";

export function run() {
  const c = new Checker("06", "Theorem 7.3 (Multiplicative Composition)");
  const rng = new Rng(23);

  for (let trial = 0; trial < 300; trial++) {
    const n = rng.int(1, 8);
    const powers = Array.from({ length: n }, () => rng.float(0, 0.99));
    const predicted = compositePower(powers);
    const residual = residualAfterChain(1.0, powers);
    const measured = 1.0 - residual;
    c.assertClose(measured, predicted, 1e-9, `composite power matches closed form on trial ${trial}`);
  }

  return c.result();
}
