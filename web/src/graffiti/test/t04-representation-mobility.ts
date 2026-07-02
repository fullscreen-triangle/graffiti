/** Test 04: Representation Mobility (Theorem 5.2). */

import { Checker } from "./harness";
import { Rng } from "../core/graph";
import { committedRecordAfterSwitch, isOnShell, representationMean, sampleRepresentation } from "../core/representation";

export function run() {
  const c = new Checker("04", "Theorem 5.2 (Representation Mobility)");
  const rng = new Rng(11);

  let totalComponents = 0;
  let offShellCount = 0;

  for (let trial = 0; trial < 200; trial++) {
    const alignment = rng.float(0.01, 0.99);
    const dimension = rng.int(2, 8);
    const components = sampleRepresentation(rng, alignment, dimension);

    const mean = representationMean(components);
    c.assertClose(mean, alignment, 1e-9, `representation mean recovers target on trial ${trial}`);

    for (const comp of components) {
      totalComponents++;
      if (!isOnShell(comp)) offShellCount++;
    }

    const recordBefore = rng.int(0, 5000);
    const recordAfter = committedRecordAfterSwitch(recordBefore);
    c.assert(recordAfter === recordBefore, "representation switch commits no new cut");
  }

  c.assert(offShellCount / totalComponents > 0.5, `off-shell fraction (${(offShellCount / totalComponents).toFixed(3)}) is the typical case`);

  return c.result();
}
