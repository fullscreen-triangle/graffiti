/** Test 09: Closure vs. Confidence Threshold (Theorem 8.2) and Convergent
 * Closure or Honest Decline (Theorem 8.3). */

import { Checker } from "./harness";
import { Rng, twoClusterGraph } from "../core/graph";
import { confidenceThresholdMet, equivalenceClasses, isClosed, resolveOutcome } from "../core/closure";

const THETA = 0.5;

export function run() {
  const c = new Checker("09", "Theorem 8.2/8.3 (Closure vs. Threshold, Convergent/Declined)");
  const rng = new Rng(31);

  for (let trial = 0; trial < 150; trial++) {
    const clusterSize = rng.int(2, 6);
    const floor = 0.05 + rng.next() * 0.5;
    const { graph, aClaims, bClaims } = twoClusterGraph({ clusterSize, floor, rng });
    const targetA = aClaims[0]!;
    const targetB = bClaims[0]!;

    const thresholdMet = confidenceThresholdMet(graph, targetA, THETA);
    const classesSoFar = equivalenceClasses(graph, [targetA]);
    const closed = isClosed([targetB], classesSoFar, graph);

    c.assert(thresholdMet && !closed, `trial ${trial}: threshold met but search not closed (second cluster available)`);

    const allClasses = equivalenceClasses(graph, [targetA, targetB]);
    const outcome = resolveOutcome(allClasses);
    c.assert(outcome.state === "declined", `trial ${trial}: two structurally distinct clusters resolve to contested closure`);
  }

  return c.result();
}
