/** Test 02: Individuation by Negation (Theorem 4.2). */

import { Checker } from "./harness";
import { Rng } from "../core/graph";
import { complement, doubleComplementIsIdentity } from "../core/individuation";

export function run() {
  const c = new Checker("02", "Theorem 4.2 (Individuation by Negation)");
  const rng = new Rng(7);
  const mediumSize = 20;
  const whole = new Set(Array.from({ length: mediumSize }, (_, i) => `c${i}`));
  const wholeArr = Array.from(whole);

  for (let trial = 0; trial < 150; trial++) {
    const k = rng.int(1, mediumSize - 1);
    const subsetArr = rng.shuffle(wholeArr).slice(0, k);
    const subset = new Set(subsetArr);

    c.assert(doubleComplementIsIdentity(whole, subset), `co(co(U)) === U for trial ${trial}`);

    const co = complement(whole, subset);
    c.assert(subset.size + co.size === whole.size, `|U|+|co(U)|=|V| for trial ${trial}`);
  }

  return c.result();
}
