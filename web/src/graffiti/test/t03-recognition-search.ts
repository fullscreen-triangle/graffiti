/** Test 03: Recognition/Search Identity (Theorem 4.5). */

import { Checker } from "./harness";
import { Decoder } from "../core/individuation";

export function run() {
  const c = new Checker("03", "Theorem 4.5 (Recognition/Search Identity)");

  const mapping = new Map<string, string>();
  for (let claimIdx = 0; claimIdx < 12; claimIdx++) {
    for (let q = 0; q < 6; q++) {
      mapping.set(`q${claimIdx}_${q}`, `claim${claimIdx}`);
    }
  }
  const decoder = new Decoder(mapping);

  for (const [q] of mapping) {
    const v = decoder.decode(q);
    c.assert(decoder.fibre(v).has(q), `Dec(${q})=${v} implies ${q} in Proj(${v})`);
  }

  const rebuilt = decoder.recoverFromFibres();
  c.assert(decoder.matchesMapping(rebuilt), "decoder exactly recovered from fibre partition");

  return c.result();
}
