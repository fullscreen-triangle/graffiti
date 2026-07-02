/** Test 07: Saturation Dichotomy (Corollary 7.5). Mirrors the corrected
 * log-space horizon-comparison methodology used in the Python suite
 * (see the manuscript's Remark: Two Sharpenings During Validation). */

import { Checker } from "./harness";
import { logResidual } from "../core/catalysis";

const N_STEPS = 500;
const HORIZONS = [10, 50, 250, 500];

export function run() {
  const c = new Checker("07", "Corollary 7.5 (Saturation Dichotomy)");

  const sequences: Record<string, { powers: number[]; divergent: boolean }> = {
    constant: { powers: Array.from({ length: N_STEPS }, () => 0.1), divergent: true },
    harmonic: { powers: Array.from({ length: N_STEPS }, (_, i) => 1 / (i + 2)), divergent: true },
    geometric: { powers: Array.from({ length: N_STEPS }, (_, i) => Math.pow(2, -(i + 1))), divergent: false },
    inverseSquare: { powers: Array.from({ length: N_STEPS }, (_, i) => 1 / Math.pow(i + 2, 2)), divergent: false },
  };

  for (const [name, { powers, divergent }] of Object.entries(sequences)) {
    const values = HORIZONS.map((h) => logResidual(powers, h));
    const strictlyDecreasing = values.every((v, i) => i === 0 || v < values[i - 1]! - 1e-9);
    const lastTwoIncrement = Math.abs(values[values.length - 1]! - values[values.length - 2]!);

    if (divergent) {
      c.assert(strictlyDecreasing && lastTwoIncrement > 1e-2, `${name}: log-residual diverges without bound`);
    } else {
      const relativeIncrement = lastTwoIncrement / Math.max(Math.abs(values[values.length - 1]!), 1e-12);
      c.assert(relativeIncrement < 1e-2, `${name}: log-residual converges to a finite limit`);
    }
  }

  return c.result();
}
