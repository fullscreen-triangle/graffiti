/** Test 08: Coherence Requires a Triangle (Theorem 7.8) and Ordinal
 * Decidability (Corollary 7.9). */

import { Checker } from "./harness";
import { Rng } from "../core/graph";
import { SupportGraph, magnitudeCoherenceVerdict, signOnlyCoherenceVerdict, supportEdge, type CatalystId } from "../core/catalysis";

const THETA = 0.5;

function buildAcyclic(rng: Rng, n: number) {
  const catalysts: CatalystId[] = Array.from({ length: n }, (_, i) => `g${i}`);
  const order = rng.shuffle(catalysts);
  const graph = new SupportGraph(catalysts);
  const strengths = new Map<ReturnType<typeof supportEdge>, number>();
  for (let idx = 0; idx < order.length; idx++) {
    for (let j = idx + 1; j < order.length; j++) {
      if (rng.next() < 0.6) {
        const s = rng.float(0, 1);
        const edge = supportEdge(order[idx]!, order[j]!);
        strengths.set(edge, s);
        if (s > THETA) graph.addSupport(order[idx]!, order[j]!);
      }
    }
  }
  return { graph, strengths };
}

function buildTriangle(rng: Rng, aboveTheta: boolean) {
  const catalysts: CatalystId[] = ["g0", "g1", "g2"];
  const graph = new SupportGraph(catalysts);
  const strengths = new Map<ReturnType<typeof supportEdge>, number>();
  for (const a of catalysts) {
    for (const b of catalysts) {
      if (a === b) continue;
      const s = aboveTheta ? rng.float(THETA + 1e-3, 1) : rng.float(0, THETA - 1e-3);
      const edge = supportEdge(a, b);
      strengths.set(edge, s);
      if (s > THETA) graph.addSupport(a, b);
    }
  }
  return { graph, strengths };
}

export function run() {
  const c = new Checker("08", "Theorem 7.8/Corollary 7.9 (Coherence Triangle, Ordinal Decidability)");
  const rng = new Rng(29);

  for (let i = 0; i < 100; i++) {
    const n = rng.int(2, 6);
    const { graph, strengths } = buildAcyclic(rng, n);
    c.assert(!graph.robustToSingleRemoval(strengths, THETA), "acyclic support graph fails robustness");
  }

  for (let i = 0; i < 100; i++) {
    const { graph, strengths } = buildTriangle(rng, true);
    c.assert(graph.isStronglyConnectedTriangle(THETA, strengths), "constructed triangle is strongly connected");
    c.assert(graph.robustToSingleRemoval(strengths, THETA), "constructed triangle survives single removal");
  }

  for (let i = 0; i < 100; i++) {
    const { graph, strengths } = buildTriangle(rng, false);
    c.assert(!graph.robustToSingleRemoval(strengths, THETA), "sub-threshold triangle fails robustness");
  }

  let agreementCount = 0;
  const totalChains = 500;
  for (let i = 0; i < totalChains; i++) {
    const n = rng.int(2, 6);
    const catalysts: CatalystId[] = Array.from({ length: n }, (_, k) => `g${k}`);
    const strengths = new Map<ReturnType<typeof supportEdge>, number>();
    for (const a of catalysts) {
      for (const b of catalysts) {
        if (a !== b && rng.next() < 0.7) {
          strengths.set(supportEdge(a, b), rng.float(0, 1));
        }
      }
    }
    const signVerdict = signOnlyCoherenceVerdict(strengths, catalysts, THETA);
    const magVerdict = magnitudeCoherenceVerdict(strengths, catalysts, THETA);
    if (signVerdict === magVerdict) agreementCount++;
  }
  c.assert(agreementCount === totalChains, `sign-only critic agrees with magnitude verdict on all ${totalChains} chains`);

  return c.result();
}
