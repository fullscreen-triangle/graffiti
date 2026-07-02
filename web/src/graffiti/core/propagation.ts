/**
 * Convergence admissibility and path opacity.
 *
 * Implements Definition 6.2 (Search Process Graph; Propagation),
 * Theorem 6.3 (Convergence Admissibility), and Theorem 6.4 (Path Opacity)
 * of semantic-causal-propagation.tex.
 */

import type { ContactGraph } from "./graph";
import type { Rng } from "./graph";
import type { NodeId } from "./maxflow";

/** A walk (v0, e1, v1, ..., ek, vk) realised as a sequence of claims. */
export class Propagation {
  readonly claims: NodeId[];

  constructor(claims: NodeId[]) {
    if (claims.length === 0) {
      throw new Error("A propagation must contain at least the seed claim");
    }
    this.claims = claims;
  }

  get seed(): NodeId {
    return this.claims[0]!;
  }

  get terminal(): NodeId {
    return this.claims[this.claims.length - 1]!;
  }

  interior(): NodeId[] {
    return this.claims.slice(1, -1);
  }
}

/** Theorem 6.3: admissible iff the terminal claim equals the target. */
export function isConvergent(prop: Propagation, target: NodeId): boolean {
  return prop.terminal === target;
}

export interface EndpointInvariants {
  seed: NodeId;
  target: NodeId;
  terminalSelfAlignment: number;
  targetMinCut: number;
}

/**
 * Compute the endpoint-only invariants of Theorem 6.4: seed, target,
 * terminal alignment to itself, and the minimum-cut value of the target
 * against the medium. None of these read the interior of the walk.
 */
export function endpointInvariants(graph: ContactGraph, prop: Propagation): EndpointInvariants {
  const target = prop.terminal;
  return {
    seed: prop.seed,
    target,
    terminalSelfAlignment: graph.alignment(target, target),
    targetMinCut: graph.separationCost(target),
  };
}

/**
 * Construct a propagation from seed to target with a randomly chosen
 * interior drawn from `pool`, of the given interior length. Different
 * calls with the same seed/target and different interiors instantiate the
 * two propagations of Theorem 6.4 (path opacity).
 */
export function randomInteriorVariant(
  rng: Rng,
  seed: NodeId,
  target: NodeId,
  pool: readonly NodeId[],
  length: number,
): Propagation {
  const candidates = pool.filter((c) => c !== seed && c !== target);
  const shuffled = rng.shuffle(candidates);
  const interior = shuffled.slice(0, length);
  return new Propagation([seed, ...interior, target]);
}
