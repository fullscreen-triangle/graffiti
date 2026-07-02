/**
 * Representation mobility.
 *
 * Implements Definition 5.1, Theorem 5.2 (Representation Mobility), and
 * Theorem 5.4 (Receiver-Relative Decoding is Not Error) of
 * semantic-causal-propagation.tex.
 *
 * A representation of a claim is a tuple (s_1,...,s_N) satisfying the
 * averaging constraint (1/N) * sum(s_j) = alignment, with components
 * otherwise free in R (in particular, a component may lie outside (0,1]).
 * Paraphrases of a fixed claim -- "Peter", "a Harvard medical student", a
 * much longer clinical description -- are representations in this sense:
 * they satisfy the same averaging constraint and switching between them
 * commits no new search step.
 */

import type { Rng } from "./graph";

/** Sample a representation tuple satisfying the averaging constraint. */
export function sampleRepresentation(rng: Rng, alignment: number, dimension: number): number[] {
  const components: number[] = [];
  for (let i = 0; i < dimension - 1; i++) {
    components.push(rng.float(-5.0, 5.0));
  }
  const sum = components.reduce((a, b) => a + b, 0);
  const last = dimension * alignment - sum;
  components.push(last);
  return rng.shuffle(components);
}

export function representationMean(components: readonly number[]): number {
  if (components.length === 0) return 0;
  return components.reduce((a, b) => a + b, 0) / components.length;
}

export function isOnShell(component: number): boolean {
  return component > 0 && component <= 1;
}

/**
 * A representation switch commits no new contact edge (Theorem 5.2(ii));
 * the committed record is unchanged.
 */
export function committedRecordAfterSwitch(recordBefore: number): number {
  return recordBefore;
}
