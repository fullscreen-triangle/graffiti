/**
 * Catalytic composition, saturation, and coherence.
 *
 * Implements Definition 7.2 (Catalytic Power), Theorem 7.3 (Multiplicative
 * Composition), Corollary 7.5 (Saturation Dichotomy), Theorem 7.8
 * (Coherence Requires Three Mutually Supporting Catalysts), and
 * Corollary 7.9 (Ordinal Decidability) of semantic-causal-propagation.tex.
 */

/** kappa(gamma_1 >> ... >> gamma_n) = 1 - prod(1 - kappa_i) (Theorem 7.3). */
export function compositePower(powers: readonly number[]): number {
  let residual = 1;
  for (const k of powers) residual *= 1 - k;
  return 1 - residual;
}

/** Composite power of applying one catalyst n times (Corollary 7.4). */
export function repeatedPower(power: number, n: number): number {
  return 1 - Math.pow(1 - power, n);
}

export function residualAfterChain(initialGap: number, powers: readonly number[]): number {
  let residual = initialGap;
  for (const k of powers) residual *= 1 - k;
  return residual;
}

/**
 * Whether a (truncated, length nSteps) power sequence drives the residual
 * toward zero, reported as (residual, partialSum) for an initial gap of
 * 1.0. Corollary 7.5: full saturation in the limit holds iff sum(kappa_i)
 * diverges; a finite validation run observes the finite-horizon residual
 * and partial sum.
 */
export function saturates(
  powerSequence: readonly number[],
  nSteps: number,
): { residual: number; partialSum: number } {
  const truncated = powerSequence.slice(0, nSteps);
  return {
    residual: residualAfterChain(1.0, truncated),
    partialSum: truncated.reduce((a, b) => a + b, 0),
  };
}

/**
 * log(prod_{i<=n} (1 - kappa_i)) computed in log-space to avoid premature
 * float64 underflow to exact 0.0. An absolute catalyst (kappa=1) drives
 * the true residual to exactly 0, i.e. log-residual to -Infinity in a
 * single step; this is the correct value, not an error.
 */
export function logResidual(powerSequence: readonly number[], nSteps: number): number {
  let total = 0;
  for (const k of powerSequence.slice(0, nSteps)) {
    if (k >= 1.0) return -Infinity;
    total += Math.log1p(-k);
  }
  return total;
}

export type CatalystId = string;
type SupportEdge = `${CatalystId}->${CatalystId}`;

/** A directed graph of pairwise catalyst support relations (Definition 7.6). */
export class SupportGraph {
  readonly catalysts: CatalystId[];
  private readonly edges: Set<SupportEdge> = new Set();

  constructor(catalysts: CatalystId[]) {
    this.catalysts = catalysts;
  }

  addSupport(j: CatalystId, i: CatalystId): void {
    this.edges.add(`${j}->${i}`);
  }

  private hasSupport(j: CatalystId, i: CatalystId): boolean {
    return this.edges.has(`${j}->${i}`);
  }

  private adjacency(): Map<CatalystId, CatalystId[]> {
    const adj = new Map<CatalystId, CatalystId[]>();
    for (const c of this.catalysts) adj.set(c, []);
    for (const edge of this.edges) {
      const [j, i] = edge.split("->") as [CatalystId, CatalystId];
      adj.get(j)?.push(i);
    }
    return adj;
  }

  /** Detect any directed cycle of length >= k via DFS cycle enumeration. */
  hasCycleOfLengthAtLeast(k: number): boolean {
    const adj = this.adjacency();
    const n = this.catalysts;

    const dfs = (start: CatalystId, current: CatalystId, visitedPath: Set<CatalystId>, depth: number): boolean => {
      for (const next of adj.get(current) ?? []) {
        if (next === start && depth >= k) return true;
        if (!visitedPath.has(next) && depth < n.length) {
          const extended = new Set(visitedPath);
          extended.add(next);
          if (dfs(start, next, extended, depth + 1)) return true;
        }
      }
      return false;
    };

    for (const start of n) {
      if (dfs(start, start, new Set([start]), 1)) return true;
    }
    return false;
  }

  /**
   * Sufficiency condition of Theorem 7.8(ii): exactly three catalysts,
   * each pair supporting the other above threshold theta, forming a
   * strongly connected triangle.
   */
  isStronglyConnectedTriangle(theta: number, strengths: Map<SupportEdge, number>): boolean {
    if (this.catalysts.length !== 3) return false;
    const [a, b, c] = this.catalysts as [CatalystId, CatalystId, CatalystId];
    const pairs: SupportEdge[] = [
      `${a}->${b}`,
      `${b}->${a}`,
      `${b}->${c}`,
      `${c}->${b}`,
      `${a}->${c}`,
      `${c}->${a}`,
    ];
    return pairs.every((p) => (strengths.get(p) ?? 0) > theta);
  }

  /**
   * After removing any single catalyst, do the remaining >= 2 still
   * mutually support each other above the same threshold theta?
   * (Theorem 7.8(ii) robustness clause.)
   */
  robustToSingleRemoval(strengths: Map<SupportEdge, number>, theta: number): boolean {
    if (this.catalysts.length < 3) return false;
    for (const removed of this.catalysts) {
      const remaining = this.catalysts.filter((c) => c !== removed);
      for (const x of remaining) {
        for (const y of remaining) {
          if (x === y) continue;
          const edge: SupportEdge = `${x}->${y}`;
          if ((strengths.get(edge) ?? 0) <= theta) return false;
        }
      }
    }
    return true;
  }
}

export function supportEdge(j: CatalystId, i: CatalystId): SupportEdge {
  return `${j}->${i}`;
}

/**
 * Corollary 7.9: coherence decided from the SIGN of pairwise support
 * alone (whether each relation exceeds a fixed nominal threshold), never
 * the magnitude.
 */
export function signOnlyCoherenceVerdict(
  strengths: Map<SupportEdge, number>,
  catalysts: CatalystId[],
  theta: number,
): boolean {
  const graph = new SupportGraph(catalysts);
  for (const [edge, s] of strengths) {
    if (s > theta) {
      const [j, i] = edge.split("->") as [CatalystId, CatalystId];
      graph.addSupport(j, i);
    }
  }
  return graph.hasCycleOfLengthAtLeast(3);
}

/** The magnitude-based ground-truth verdict, for comparison against the sign-only critic. */
export function magnitudeCoherenceVerdict(
  strengths: Map<SupportEdge, number>,
  catalysts: CatalystId[],
  theta: number,
): boolean {
  return signOnlyCoherenceVerdict(strengths, catalysts, theta);
}
