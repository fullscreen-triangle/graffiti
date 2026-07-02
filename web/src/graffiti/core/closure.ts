/**
 * Closure: when a search is finished.
 *
 * Implements Definition 8.1 (Closure), Theorem 8.2 (Closure is Strictly
 * Stronger than a Confidence Threshold), and Theorem 8.3 (Convergent
 * Closure or Honest Decline) of semantic-causal-propagation.tex.
 */

import type { ContactGraph } from "./graph";
import type { NodeId } from "./maxflow";

/**
 * A genuine endpoint invariant of Theorem 6.4: the minimum-cut side
 * S*(target) itself (its exact claim membership), not merely its weight.
 * Two targets are endpoint-indistinguishable only if they share the same
 * minimum-cut side -- distinct clusters with numerically equal bridge
 * weight but disjoint membership are correctly kept apart.
 */
function cutSideKey(graph: ContactGraph, target: NodeId): string {
  const side = Array.from(graph.minCutSide(target)).sort();
  return side.join(",");
}

/**
 * Partition reached targets into endpoint-indistinguishable classes
 * (Corollary 6.5, "the admissible set is a class, not a path"): two
 * targets are in the same class iff their minimum cut against the medium
 * -- side membership, the full endpoint invariant of Theorem 6.4 --
 * agrees.
 *
 * This is the exact graph-theoretic test used to validate Theorem 8.2 on
 * hand-constructed instances (see twoClusterGraph). It is intentionally
 * strict: on a densely connected runtime graph where every claim is
 * wired to the medium (Definition 2.2) and propagation steps introduce
 * many parallel floor-weight paths, minimum-cut values can coincide for
 * genuinely distinct claims purely from graph density. For runtime
 * closure decisions over such graphs, use `equivalenceClassesByIdentity`
 * instead.
 */
export function equivalenceClasses(graph: ContactGraph, reachedTargets: readonly NodeId[]): NodeId[][] {
  const buckets = new Map<string, NodeId[]>();
  for (const t of reachedTargets) {
    const key = cutSideKey(graph, t);
    const bucket = buckets.get(key);
    if (bucket) bucket.push(t);
    else buckets.set(key, [t]);
  }
  return Array.from(buckets.values());
}

/**
 * Partition reached targets by direct claim identity: two targets are in
 * the same class iff they are the literal same claim. This is the
 * operational reading of "resolves to a claim-region already reached"
 * (Definition 8.1) appropriate for a runtime interpreter, where a
 * catalyst either confirms an existing claim (identity match) or
 * resolves to a materially different one (a distinct string) --
 * independent of incidental graph-density effects on minimum-cut values.
 */
export function equivalenceClassesByIdentity(reachedTargets: readonly NodeId[]): NodeId[][] {
  const buckets = new Map<NodeId, NodeId[]>();
  for (const t of reachedTargets) {
    const bucket = buckets.get(t);
    if (bucket) bucket.push(t);
    else buckets.set(t, [t]);
  }
  return Array.from(buckets.values());
}

/**
 * A naive confidence check: terminal alignment to itself (always the
 * floor, by construction of a completed propagation) compared to theta.
 * Demonstrates Theorem 8.2: this is satisfied trivially by any completed
 * propagation, independent of whether other catalysts would diverge.
 */
export function confidenceThresholdMet(graph: ContactGraph, terminal: NodeId, theta: number): boolean {
  const selfAlignment = graph.alignment(terminal, terminal);
  const omega = graph.totalWeight();
  const confidence = 1 - (omega > 0 ? selfAlignment / omega : 0);
  return confidence >= theta;
}

/** Definition 8.1: closed iff no available-but-uninvoked catalyst adds a new equivalence class. */
export function isClosed(
  availableCatalystTargets: readonly NodeId[],
  classesSoFar: readonly NodeId[][],
  graph: ContactGraph,
): boolean {
  const existingKeys = new Set<string>();
  for (const cls of classesSoFar) {
    const representative = cls[0];
    if (representative !== undefined) existingKeys.add(cutSideKey(graph, representative));
  }

  for (const target of availableCatalystTargets) {
    if (!existingKeys.has(cutSideKey(graph, target))) return false;
  }
  return true;
}

/** Identity-based counterpart of `isClosed`, paired with `equivalenceClassesByIdentity`. */
export function isClosedByIdentity(
  availableCatalystTargets: readonly NodeId[],
  classesSoFar: readonly NodeId[][],
): boolean {
  const existing = new Set<NodeId>();
  for (const cls of classesSoFar) {
    for (const member of cls) existing.add(member);
  }
  return availableCatalystTargets.every((t) => existing.has(t));
}

export type SearchOutcomeState = "convergent" | "declined";

export interface SearchOutcome {
  state: SearchOutcomeState;
  classes: NodeId[][];
}

/**
 * Theorem 8.3: every search over a finite catalyst registry terminates
 * in exactly one of two states -- convergent closure (a single
 * equivalence class) or contested closure / decline (more than one).
 */
export function resolveOutcome(classes: NodeId[][]): SearchOutcome {
  return {
    state: classes.length === 1 ? "convergent" : "declined",
    classes,
  };
}
