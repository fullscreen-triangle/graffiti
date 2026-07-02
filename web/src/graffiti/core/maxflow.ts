/**
 * Exact maximum-flow / minimum-cut computation (Edmonds-Karp), from
 * scratch, with no external graph library.
 *
 * This mirrors, line for line in spirit, the Python reference
 * implementation in
 * docs/publications/semantic-causal-propagation/validation/scp/maxflow.py,
 * which the manuscript's numerical validation suite is built on. Every
 * downstream claim in this library (the floor, alignment, path opacity,
 * closure) reduces to a computation performed by this file, so it is kept
 * dependency-free and exactly checkable.
 */

export type NodeId = string;

interface AdjacencyMap {
  [node: string]: Map<NodeId, number>;
}

/**
 * A directed capacitated graph supporting exact max-flow / min-cut.
 *
 * Built from an undirected weighted contact graph by installing both
 * directed arcs at the given capacity -- the standard construction for
 * computing an undirected minimum cut via max-flow / min-cut duality
 * (Ford & Fulkerson 1956; Menger 1927).
 */
export class FlowNetwork {
  private capacity: AdjacencyMap = {};
  private nodeSet: Set<NodeId> = new Set();

  addUndirectedEdge(u: NodeId, v: NodeId, weight: number): void {
    this.nodeSet.add(u);
    this.nodeSet.add(v);
    this.ensure(u).set(v, (this.ensure(u).get(v) ?? 0) + weight);
    this.ensure(v).set(u, (this.ensure(v).get(u) ?? 0) + weight);
  }

  nodes(): NodeId[] {
    return Array.from(this.nodeSet);
  }

  private ensure(node: NodeId): Map<NodeId, number> {
    if (!this.capacity[node]) {
      this.capacity[node] = new Map();
    }
    return this.capacity[node]!;
  }

  private bfsAugmentingPath(
    residual: AdjacencyMap,
    source: NodeId,
    sink: NodeId,
  ): Array<[NodeId, NodeId]> | null {
    const parent = new Map<NodeId, NodeId>();
    parent.set(source, source);
    const queue: NodeId[] = [source];
    let head = 0;

    while (head < queue.length) {
      const u = queue[head++]!;
      const neighbours = residual[u];
      if (!neighbours) continue;
      for (const [v, cap] of neighbours) {
        if (cap > 1e-12 && !parent.has(v)) {
          parent.set(v, u);
          if (v === sink) {
            const path: Array<[NodeId, NodeId]> = [];
            let node = sink;
            while (node !== source) {
              const p = parent.get(node)!;
              path.push([p, node]);
              node = p;
            }
            path.reverse();
            return path;
          }
          queue.push(v);
        }
      }
    }
    return null;
  }

  /**
   * Returns (maxFlowValue, reachableSetInResidualGraph). The reachable
   * set from `source` in the final residual graph is the source-side S
   * of a minimum cut (Ford & Fulkerson 1956; Edmonds & Karp 1972).
   */
  maxFlowMinCut(source: NodeId, sink: NodeId): { flow: number; reachable: Set<NodeId> } {
    const residual: AdjacencyMap = {};
    for (const u of this.nodeSet) {
      residual[u] = new Map();
    }
    for (const u of this.nodeSet) {
      const neighbours = this.capacity[u];
      if (!neighbours) continue;
      for (const [v, cap] of neighbours) {
        residual[u]!.set(v, (residual[u]!.get(v) ?? 0) + cap);
        if (!residual[v]!.has(u)) {
          residual[v]!.set(u, residual[v]!.get(u) ?? 0);
        }
      }
    }

    let flowValue = 0;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const path = this.bfsAugmentingPath(residual, source, sink);
      if (path === null) break;

      let bottleneck = Infinity;
      for (const [u, v] of path) {
        bottleneck = Math.min(bottleneck, residual[u]!.get(v) ?? 0);
      }
      for (const [u, v] of path) {
        residual[u]!.set(v, (residual[u]!.get(v) ?? 0) - bottleneck);
        residual[v]!.set(u, (residual[v]!.get(u) ?? 0) + bottleneck);
      }
      flowValue += bottleneck;
    }

    const reachable = new Set<NodeId>([source]);
    const queue: NodeId[] = [source];
    let head = 0;
    while (head < queue.length) {
      const u = queue[head++]!;
      const neighbours = residual[u];
      if (!neighbours) continue;
      for (const [v, cap] of neighbours) {
        if (cap > 1e-9 && !reachable.has(v)) {
          reachable.add(v);
          queue.push(v);
        }
      }
    }

    return { flow: flowValue, reachable };
  }

  minCutValue(source: NodeId, sink: NodeId): number {
    return this.maxFlowMinCut(source, sink).flow;
  }
}
