"""Exact maximum-flow / minimum-cut computation (Edmonds-Karp), from scratch.

No external graph library is used, in keeping with the paper's discipline
that every object is a finite weighted graph checkable by direct
computation (Ford-Fulkerson / Edmonds-Karp, Menger's theorem).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Hashable, Iterable, Tuple

Edge = Tuple[Hashable, Hashable]


class FlowNetwork:
    """A directed capacitated graph supporting exact max-flow / min-cut.

    Built from an undirected weighted contact graph by installing both
    directed arcs at the given capacity (the standard construction for
    computing an undirected minimum cut via max-flow / min-cut duality).
    """

    def __init__(self) -> None:
        self._capacity: Dict[Hashable, Dict[Hashable, float]] = defaultdict(dict)
        self._nodes: set = set()

    def add_undirected_edge(self, u: Hashable, v: Hashable, weight: float) -> None:
        self._nodes.add(u)
        self._nodes.add(v)
        self._capacity[u][v] = self._capacity[u].get(v, 0.0) + weight
        self._capacity[v][u] = self._capacity[v].get(u, 0.0) + weight

    def nodes(self) -> Iterable[Hashable]:
        return iter(self._nodes)

    def _bfs_augmenting_path(
        self, residual: Dict[Hashable, Dict[Hashable, float]], source: Hashable, sink: Hashable
    ):
        parent: Dict[Hashable, Hashable] = {source: source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in residual[u].items():
                if cap > 1e-12 and v not in parent:
                    parent[v] = u
                    if v == sink:
                        path = []
                        node = sink
                        while node != source:
                            path.append((parent[node], node))
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append(v)
        return None

    def max_flow_min_cut(self, source: Hashable, sink: Hashable):
        """Return (max_flow_value, reachable_set_in_residual_graph).

        The reachable set from ``source`` in the final residual graph is
        the source-side ``S`` of a minimum cut (standard max-flow/min-cut
        construction, Ford & Fulkerson 1956; Edmonds & Karp 1972).
        """
        residual: Dict[Hashable, Dict[Hashable, float]] = defaultdict(dict)
        for u in self._nodes:
            for v, cap in self._capacity[u].items():
                residual[u][v] = residual[u].get(v, 0.0) + cap
                residual[v].setdefault(u, residual[v].get(u, 0.0))

        flow_value = 0.0
        while True:
            path = self._bfs_augmenting_path(residual, source, sink)
            if path is None:
                break
            bottleneck = min(residual[u][v] for u, v in path)
            for u, v in path:
                residual[u][v] -= bottleneck
                residual[v][u] = residual[v].get(u, 0.0) + bottleneck
            flow_value += bottleneck

        # Reachable set in the final residual graph = source side of min cut.
        reachable = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in residual[u].items():
                if cap > 1e-9 and v not in reachable:
                    reachable.add(v)
                    queue.append(v)
        return flow_value, reachable

    def cut_weight(self, side: set) -> float:
        """Weight of the edge boundary of ``side`` (sum of crossing edge weights)."""
        total = 0.0
        for u in side:
            for v, cap in self._capacity[u].items():
                if v not in side:
                    total += cap
        return total / 1.0  # capacities already store full undirected weight per direction

    def min_cut_value(self, source: Hashable, sink: Hashable) -> float:
        value, _ = self.max_flow_min_cut(source, sink)
        return value
