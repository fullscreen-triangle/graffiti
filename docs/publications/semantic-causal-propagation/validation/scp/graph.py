"""Contact graphs, separation cost, and the resolution floor.

Implements Definitions 2.1-2.4 and Theorem 3.2 (Resolution Floor) of
``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from .maxflow import FlowNetwork

MEDIUM = "__medium__"


@dataclass
class ContactGraph:
    """A finite weighted graph with a distinguished medium vertex.

    Vertices other than the medium are "claims" (Definition 2.2). Edge
    weights are strictly positive; the floor beta is the infimum edge
    weight actually used when constructing the graph.
    """

    claims: List[str]
    weights: Dict[FrozenSet[str], float] = field(default_factory=dict)
    floor: float = 1e-6

    def edges(self) -> List[Tuple[str, str, float]]:
        return [(tuple(k)[0], tuple(k)[1], v) if len(k) == 2 else (tuple(k)[0], tuple(k)[0], v)
                for k, v in self.weights.items()]

    def _network(self) -> FlowNetwork:
        net = FlowNetwork()
        for pair, w in self.weights.items():
            u, v = tuple(pair) if len(pair) == 2 else (tuple(pair)[0], tuple(pair)[0])
            net.add_undirected_edge(u, v, w)
        return net

    def separation_cost(self, claim: str) -> float:
        """sigma(v): minimum cut weight separating ``claim`` from the medium."""
        net = self._network()
        return net.min_cut_value(claim, MEDIUM)

    def min_cut_side(self, claim: str) -> set:
        """The source-side S*(v) of the minimum cut separating claim from medium."""
        net = self._network()
        _, side = net.max_flow_min_cut(claim, MEDIUM)
        return side

    def alignment(self, x: str, x_star: str) -> float:
        """sigma(x, x*): minimum cut weight separating x from x* (Definition 6.1)."""
        if x == x_star:
            return self.floor
        net = self._network()
        return net.min_cut_value(x, x_star)

    def total_weight(self) -> float:
        return sum(self.weights.values())

    def alignment_score(self, x: str, x_star: str) -> float:
        omega = self.total_weight()
        if omega <= 0:
            return 1.0
        return self.alignment(x, x_star) / omega


def _edge(u: str, v: str) -> FrozenSet[str]:
    return frozenset((u, v))


def random_contact_graph(
    rng: random.Random,
    n_claims: int,
    floor: float,
    max_extra_weight: float = 5.0,
    edge_prob: float = 0.3,
) -> ContactGraph:
    """Construct a random connected contact graph with every weight >= floor."""
    claims = [f"c{i}" for i in range(n_claims)]
    weights: Dict[FrozenSet[str], float] = {}

    # Ensure connectivity: every claim gets an edge to the medium.
    for c in claims:
        weights[_edge(c, MEDIUM)] = floor + rng.random() * max_extra_weight

    # Random additional claim-claim contacts.
    for i, u in enumerate(claims):
        for v in claims[i + 1 :]:
            if rng.random() < edge_prob:
                weights[_edge(u, v)] = floor + rng.random() * max_extra_weight

    return ContactGraph(claims=claims, weights=weights, floor=floor)


def two_cluster_graph(
    rng: random.Random,
    cluster_size: int,
    floor: float,
    internal_weight: float = 10.0,
    bridge_weight: float | None = None,
) -> Tuple[ContactGraph, List[str], List[str]]:
    """Two densely-internally-bound clusters A, B, each with one contact to the medium.

    Used for: (a) Theorem 5.4 (identity is a region, non-singleton minimum
    cut), and (b) Theorem 8.2 (closure is stronger than a confidence
    threshold): a propagation into cluster A alone can satisfy any fixed
    confidence threshold trivially, while cluster B remains reachable and
    endpoint-distinguishable.
    """
    if bridge_weight is None:
        bridge_weight = floor

    a_claims = [f"a{i}" for i in range(cluster_size)]
    b_claims = [f"b{i}" for i in range(cluster_size)]
    claims = a_claims + b_claims
    weights: Dict[FrozenSet[str], float] = {}

    # Dense internal contacts within each cluster.
    for i, u in enumerate(a_claims):
        for v in a_claims[i + 1 :]:
            weights[_edge(u, v)] = internal_weight + rng.random()
    for i, u in enumerate(b_claims):
        for v in b_claims[i + 1 :]:
            weights[_edge(u, v)] = internal_weight + rng.random()

    # Each cluster touches the medium through a single representative claim,
    # at the floor weight (the "single catalyst" bottleneck of the proof).
    weights[_edge(a_claims[0], MEDIUM)] = bridge_weight
    weights[_edge(b_claims[0], MEDIUM)] = bridge_weight

    # The rest of each cluster is NOT directly adjacent to the medium;
    # give every claim in the graph at least the floor by inheriting
    # connectivity through the cluster's internal weights (already >= floor
    # since internal_weight >= floor by construction below).
    graph = ContactGraph(claims=claims, weights=weights, floor=floor)
    return graph, a_claims, b_claims
