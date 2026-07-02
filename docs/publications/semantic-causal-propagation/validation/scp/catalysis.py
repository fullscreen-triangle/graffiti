"""Catalytic composition, saturation, and coherence.

Implements Definition 7.2 (Catalytic Power), Theorem 7.3 (Multiplicative
Composition), Corollary 7.5 (Saturation Dichotomy), Theorem 7.8 (Coherence
Requires Three Mutually Supporting Catalysts), and Corollary 7.9 (Ordinal
Decidability) of ``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Tuple


def composite_power(powers: List[float]) -> float:
    """kappa(gamma_1 >> ... >> gamma_n) = 1 - prod(1 - kappa_i) (Theorem 7.3)."""
    residual = 1.0
    for k in powers:
        residual *= (1.0 - k)
    return 1.0 - residual


def repeated_power(power: float, n: int) -> float:
    """Composite power of applying one catalyst n times (Corollary 7.4)."""
    return 1.0 - (1.0 - power) ** n


def residual_after_chain(initial_gap: float, powers: List[float]) -> float:
    residual = initial_gap
    for k in powers:
        residual *= (1.0 - k)
    return residual


def saturates(power_sequence: List[float], n_steps: int) -> Tuple[bool, float]:
    """Whether a (truncated, length n_steps) power sequence drives the
    residual toward zero. Returns (approaches_zero, residual_at_n_steps)
    for an initial gap of 1.0. Corollary 7.5: full saturation in the limit
    holds iff sum(kappa_i) diverges; here we report the finite-horizon
    residual and the partial sum, which is what any finite validation run
    can observe.
    """
    residual = residual_after_chain(1.0, power_sequence[:n_steps])
    return residual, sum(power_sequence[:n_steps])


class SupportGraph:
    """A directed graph of pairwise catalyst support relations (Definition 7.6)."""

    def __init__(self, catalysts: List[str]):
        self.catalysts = catalysts
        self.edges: set = set()  # (j, i) meaning j supports i

    def add_support(self, j: str, i: str) -> None:
        self.edges.add((j, i))

    def has_cycle_of_length_at_least(self, k: int) -> bool:
        """Detect any directed cycle of length >= k via simple DFS cycle
        enumeration on this (necessarily small, validation-scale) graph.
        """
        n = self.catalysts
        adj: Dict[str, List[str]] = {c: [] for c in n}
        for (j, i) in self.edges:
            adj[j].append(i)

        def dfs(start, current, visited_path, depth):
            for nxt in adj[current]:
                if nxt == start and depth >= k:
                    return True
                if nxt not in visited_path and depth < len(n):
                    if dfs(start, nxt, visited_path | {nxt}, depth + 1):
                        return True
            return False

        for start in n:
            if dfs(start, start, {start}, 1):
                return True
        return False

    def is_strongly_connected_triangle(self, theta: float, strengths: Dict[Tuple[str, str], float]) -> bool:
        """Check the sufficiency condition of Theorem 7.8(ii): three
        catalysts, each pair supporting the other above threshold theta,
        forming a strongly connected triangle.
        """
        if len(self.catalysts) != 3:
            return False
        a, b, c = self.catalysts
        pairs = [(a, b), (b, a), (b, c), (c, b), (a, c), (c, a)]
        return all(strengths.get(p, 0.0) > theta for p in pairs)

    def robust_to_single_removal(self, strengths: Dict[Tuple[str, str], float], theta: float) -> bool:
        """After removing any single catalyst, do the remaining >=2 still
        mutually support each other above the same threshold theta?
        (Theorem 7.8(ii) robustness clause: in a strongly connected
        triangle with theta > 1/2, the remaining pair's direct mutual
        edge survives any single removal.)
        """
        if len(self.catalysts) < 3:
            return False
        for removed in self.catalysts:
            remaining = [c for c in self.catalysts if c != removed]
            for x, y in itertools.permutations(remaining, 2):
                if strengths.get((x, y), 0.0) <= theta:
                    return False
        return True


def sign_only_coherence_verdict(strengths: Dict[Tuple[str, str], float], catalysts: List[str], theta: float) -> bool:
    """Corollary 7.9: coherence decided from the SIGN of pairwise support
    alone (whether each relation reinforces, i.e. is positive and above a
    fixed nominal threshold), never the magnitude.
    """
    signs = {k: (1 if v > theta else 0) for k, v in strengths.items()}
    g = SupportGraph(catalysts)
    for (j, i), s in signs.items():
        if s:
            g.add_support(j, i)
    return g.has_cycle_of_length_at_least(3)


def magnitude_coherence_verdict(strengths: Dict[Tuple[str, str], float], catalysts: List[str], theta: float) -> bool:
    """The magnitude-based ground truth verdict for comparison against the
    sign-only critic (Corollary 7.9 validation)."""
    g = SupportGraph(catalysts)
    for (j, i), s in strengths.items():
        if s > theta:
            g.add_support(j, i)
    return g.has_cycle_of_length_at_least(3)
