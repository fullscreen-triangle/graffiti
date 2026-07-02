"""Convergence admissibility and path opacity.

Implements Definition 6.2 (Search Process Graph; Propagation),
Theorem 6.3 (Convergence Admissibility), and Theorem 6.4 (Path Opacity)
of ``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .graph import ContactGraph


@dataclass
class Propagation:
    """A walk (v0, e1, v1, ..., ek, vk) realised as a sequence of claims."""

    claims: List[str]

    @property
    def seed(self) -> str:
        return self.claims[0]

    @property
    def terminal(self) -> str:
        return self.claims[-1]

    def interior(self) -> List[str]:
        return self.claims[1:-1]


def is_convergent(prop: Propagation, target: str) -> bool:
    """Theorem 6.3: admissible iff terminal claim equals the target."""
    return prop.terminal == target


def endpoint_invariants(graph: ContactGraph, prop: Propagation) -> dict:
    """Compute the endpoint-only invariants of Theorem 6.4: seed, target,
    terminal alignment to itself, and the minimum-cut value of the target
    against the medium. None of these read the interior of the walk.
    """
    target = prop.terminal
    return {
        "seed": prop.seed,
        "target": target,
        "terminal_self_alignment": graph.alignment(target, target),
        "target_min_cut": graph.separation_cost(target),
    }


def random_interior_variant(
    rng: random.Random, seed: str, target: str, pool: List[str], length: int
) -> Propagation:
    """Construct a propagation from seed to target with a randomly chosen
    interior drawn from ``pool`` (claims other than seed/target), of the
    given interior length. Different calls with the same seed/target and
    different interiors instantiate the two propagations of Theorem 6.4.
    """
    candidates = [c for c in pool if c not in (seed, target)]
    rng.shuffle(candidates)
    interior = candidates[:length]
    return Propagation(claims=[seed, *interior, target])
