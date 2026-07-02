"""Closure: when a search is finished.

Implements Definition 8.1 (Closure), Theorem 8.2 (Closure is Strictly
Stronger than a Confidence Threshold), and Theorem 8.3 (Convergent Closure
or Honest Decline) of ``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple

from .graph import ContactGraph


def _cut_side_key(graph: ContactGraph, target: str) -> FrozenSet[str]:
    """A genuine endpoint invariant of Theorem 6.4: the minimum-cut side
    S*(target) itself (its exact claim membership), not merely its
    weight. Two targets are endpoint-indistinguishable only if they share
    the same minimum-cut side -- distinct clusters with numerically equal
    bridge weight but disjoint membership are correctly kept apart.
    """
    return frozenset(graph.min_cut_side(target))


def equivalence_classes(graph: ContactGraph, reached_targets: List[str]) -> List[FrozenSet[str]]:
    """Partition reached targets into endpoint-indistinguishable classes
    (Corollary 6.5 "the admissible set is a class, not a path"): two
    targets are in the same class iff their minimum cut against the medium
    -- weight AND side membership, the full endpoint invariant of
    Theorem 6.4 -- agrees.
    """
    buckets: Dict[FrozenSet[str], List[str]] = {}
    for t in reached_targets:
        key = _cut_side_key(graph, t)
        buckets.setdefault(key, []).append(t)
    return [frozenset(v) for v in buckets.values()]


def confidence_threshold_met(graph: ContactGraph, terminal: str, theta: float) -> bool:
    """A naive confidence check: terminal alignment to itself (always the
    floor, by construction of a completed propagation) compared to theta.
    Demonstrates Theorem 8.2: this is satisfied trivially by any completed
    propagation, independent of whether other catalysts would diverge.
    """
    self_alignment = graph.alignment(terminal, terminal)
    # Normalise to a "confidence" in [0,1]: 1 - alignment/omega, higher is
    # more confident; a completed propagation's terminal is at the floor,
    # i.e. maximal confidence under this (deliberately naive) metric.
    omega = graph.total_weight()
    confidence = 1.0 - (self_alignment / omega if omega > 0 else 0.0)
    return confidence >= theta


def is_closed(available_catalyst_targets: List[str], classes_so_far: List[FrozenSet[str]], graph: ContactGraph) -> bool:
    """Definition 8.1: closed iff no available-but-uninvoked catalyst adds
    a new equivalence class."""
    existing_keys = set()
    for cls in classes_so_far:
        # Each class was built from targets sharing one cut-side key;
        # recompute that key from any representative member.
        representative = next(iter(cls))
        existing_keys.add(_cut_side_key(graph, representative))

    for target in available_catalyst_targets:
        key = _cut_side_key(graph, target)
        if key not in existing_keys:
            return False
    return True


@dataclass
class SearchOutcome:
    state: str  # "convergent" or "declined"
    classes: List[FrozenSet[str]]
