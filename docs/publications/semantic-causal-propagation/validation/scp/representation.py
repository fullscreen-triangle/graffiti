"""Representation mobility and receiver-relative decoding.

Implements Definition 5.1, Theorem 5.2 (Representation Mobility), and
Theorem 5.4 (Receiver-Relative Decoding is Not Error) of
``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import random
from typing import List, Tuple


def sample_representation(rng: random.Random, alignment: float, dimension: int) -> Tuple[float, ...]:
    """Sample a representation tuple (s_1,...,s_N) satisfying the averaging
    constraint (1/N) * sum(s_j) = alignment, with components otherwise free
    in R (Definition 5.1). Free components are drawn from a wide range so
    that some fall outside (0, 1] ("off-shell"), exactly as the theory
    permits.
    """
    components = [rng.uniform(-5.0, 5.0) for _ in range(dimension - 1)]
    s_n = dimension * alignment - sum(components)
    components.append(s_n)
    rng.shuffle(components)
    return tuple(components)


def representation_mean(components: Tuple[float, ...]) -> float:
    return sum(components) / len(components)


def committed_record_after_switch(record_before: int) -> int:
    """A representation switch commits no new contact edge (Theorem 5.2(ii));
    the committed record is unchanged."""
    return record_before
