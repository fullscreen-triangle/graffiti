"""Scheduler priority and soundness.

Implements Definition 10.4 (Live Seek; Residue Descent), Definition 10.5
(Scheduler Priority), and Theorem 10.6 (Scheduler Soundness) of
``semantic-causal-propagation.tex``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class LiveSeek:
    name: str
    residue_history: List[float] = field(default_factory=list)  # sigma(x_s, x*_s) over ticks
    floor: float = 1e-3
    closed: bool = False

    def descent_rate(self) -> float:
        if len(self.residue_history) < 2:
            return 0.0
        return self.residue_history[-2] - self.residue_history[-1]

    def current_residue(self) -> float:
        return self.residue_history[-1] if self.residue_history else self.floor


def priority(seek: LiveSeek) -> float:
    """Definition 10.5."""
    if seek.closed:
        return math.inf
    delta = seek.descent_rate()
    denom = max(seek.current_residue() - seek.floor, seek.floor)
    if delta <= 0:
        return 0.0
    return delta / denom


def select_next(live_seeks: List[LiveSeek]) -> LiveSeek | None:
    """argmax_s P(s); returns None if all priorities are 0 (all stalled)."""
    if not live_seeks:
        return None
    best = max(live_seeks, key=priority)
    if priority(best) <= 0:
        return None
    return best
