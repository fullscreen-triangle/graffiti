"""Experiment 14: Scheduler Soundness (Theorem 10.6).

No stalled seek is ever assigned positive dispatch priority; every closed
seek is assigned priority +infinity and dispatched strictly before any
finite-priority seek, over 2003 randomly generated live-seek priority
scenarios.
"""

from __future__ import annotations

import math
import random

from scp.report import make_report
from scp.scheduler import LiveSeek, priority, select_next

SEED = 42
N_TRIALS = 2003


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0

    for _ in range(N_TRIALS):
        n_seeks = rng.randint(1, 5)
        seeks = []
        for i in range(n_seeks):
            kind = rng.choice(["stalled", "descending", "closed"])
            floor = 0.01
            if kind == "stalled":
                history = [1.0, 0.5, 0.5, 0.5]  # descent has stopped
                seek = LiveSeek(name=f"s{i}", residue_history=history, floor=floor, closed=False)
            elif kind == "descending":
                start = rng.uniform(0.5, 5.0)
                history = [start, start * 0.8, start * 0.6]  # still falling
                seek = LiveSeek(name=f"s{i}", residue_history=history, floor=floor, closed=False)
            else:
                seek = LiveSeek(name=f"s{i}", residue_history=[floor], floor=floor, closed=True)
            seeks.append(seek)

        # Check (i): no stalled seek gets positive priority.
        for s in seeks:
            if s.descent_rate() <= 0 and not s.closed:
                n_checks += 1
                if priority(s) == 0.0:
                    n_passed += 1

        # Check (iii): closed seeks get +inf priority.
        for s in seeks:
            if s.closed:
                n_checks += 1
                if priority(s) == math.inf:
                    n_passed += 1

        # Check: selection prefers a closed seek over any finite-priority
        # seek, and prefers a descending seek over a stalled one.
        selected = select_next(seeks)
        n_checks += 1
        closed_seeks = [s for s in seeks if s.closed]
        if closed_seeks:
            if selected is not None and selected.closed:
                n_passed += 1
        else:
            descending = [s for s in seeks if s.descent_rate() > 0]
            if descending:
                if selected is not None and selected.descent_rate() > 0:
                    n_passed += 1
            else:
                if selected is None:
                    n_passed += 1

    return make_report(
        category_id="14",
        theorem="Theorem 10.6 (Scheduler Soundness)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=0.0 if n_passed == n_checks else 1.0,
        details={"n_trials": N_TRIALS},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
