"""Experiment 02: Individuation by Negation (Theorem 4.2).

Complementation is an exact involution on subsets of a finite medium:
co(co(U)) == U, checked over 200 random subsets.
"""

from __future__ import annotations

import random

from scp.individuation import complement, double_complement_is_identity
from scp.report import make_report

SEED = 42
N_TRIALS = 200
MEDIUM_SIZE = 8


def run() -> dict:
    rng = random.Random(SEED)
    whole = frozenset(f"c{i}" for i in range(MEDIUM_SIZE))
    n_passed = 0
    partition_checks = 0
    partition_passed = 0

    for _ in range(N_TRIALS):
        k = rng.randint(1, MEDIUM_SIZE - 1)
        subset = frozenset(rng.sample(sorted(whole), k))
        if double_complement_is_identity(whole, subset):
            n_passed += 1

        # Also check U and co(U) partition the whole exactly (|U|+|coU|=|V|).
        co = complement(whole, subset)
        partition_checks += 1
        if len(subset) + len(co) == len(whole) and (subset | co) == whole and not (subset & co):
            partition_passed += 1

    return make_report(
        category_id="02",
        theorem="Theorem 4.2 (Individuation by Negation)",
        check_type="Identity",
        seed=SEED,
        n_checks=N_TRIALS + partition_checks,
        n_passed=n_passed + partition_passed,
        max_error=0.0 if n_passed == N_TRIALS else 1.0,
        details={
            "medium_size": MEDIUM_SIZE,
            "n_involution_trials": N_TRIALS,
            "involution_passed": n_passed,
            "n_partition_trials": partition_checks,
            "partition_passed": partition_passed,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
