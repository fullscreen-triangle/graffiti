"""Experiment 08: Multiplicative Composition Law (Theorem 7.3).

Measured composite power matches 1 - prod(1 - kappa_i) to machine
precision across 500 randomly composed catalyst chains.
"""

from __future__ import annotations

import random

from scp.catalysis import composite_power, residual_after_chain
from scp.report import make_report

SEED = 42
N_CHAINS = 500


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    max_error = 0.0

    for _ in range(N_CHAINS):
        chain_len = rng.randint(1, 8)
        powers = [rng.uniform(0.0, 0.99) for _ in range(chain_len)]

        predicted = composite_power(powers)

        # "Measured": simulate applying each catalyst in sequence to an
        # initial above-floor gap and read off the fraction closed.
        initial_gap = 1.0
        residual = residual_after_chain(initial_gap, powers)
        measured = 1.0 - residual / initial_gap

        n_checks += 1
        error = abs(predicted - measured)
        max_error = max(max_error, error)
        if error < 1e-12:
            n_passed += 1

    return make_report(
        category_id="08",
        theorem="Theorem 7.3 (Multiplicative Composition)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=max_error,
        details={"n_chains": N_CHAINS},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
