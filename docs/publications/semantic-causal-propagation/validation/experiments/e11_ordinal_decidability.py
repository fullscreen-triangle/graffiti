"""Experiment 11: Ordinal Decidability of Coherence (Corollary 7.9).

A sign-only critic (reading only the sign of each pairwise support
relation) reproduces the magnitude-based coherence verdict across 2000
randomly generated chains.
"""

from __future__ import annotations

import itertools
import random

from scp.catalysis import magnitude_coherence_verdict, sign_only_coherence_verdict
from scp.report import make_report

SEED = 42
N_CHAINS = 2000
THETA = 0.5


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    n_incoherent_flagged = 0
    n_incoherent_total = 0

    for _ in range(N_CHAINS):
        n_catalysts = rng.randint(2, 6)
        catalysts = [f"g{i}" for i in range(n_catalysts)]
        strengths = {
            (a, b): rng.uniform(0.0, 1.0)
            for a, b in itertools.permutations(catalysts, 2)
            if rng.random() < 0.7
        }

        sign_verdict = sign_only_coherence_verdict(strengths, catalysts, THETA)
        magnitude_verdict = magnitude_coherence_verdict(strengths, catalysts, THETA)

        n_checks += 1
        if sign_verdict == magnitude_verdict:
            n_passed += 1

        if not magnitude_verdict:
            n_incoherent_total += 1
            if not sign_verdict:
                n_incoherent_flagged += 1

    return make_report(
        category_id="11",
        theorem="Corollary 7.9 (Ordinal Decidability of Coherence)",
        check_type="Structural",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={
            "n_chains": N_CHAINS,
            "theta": THETA,
            "agreement_rate": n_passed / n_checks,
            "n_incoherent_total": n_incoherent_total,
            "n_incoherent_correctly_flagged": n_incoherent_flagged,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
