"""Experiment 01: Resolution Floor (Theorem 3.2).

sigma(v) >= beta on every claim of 300 random contact graphs; no cut of
weight 0 occurs.
"""

from __future__ import annotations

import random

from scp.graph import random_contact_graph
from scp.report import make_report

SEED = 42
N_GRAPHS = 300


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    max_violation = 0.0
    sample_floors = []

    for trial in range(N_GRAPHS):
        n_claims = rng.randint(3, 12)
        floor = 0.05 + rng.random() * 2.0
        graph = random_contact_graph(rng, n_claims, floor, edge_prob=0.3)
        for claim in graph.claims:
            n_checks += 1
            sigma = graph.separation_cost(claim)
            if sigma >= floor - 1e-9:
                n_passed += 1
            else:
                max_violation = max(max_violation, floor - sigma)
        sample_floors.append(floor)

    return make_report(
        category_id="01",
        theorem="Theorem 3.2 (Resolution Floor)",
        check_type="Bound",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=max_violation,
        details={
            "n_graphs": N_GRAPHS,
            "grid": "300 random graphs, 3-12 claims each",
            "min_floor_tested": min(sample_floors),
            "max_floor_tested": max(sample_floors),
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
