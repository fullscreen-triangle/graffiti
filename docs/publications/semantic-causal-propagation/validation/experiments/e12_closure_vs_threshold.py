"""Experiment 12: Closure is Strictly Stronger than a Confidence Threshold
(Theorem 8.2).

A fixed confidence threshold is satisfied by the first cluster's
propagation alone in every instance, while a second, not-yet-invoked
catalyst is confirmed available to reach a materially distinct
equivalence class, over 300 two-cluster process graphs.
"""

from __future__ import annotations

import random

from scp.closure import confidence_threshold_met, equivalence_classes, is_closed
from scp.graph import two_cluster_graph
from scp.report import make_report

SEED = 42
N_GRAPHS = 300
THETA = 0.5


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0

    for _ in range(N_GRAPHS):
        cluster_size = rng.randint(2, 6)
        floor = 0.05 + rng.random() * 0.5
        graph, a_claims, b_claims = two_cluster_graph(rng, cluster_size, floor)

        target_a = a_claims[0]
        target_b = b_claims[0]

        # Confidence threshold is met by cluster A's propagation alone.
        threshold_met = confidence_threshold_met(graph, target_a, THETA)

        # But the search is not closed: cluster B is a genuinely distinct,
        # not-yet-invoked, reachable equivalence class.
        classes_so_far = equivalence_classes(graph, [target_a])
        closed = is_closed([target_b], classes_so_far, graph)

        n_checks += 1
        if threshold_met and not closed:
            n_passed += 1

    return make_report(
        category_id="12",
        theorem="Theorem 8.2 (Closure is Strictly Stronger than a Confidence Threshold)",
        check_type="Structural",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={
            "n_graphs": N_GRAPHS,
            "theta": THETA,
            "note": (
                "confidence threshold satisfied by a single completed "
                "propagation while a second, distinct-cluster catalyst "
                "remains available and unclosed"
            ),
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
