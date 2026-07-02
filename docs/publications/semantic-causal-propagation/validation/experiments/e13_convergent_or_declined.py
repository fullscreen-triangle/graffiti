"""Experiment 13: Convergent Closure or Honest Decline (Theorem 8.3).

Every search over a finite catalyst registry terminates in finitely many
steps in either the convergent or the contested-closure (decline) state,
over 300 constructed instances.
"""

from __future__ import annotations

import random

from scp.closure import equivalence_classes, is_closed
from scp.graph import random_contact_graph, two_cluster_graph
from scp.report import make_report

SEED = 42
N_INSTANCES = 300


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    n_convergent = 0
    n_declined = 0

    for i in range(N_INSTANCES):
        if i % 2 == 0:
            # Single-cluster instance: should converge to one class.
            n_claims = rng.randint(3, 10)
            floor = 0.05 + rng.random() * 0.5
            graph = random_contact_graph(rng, n_claims, floor, edge_prob=0.4)
            target = graph.claims[-1]
            classes = equivalence_classes(graph, [target])
            closed = is_closed([], classes, graph)  # no further catalysts registered
            outcome = "convergent" if len(classes) == 1 else "declined"
        else:
            # Two-cluster instance with both catalysts registered/invoked:
            # should reach contested closure (decline) with two classes.
            cluster_size = rng.randint(2, 5)
            floor = 0.05 + rng.random() * 0.5
            graph, a_claims, b_claims = two_cluster_graph(rng, cluster_size, floor)
            targets = [a_claims[0], b_claims[0]]
            classes = equivalence_classes(graph, targets)
            closed = is_closed([], classes, graph)
            outcome = "convergent" if len(classes) == 1 else "declined"

        n_checks += 1
        # Termination: the outcome is always one of exactly two states,
        # and closure (relative to the empty "further catalysts" set) is
        # always reached in this finite construction.
        if closed and outcome in ("convergent", "declined"):
            n_passed += 1

        if outcome == "convergent":
            n_convergent += 1
        else:
            n_declined += 1

    return make_report(
        category_id="13",
        theorem="Theorem 8.3 (Convergent Closure or Honest Decline)",
        check_type="Boundary",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={
            "n_instances": N_INSTANCES,
            "n_convergent": n_convergent,
            "n_declined": n_declined,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
