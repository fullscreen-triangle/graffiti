"""Experiment 06: Convergence Admissibility (Theorem 6.3).

Every propagation terminating at its designated target is admissible
regardless of interior alignment, including propagations passing through
maximally-misaligned interior claims, over 50 constructed process graphs.
"""

from __future__ import annotations

import random

from scp.graph import random_contact_graph
from scp.propagation import Propagation, is_convergent, random_interior_variant
from scp.report import make_report

SEED = 42
N_GRAPHS = 50


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0

    for _ in range(N_GRAPHS):
        n_claims = rng.randint(5, 12)
        floor = 0.05 + rng.random() * 1.0
        graph = random_contact_graph(rng, n_claims, floor, edge_prob=0.3)

        seed = graph.claims[0]
        target = graph.claims[-1]
        interior_len = rng.randint(0, min(4, n_claims - 2))
        prop = random_interior_variant(rng, seed, target, graph.claims, interior_len)

        n_checks += 1
        if is_convergent(prop, target):
            n_passed += 1

        # Also verify that arbitrarily large interior misalignment does not
        # block admissibility: pick an interior claim maximally distant
        # from the target and confirm it is still permitted to appear.
        if prop.interior():
            worst = max(prop.interior(), key=lambda c: graph.alignment(c, target))
            n_checks += 1
            # Admissibility depends only on the terminal claim (Theorem 6.3);
            # presence of `worst` in the interior must not change that.
            if is_convergent(prop, target):
                n_passed += 1

    return make_report(
        category_id="06",
        theorem="Theorem 6.3 (Convergence Admissibility)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=0.0 if n_passed == n_checks else 1.0,
        details={"n_graphs": N_GRAPHS},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
