"""Experiment 07: Path Opacity (Theorem 6.4).

Endpoint invariants (target minimum cut, terminal self-alignment) are
measured identical across 40 interior-permutation variants of a
fixed-endpoint propagation.
"""

from __future__ import annotations

import random

from scp.graph import random_contact_graph
from scp.propagation import endpoint_invariants, random_interior_variant
from scp.report import make_report

SEED = 42
N_VARIANTS = 40


def run() -> dict:
    rng = random.Random(SEED)
    n_claims = 12
    floor = 0.2
    graph = random_contact_graph(rng, n_claims, floor, edge_prob=0.35)

    seed = graph.claims[0]
    target = graph.claims[-1]

    invariants = []
    for _ in range(N_VARIANTS):
        interior_len = rng.randint(0, min(5, n_claims - 2))
        prop = random_interior_variant(rng, seed, target, graph.claims, interior_len)
        invariants.append(endpoint_invariants(graph, prop))

    n_checks = 0
    n_passed = 0
    max_delta = 0.0
    reference = invariants[0]

    for inv in invariants[1:]:
        for key in ("seed", "target", "terminal_self_alignment", "target_min_cut"):
            n_checks += 1
            if key in ("seed", "target"):
                if inv[key] == reference[key]:
                    n_passed += 1
            else:
                delta = abs(inv[key] - reference[key])
                max_delta = max(max_delta, delta)
                if delta < 1e-9:
                    n_passed += 1

    return make_report(
        category_id="07",
        theorem="Theorem 6.4 (Path Opacity)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=max_delta,
        details={
            "n_interior_variants": N_VARIANTS,
            "n_claims": n_claims,
            "reference_invariants": reference,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
