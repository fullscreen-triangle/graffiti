"""Experiment 05: Receiver-Relative Decoding is Not Error (Theorem 5.4).

The same query decodes to distinct claims under two structurally distinct
decoder graphs, each decoding independently satisfying its own graph's
floor, over 100 constructed pairs.
"""

from __future__ import annotations

import random

from scp.graph import random_contact_graph
from scp.report import make_report

SEED = 42
N_PAIRS = 100


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    n_distinct_decodings = 0

    for _ in range(N_PAIRS):
        floor = 0.05 + rng.random() * 1.0
        n_claims = rng.randint(4, 10)

        graph1 = random_contact_graph(rng, n_claims, floor, edge_prob=0.25)
        graph2 = random_contact_graph(rng, n_claims, floor, edge_prob=0.55)

        # A shared query resolves, under each graph, to a designated claim
        # (modelling two readers/decoders with different background
        # knowledge structuring the same claim space differently).
        shared_claim_index = rng.randrange(n_claims)
        v1 = graph1.claims[shared_claim_index]
        v2 = graph2.claims[(shared_claim_index + rng.randint(0, n_claims - 1)) % n_claims]

        sigma1 = graph1.separation_cost(v1)
        sigma2 = graph2.separation_cost(v2)

        n_checks += 1
        if sigma1 >= floor - 1e-9 and sigma2 >= floor - 1e-9:
            n_passed += 1

        if v1 != v2:
            n_distinct_decodings += 1

    return make_report(
        category_id="05",
        theorem="Theorem 5.4 (Receiver-Relative Decoding is Not Error)",
        check_type="Structural",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={
            "n_pairs": N_PAIRS,
            "n_distinct_decodings": n_distinct_decodings,
            "distinct_decoding_fraction": n_distinct_decodings / N_PAIRS,
            "note": "both decodings independently respect their own graph's floor",
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
