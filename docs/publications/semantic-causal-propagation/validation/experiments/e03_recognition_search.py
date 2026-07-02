"""Experiment 03: Recognition/Search Identity (Theorem 4.5).

Dec(q) = v iff q in Proj(v), checked over 150 synthetic (decoder, query)
pairs, and Dec is exactly recovered from the fibre partition {Proj(v)}.
"""

from __future__ import annotations

import random

from scp.individuation import build_synthetic_decoder
from scp.report import make_report

SEED = 42
N_CLAIMS = 15
N_QUERIES_PER_CLAIM = 10


def run() -> dict:
    rng = random.Random(SEED)
    decoder = build_synthetic_decoder(rng, N_CLAIMS, N_QUERIES_PER_CLAIM)

    n_checks = 0
    n_passed = 0

    queries = list(decoder.mapping.keys())
    rng.shuffle(queries)
    sample = queries[:150]

    for q in sample:
        v = decoder.decode(q)
        n_checks += 1
        if q in decoder.fibre(v):
            n_passed += 1

    # Recover Dec from the fibre partition and check it matches exactly.
    rebuilt = decoder.recovers_decoder()
    n_checks += 1
    if rebuilt == decoder.mapping:
        n_passed += 1

    return make_report(
        category_id="03",
        theorem="Theorem 4.5 (Recognition/Search Identity)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=0.0 if n_passed == n_checks else 1.0,
        details={
            "n_claims": N_CLAIMS,
            "n_queries_per_claim": N_QUERIES_PER_CLAIM,
            "n_query_pairs_checked": len(sample),
            "decoder_exactly_recovered_from_fibres": rebuilt == decoder.mapping,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
