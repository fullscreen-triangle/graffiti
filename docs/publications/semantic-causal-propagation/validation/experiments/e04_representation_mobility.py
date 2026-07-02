"""Experiment 04: Representation Mobility (Theorem 5.2).

Every sampled representation tuple satisfies the averaging constraint to
floating-point precision, and representation switching leaves the
committed-record counter unchanged, over 200 representation fibres.
"""

from __future__ import annotations

import random

from scp.representation import committed_record_after_switch, representation_mean, sample_representation
from scp.report import make_report

SEED = 42
N_FIBRES = 200


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0
    max_error = 0.0
    off_shell_count = 0
    total_components = 0

    for _ in range(N_FIBRES):
        alignment = rng.uniform(0.01, 0.99)
        dimension = rng.randint(2, 8)
        components = sample_representation(rng, alignment, dimension)

        n_checks += 1
        mean = representation_mean(components)
        error = abs(mean - alignment)
        max_error = max(max_error, error)
        if error < 1e-9:
            n_passed += 1

        for c in components:
            total_components += 1
            if not (0.0 < c <= 1.0):
                off_shell_count += 1

        # Representation switch: record before/after must be unchanged.
        record_before = rng.randint(0, 1000)
        record_after = committed_record_after_switch(record_before)
        n_checks += 1
        if record_after == record_before:
            n_passed += 1

    return make_report(
        category_id="04",
        theorem="Theorem 5.2 (Representation Mobility)",
        check_type="Identity",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=max_error,
        details={
            "n_fibres": N_FIBRES,
            "total_components_sampled": total_components,
            "off_shell_component_count": off_shell_count,
            "off_shell_fraction": off_shell_count / total_components if total_components else 0.0,
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
