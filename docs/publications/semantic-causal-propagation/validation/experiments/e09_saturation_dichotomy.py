"""Experiment 09: Saturation Dichotomy (Corollary 7.5).

Divergent-sum power sequences (constant, harmonic) drive the residual
toward zero without a floor at the tested horizon; convergent-sum
sequences (geometric, inverse-square) plateau at a strictly positive
residual.
"""

from __future__ import annotations

import math

from scp.report import make_report

SEED = 42
N_STEPS = 2000
# Horizons at which the residual is inspected. The dichotomy is a
# statement about the *rate* of decay, not the raw magnitude at one fixed
# n: a divergent-sum sequence's log-residual diverges to -infinity as n
# grows, while a convergent-sum sequence's log-residual converges to a
# finite limit. Comparing log-residual growth across horizons (rather
# than the raw residual at a single very large n, which underflows to
# exact 0.0 in float64 for both classes indiscriminately) is the correct,
# numerically well-posed test.
HORIZONS = [10, 100, 1000, 2000]


def log_residual(power_sequence, n_steps) -> float:
    """log(prod_{i<=n} (1 - kappa_i)) computed in log-space to avoid
    premature float64 underflow to exact 0.0. An absolute catalyst
    (kappa=1) drives the true residual to exactly 0, i.e. log-residual to
    -infinity in a single step; this is the correct value, not an error."""
    total = 0.0
    for k in power_sequence[:n_steps]:
        if k >= 1.0:
            return -math.inf
        total += math.log1p(-k)
    return total


def run() -> dict:
    sequences = {
        "constant_0.1_divergent": [0.1] * N_STEPS,
        "harmonic_divergent": [1.0 / (i + 2) for i in range(N_STEPS)],
        "geometric_2^-i_convergent": [2.0 ** (-(i + 1)) for i in range(N_STEPS)],
        "inverse_square_convergent": [1.0 / (i + 2) ** 2 for i in range(N_STEPS)],
    }

    n_checks = 0
    n_passed = 0
    results = {}

    for name, seq in sequences.items():
        is_divergent_construction = "divergent" in name
        log_residuals_by_horizon = {h: log_residual(seq, h) for h in HORIZONS}
        partial_sum = sum(seq[: HORIZONS[-1]])

        results[name] = {
            "log_residual_by_horizon": log_residuals_by_horizon,
            "residual_at_smallest_horizon": math.exp(log_residuals_by_horizon[HORIZONS[0]]),
            "partial_sum_of_powers": partial_sum,
        }

        n_checks += 1
        values = [log_residuals_by_horizon[h] for h in HORIZONS]
        strictly_decreasing = all(values[i + 1] < values[i] - 1e-9 for i in range(len(values) - 1))
        last_two_increment = abs(log_residuals_by_horizon[HORIZONS[-1]] - log_residuals_by_horizon[HORIZONS[-2]])
        if is_divergent_construction:
            # log-residual must be monotonically decreasing without bound
            # (sum(kappa_i) = infinity): each doubling-plus horizon still
            # contributes a non-vanishing negative increment.
            if strictly_decreasing and last_two_increment > 1e-2:
                n_passed += 1
        else:
            # log-residual must converge to a finite limit: the increment
            # between the two largest horizons has already become small
            # relative to the total accumulated log-residual, since
            # sum(kappa_i) < infinity means the tail contributes a
            # vanishing (not necessarily machine-zero at finite horizon)
            # increment to the log-residual.
            relative_increment = last_two_increment / max(abs(values[-1]), 1e-12)
            if relative_increment < 1e-2:
                n_passed += 1

    return make_report(
        category_id="09",
        theorem="Corollary 7.5 (Saturation Dichotomy)",
        check_type="Boundary",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={"n_steps": N_STEPS, "horizons": HORIZONS, "sequences": results},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
