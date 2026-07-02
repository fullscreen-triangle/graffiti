"""Experiment 10: Coherence Requires a Triangle (Theorem 7.8).

Necessity: every acyclic support graph fails the single-removal
robustness check (linear support cannot ground a claim, Theorem 7.6).
Sufficiency: every constructed strongly-connected 3-catalyst support
triangle at threshold theta > 1/2 survives removal of any single member;
2-cycles (mutual support between exactly two catalysts, no independent
third check) do not.

800 randomly generated instances total, split across the necessity check
(acyclic graphs) and the sufficiency/robustness check (constructed
triangles and 2-cycles).
"""

from __future__ import annotations

import itertools
import random

from scp.catalysis import SupportGraph
from scp.report import make_report

SEED = 42
N_ACYCLIC_GRAPHS = 400
N_TRIANGLES = 200
N_TWO_CYCLES = 200
THETA = 0.5


def random_acyclic_support_graph(rng: random.Random, n_catalysts: int):
    """Build a support graph on a random topological order, adding only
    forward edges (i -> j for i before j in the order), which is
    necessarily acyclic."""
    catalysts = [f"g{i}" for i in range(n_catalysts)]
    order = catalysts[:]
    rng.shuffle(order)
    graph = SupportGraph(catalysts)
    strengths = {}
    for idx, j in enumerate(order):
        for i in order[idx + 1 :]:
            if rng.random() < 0.6:
                strength = rng.uniform(0.0, 1.0)
                strengths[(j, i)] = strength
                if strength > THETA:
                    graph.add_support(j, i)
    return graph, strengths


def construct_triangle(rng: random.Random, above_theta: bool):
    """A 3-catalyst support graph with every ordered pair's support
    strength either strictly above theta (a genuine sufficiency-triangle
    instance) or exactly at/below theta (violates the hypothesis, used as
    a negative control)."""
    catalysts = ["g0", "g1", "g2"]
    graph = SupportGraph(catalysts)
    strengths = {}
    for a, b in itertools.permutations(catalysts, 2):
        strength = rng.uniform(THETA + 1e-3, 1.0) if above_theta else rng.uniform(0.0, THETA - 1e-3)
        strengths[(a, b)] = strength
        if strength > THETA:
            graph.add_support(a, b)
    return graph, strengths


def construct_two_cycle(rng: random.Random):
    """Exactly two catalysts, mutually supporting each other above theta,
    with no independent third member: a 2-cycle, which Theorem 7.8(i)
    excludes from grounding a claim (fails the majority condition)."""
    catalysts = ["g0", "g1"]
    graph = SupportGraph(catalysts)
    strengths = {
        ("g0", "g1"): rng.uniform(THETA + 1e-3, 1.0),
        ("g1", "g0"): rng.uniform(THETA + 1e-3, 1.0),
    }
    graph.add_support("g0", "g1")
    graph.add_support("g1", "g0")
    return graph, strengths


def run() -> dict:
    rng = random.Random(SEED)
    n_checks = 0
    n_passed = 0

    # --- Necessity: acyclic support graphs never ground a claim robustly. ---
    acyclic_checked = 0
    acyclic_correctly_fails = 0
    for _ in range(N_ACYCLIC_GRAPHS):
        n_catalysts = rng.randint(2, 6)
        graph, strengths = random_acyclic_support_graph(rng, n_catalysts)
        assert not graph.has_cycle_of_length_at_least(3)

        n_checks += 1
        acyclic_checked += 1
        robust = graph.robust_to_single_removal(strengths, THETA)
        if not robust:
            n_passed += 1
            acyclic_correctly_fails += 1

    # --- Sufficiency: a genuine >theta strongly-connected triangle is robust. ---
    triangle_checked = 0
    triangle_correctly_robust = 0
    for _ in range(N_TRIANGLES):
        graph, strengths = construct_triangle(rng, above_theta=True)
        assert graph.has_cycle_of_length_at_least(3)
        assert graph.is_strongly_connected_triangle(THETA, strengths)

        n_checks += 1
        triangle_checked += 1
        if graph.robust_to_single_removal(strengths, THETA):
            n_passed += 1
            triangle_correctly_robust += 1

    # --- Negative control: 2-cycles do not ground a claim robustly. ---
    two_cycle_checked = 0
    two_cycle_correctly_fails = 0
    for _ in range(N_TWO_CYCLES):
        graph, strengths = construct_two_cycle(rng)

        n_checks += 1
        two_cycle_checked += 1
        # A 2-cycle has fewer than 3 catalysts, so robust_to_single_removal
        # returns False by construction (Theorem 7.8(i): a 2-cycle fails
        # the majority condition and cannot ground a claim).
        if not graph.robust_to_single_removal(strengths, THETA):
            n_passed += 1
            two_cycle_correctly_fails += 1

    return make_report(
        category_id="10",
        theorem="Theorem 7.8 (Coherence Requires Three Mutually Supporting Catalysts)",
        check_type="Structural",
        seed=SEED,
        n_checks=n_checks,
        n_passed=n_passed,
        max_error=None,
        details={
            "theta": THETA,
            "necessity": {
                "n_acyclic_graphs": acyclic_checked,
                "n_correctly_non_robust": acyclic_correctly_fails,
            },
            "sufficiency": {
                "n_triangles": triangle_checked,
                "n_correctly_robust": triangle_correctly_robust,
            },
            "two_cycle_negative_control": {
                "n_two_cycles": two_cycle_checked,
                "n_correctly_non_robust": two_cycle_correctly_fails,
            },
        },
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
