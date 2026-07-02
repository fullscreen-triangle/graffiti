#!/usr/bin/env python3
"""Run the complete Semantic Causal Propagation validation suite.

Executes all fourteen experiment categories of
``semantic-causal-propagation.tex`` (Table 12.1), writes one JSON record
per category to ``data/scp_e<NN>_<name>.json``, and writes an aggregate
summary to ``data/scp_master_results.json``.

Usage:
    python run_all.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scp.report import write_json

EXPERIMENTS = [
    ("e01_floor", "01"),
    ("e02_individuation", "02"),
    ("e03_recognition_search", "03"),
    ("e04_representation_mobility", "04"),
    ("e05_receiver_relativity", "05"),
    ("e06_convergence_admissibility", "06"),
    ("e07_path_opacity", "07"),
    ("e08_multiplicative_law", "08"),
    ("e09_saturation_dichotomy", "09"),
    ("e10_coherence_triangle", "10"),
    ("e11_ordinal_decidability", "11"),
    ("e12_closure_vs_threshold", "12"),
    ("e13_convergent_or_declined", "13"),
    ("e14_scheduler_soundness", "14"),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main() -> int:
    start = time.time()
    results = []
    all_passed = True

    for module_name, category_id in EXPERIMENTS:
        module = __import__(f"experiments.{module_name}", fromlist=["run"])
        report = module.run()
        out_path = os.path.join(DATA_DIR, f"scp_{module_name}.json")
        write_json(out_path, report)

        status = "PASS" if report["passed"] else "FAIL"
        if not report["passed"]:
            all_passed = False
        print(
            f"[{category_id}] {report['theorem']:<70s} "
            f"{report['n_passed']:>5d}/{report['n_checks']:<5d} {status}"
        )
        results.append(
            {
                "category_id": category_id,
                "module": module_name,
                "theorem": report["theorem"],
                "check_type": report["check_type"],
                "n_checks": report["n_checks"],
                "n_passed": report["n_passed"],
                "passed": report["passed"],
                "max_error": report["max_error"],
            }
        )

    elapsed = time.time() - start
    total_checks = sum(r["n_checks"] for r in results)
    total_passed = sum(r["n_passed"] for r in results)

    summary = {
        "suite": "semantic-causal-propagation",
        "n_categories": len(results),
        "n_categories_passed": sum(1 for r in results if r["passed"]),
        "total_checks": total_checks,
        "total_passed": total_passed,
        "all_passed": all_passed,
        "elapsed_seconds": elapsed,
        "categories": results,
    }

    write_json(os.path.join(DATA_DIR, "scp_master_results.json"), summary)

    print()
    print(
        f"Pass rate: {summary['n_categories_passed']}/{summary['n_categories']} categories, "
        f"{total_passed}/{total_checks} individual checks, "
        f"in {elapsed:.2f}s"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
