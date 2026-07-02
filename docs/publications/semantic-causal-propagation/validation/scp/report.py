"""Shared JSON reporting helpers for validation experiments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default, sort_keys=False)


def make_report(
    category_id: str,
    theorem: str,
    check_type: str,
    seed: int,
    n_checks: int,
    n_passed: int,
    max_error: float | None,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "category_id": category_id,
        "theorem": theorem,
        "check_type": check_type,
        "seed": seed,
        "n_checks": n_checks,
        "n_passed": n_passed,
        "passed": n_passed == n_checks,
        "max_error": max_error,
        "details": details,
    }
