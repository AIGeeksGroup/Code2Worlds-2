"""Method-level SimCode4D metrics, separate from frozen reference checks.

Every registered first output and episode stays in its denominator. Failed
builds, missing overlays, and execution errors contribute zero to RTS.
"""
from __future__ import annotations

from typing import Any, Iterable


def micro_f1_counts(predicted: Iterable[Any], expected: Iterable[Any]) -> dict[str, int | float]:
    predicted_set, expected_set = set(predicted), set(expected)
    tp = len(predicted_set & expected_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / denominator if denominator else 1.0}


def pooled_micro_f1(rows: Iterable[dict[str, Any]]) -> dict[str, int | float | None]:
    rows = list(rows)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    denominator = 2 * tp + fp + fn
    return {"eligible_packages": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / denominator if denominator else None}


def robust_task_success(build: bool, overlay: bool, validation: dict[str, Any] | None) -> bool:
    """Seven-factor Eq. 4 gate: B, D, M, K, T, S, and I."""
    if not build or not overlay or validation is None:
        return False
    return all((
        validation.get("mechanics_consistency_pass", False),
        validation.get("kinematics_consistency_pass", False),
        validation.get("task_predicates_pass", False),
        validation.get("safety_pass", False),
        validation.get("integrity_pass", False),
    ))


def summarize_method(packages: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """WorldBuildPass@1, TaskOverlayPass@1, micro-F1, and episode RTS."""
    worlds = len(packages)
    trials = len(episodes)
    build_count = sum(bool(row["build_pass"]) for row in packages)
    overlay_count = sum(bool(row["overlay_pass"]) for row in packages)
    return {
        "first_output_world_count": worlds,
        "world_build_pass_at_1": build_count / worlds if worlds else None,
        "task_overlay_pass_at_1": overlay_count / worlds if worlds else None,
        "required_entity_binding_micro_f1": pooled_micro_f1(
            row["entity_binding"] for row in packages if row.get("entity_binding") is not None
        ),
        "source_name_binding_micro_f1": None,
        "initial_support_relation_micro_f1": pooled_micro_f1(
            row["support_relation"] for row in packages if row.get("support_relation") is not None
        ),
        "registered_episode_count": trials,
        "completed_episode_count": sum(row.get("validation") is not None for row in episodes),
        "robust_task_success": sum(bool(row["robust_task_success"]) for row in episodes) / trials if trials else None,
        "task_pass_rate": sum(bool((row.get("validation") or {}).get("task_predicates_pass")) for row in episodes) / trials if trials else None,
        "physical_pass_rate": sum(bool((row.get("validation") or {}).get("physical_admissibility_pass")) for row in episodes) / trials if trials else None,
        "integrity_pass_rate": sum(bool((row.get("validation") or {}).get("integrity_pass")) for row in episodes) / trials if trials else None,
    }
