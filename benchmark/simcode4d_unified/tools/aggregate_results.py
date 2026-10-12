#!/usr/bin/env python3
"""Audit and aggregate a complete SimCode4D seed matrix into formal tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any




def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rate(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[field]) for row in rows) / len(rows)


def vector_norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def relative_episode_dir(run_root: Path, relative: str) -> Path:
    candidate = (run_root / relative).resolve()
    root = run_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise RuntimeError("episode path escapes run root: " + relative)
    return candidate


def canonical_validation_fingerprint(validation: dict[str, Any]) -> str:
    payload = dict(validation)
    payload.pop("generation_seed", None)
    payload.pop("raw_event_log_sha256", None)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def expected_artifact_hashes(
    workspace: Path,
    suite_root: Path,
    registry: dict[str, Any],
) -> dict[str, dict[str, str]]:
    runtime_hash = sha256_file(suite_root / "tools/runtime.py")
    physics_diagnostics_hash = sha256_file(suite_root / "tools/physics_diagnostics.py")
    expected: dict[str, dict[str, str]] = {}
    for item in registry["cases"]:
        case_id = item["case_id"]
        spec_path = workspace / item["case_spec"]["path"]
        gold_path = workspace / item["case_gold"]["path"]
        if sha256_file(spec_path) != item["case_spec"]["sha256"]:
            raise RuntimeError(case_id + ": current case spec does not match registry")
        if sha256_file(gold_path) != item["case_gold"]["sha256"]:
            raise RuntimeError(case_id + ": current case gold does not match registry")
        spec = load_json(spec_path)
        expected[case_id] = {
            "runtime_model_sha256": spec["artifacts"]["runtime_model"]["sha256"],
            "case_spec_sha256": item["case_spec"]["sha256"],
            "case_gold_sha256": item["case_gold"]["sha256"],
            "probe_sha256": spec["probe"]["sha256"],
            "validator_sha256": spec["validator"]["sha256"],
            "runtime_sha256": runtime_hash,
            "physics_diagnostics_sha256": physics_diagnostics_hash,
        }
    return expected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    order_rows = [row for row in rows if row["event_order_applicable"]]
    return {
        "episode_count": len(rows),
        "task_success_rate": rate(rows, "task_predicates_pass"),
        "safety_pass_rate": rate(rows, "safety_pass"),
        "integrity_pass_rate": rate(rows, "integrity_pass"),
        "robust_task_success_rate": rate(rows, "dev_reference_success"),
        "forbidden_contact_episode_rate": rate(rows, "forbidden_contact_episode"),
        "joint_limit_violation_episode_rate": rate(rows, "joint_limit_violation_episode"),
        "friction_cone_failure_episode_rate": rate(rows, "friction_cone_failure_episode"),
        "friction_contact_coverage_rate": (
            sum(row["friction_contact_sample_count"] > 0 for row in rows) / len(rows)
            if rows else None
        ),
        "mechanics_consistency_pass_rate": rate(rows, "mechanics_consistency_pass"),
        "kinematics_consistency_pass_rate": rate(rows, "kinematics_consistency_pass"),
        "diagnostics_finite_rate": rate(rows, "diagnostics_finite"),
        "action_bound_violation_rate": rate(rows, "action_bound_violation"),
        "event_order_accuracy": rate(order_rows, "event_order_pass"),
        "event_order_valid_denominator": len(order_rows),
        "forbidden_contact_time_ratio_mean": (
            sum(row["forbidden_contact_time_ratio"] for row in rows) / len(rows)
            if rows else None
        ),
        "joint_limit_violation_time_ratio_mean": (
            sum(row["joint_limit_violation_time_ratio"] for row in rows) / len(rows)
            if rows else None
        ),
        "friction_cone_failure_time_ratio_mean": (
            sum(row["friction_cone_failure_time_ratio"] for row in rows) / len(rows)
            if rows else None
        ),
        "friction_contact_sample_count_total": sum(row["friction_contact_sample_count"] for row in rows),
        "dynamic_penetration_episode_p95_distribution_p95_m": percentile(
            [row["dynamic_penetration_p95_m"] for row in rows], 0.95
        ),
        "maximum_penetration_m": (
            max(row["maximum_penetration_m"] for row in rows) if rows else None
        ),
        "contact_force_episode_p95_distribution_p95_N": percentile(
            [row["contact_force_p95_N"] for row in rows], 0.95
        ),
        "friction_utilization_episode_p95_distribution_p95": percentile([row["friction_utilization_p95"] for row in rows], 0.95),
        "maximum_friction_utilization": max((row["friction_utilization_max"] for row in rows), default=None),
        "tangential_slip_episode_p95_distribution_p95_mps": percentile([row["tangential_slip_speed_p95_mps"] for row in rows], 0.95),
        "maximum_tangential_slip_speed_mps": max((row["tangential_slip_speed_max_mps"] for row in rows), default=None),
        "maximum_generalized_force_residual_relative": max((row["generalized_force_residual_relative_max"] for row in rows), default=None),
        "maximum_constraint_force_residual_relative": max((row["constraint_force_residual_relative_max"] for row in rows), default=None),
        "maximum_forward_inverse_residual_relative": max((row["generalized_force_residual_relative_max"] for row in rows), default=None),
        "maximum_quaternion_norm_error": max((row["maximum_quaternion_norm_error"] for row in rows), default=None),
        "maximum_joint_speed": max((row["maximum_joint_speed"] for row in rows), default=None),
        "linear_acceleration_episode_max_distribution_p95_mps2": percentile([row["maximum_linear_acceleration_mps2"] for row in rows], 0.95),
        "angular_acceleration_episode_max_distribution_p95_radps2": percentile([row["maximum_angular_acceleration_radps2"] for row in rows], 0.95),
        "linear_jerk_episode_max_distribution_p95_mps3": percentile([row["maximum_linear_jerk_mps3"] for row in rows], 0.95),
        "angular_jerk_episode_max_distribution_p95_radps3": percentile([row["maximum_angular_jerk_radps3"] for row in rows], 0.95),
        "total_energy_range_episode_distribution_p95_J": percentile([row["total_energy_range_J"] for row in rows], 0.95),
        "kinetic_energy_episode_max_distribution_p95_J": percentile([row["kinetic_energy_max_J"] for row in rows], 0.95),
        "minimum_terminal_margin_mps": (
            min(row["terminal_margin_mps"] for row in rows) if rows else None
        ),
        "wall_time_s_total": sum(row["wall_time_s"] for row in rows),
    }


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def format_number(value: float | None, scale: float = 1.0, digits: int = 3) -> str:
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()

    suite_root = args.suite.resolve()
    workspace = suite_root.parents[1]
    run_root = args.run_root.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError("output already exists: " + str(output_root))

    registry_path = suite_root / "config/suite.json"
    manifest_path = run_root / "run_manifest.json"
    registry = load_json(registry_path)
    manifest = load_json(manifest_path)
    registry_hash = sha256_file(registry_path)
    if manifest["suite_registry_sha256"] != registry_hash:
        raise RuntimeError("run manifest was produced from a different suite registry")
    if manifest["runtime_sha256"] != sha256_file(suite_root / "tools/runtime.py"):
        raise RuntimeError("runtime changed after matrix execution")
    if manifest.get("physics_diagnostics_sha256") != sha256_file(suite_root / "tools/physics_diagnostics.py"):
        raise RuntimeError("physics diagnostics changed after matrix execution")
    expected_toolchain = {label: ref["sha256"] for label, ref in registry.get("toolchain", {}).items()}
    if manifest.get("toolchain_hashes") != expected_toolchain:
        raise RuntimeError("matrix manifest toolchain hashes do not match the suite registry")
    for label, ref in registry.get("toolchain", {}).items():
        if sha256_file(workspace / ref["path"]) != ref["sha256"]:
            raise RuntimeError("current toolchain artifact differs from registry: " + label)
    if manifest["execution_errors"]:
        raise RuntimeError("matrix contains execution errors")
    if not manifest["execution_complete"]:
        raise RuntimeError("matrix execution is incomplete")

    registered_cases = [item["case_id"] for item in registry["cases"]]
    case_by_id = {item["case_id"]: item for item in registry["cases"]}
    registered_generations = registry["generation_seeds"]
    registered_physics = registry["physics_seeds"]
    selection = manifest["selection"]
    if not args.allow_subset:
        if selection["case_ids"] != registered_cases:
            raise RuntimeError("formal aggregation requires every registered case")
        if selection["generation_seeds"] != registered_generations:
            raise RuntimeError("formal aggregation requires every generation seed")
        if selection["physics_seeds"] != registered_physics:
            raise RuntimeError("formal aggregation requires every physics seed")

    selected_cases = selection["case_ids"]
    selected_generations = selection["generation_seeds"]
    selected_physics = selection["physics_seeds"]
    expected_keys = {
        (case_id, generation_seed, physics_seed)
        for case_id in selected_cases
        for generation_seed in selected_generations
        for physics_seed in selected_physics
    }
    if manifest["expected_episode_count"] != len(expected_keys):
        raise RuntimeError("manifest expected count does not match its selection")
    if manifest["completed_episode_count"] != len(expected_keys):
        raise RuntimeError("manifest completed count does not match its selection")

    expected_hashes = expected_artifact_hashes(workspace, suite_root, registry)
    seen: set[tuple[str, int, int]] = set()
    episodes: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    replay_fingerprints: dict[tuple[str, int], set[str]] = {}

    for manifest_row in manifest["episodes"]:
        key = (
            manifest_row["case_id"],
            int(manifest_row["generation_seed"]),
            int(manifest_row["physics_seed"]),
        )
        if key in seen:
            raise RuntimeError("duplicate episode in manifest: " + repr(key))
        if key not in expected_keys:
            raise RuntimeError("unexpected episode in manifest: " + repr(key))
        seen.add(key)
        case_id, generation_seed, physics_seed = key
        episode_dir = relative_episode_dir(run_root, manifest_row["output_dir"])
        raw_path = episode_dir / "episode_log.json"
        validation_path = episode_dir / "validation.json"
        raw = load_json(raw_path)
        validation = load_json(validation_path)
        raw_identity = (raw["case_id"], raw["generation_seed"], raw["physics_seed"])
        validation_identity = (
            validation["case_id"],
            validation["generation_seed"],
            validation["physics_seed"],
        )
        if raw_identity != key or validation_identity != key:
            raise RuntimeError("episode identity mismatch: " + repr(key))
        raw_hash = sha256_file(raw_path)
        if validation["raw_event_log_sha256"] != raw_hash:
            raise RuntimeError("validation/raw hash mismatch: " + repr(key))
        if manifest_row["raw_event_log_sha256"] != raw_hash:
            raise RuntimeError("manifest/raw hash mismatch: " + repr(key))
        if raw["artifact_hashes"] != expected_hashes[case_id]:
            raise RuntimeError("artifact hash mismatch: " + repr(key))
        if manifest_row["artifact_hashes"] != expected_hashes[case_id]:
            raise RuntimeError("manifest artifact hash mismatch: " + repr(key))
        if validation["validator_sha256"] != expected_hashes[case_id]["validator_sha256"]:
            raise RuntimeError("validation uses unexpected validator: " + repr(key))

        item = case_by_id[case_id]
        gold = load_json(workspace / item["case_gold"]["path"])
        frames = raw["frames"]
        if not frames:
            raise RuntimeError("episode has no frames: " + repr(key))
        hold_s = float(gold["thresholds"]["terminal_hold_s"])
        window_start = float(frames[-1]["t"]) - hold_s
        window = [frame for frame in frames if float(frame["t"]) >= window_start]
        terminal_body = item["terminal_body"]
        maximum_terminal_speed = max(
            vector_norm(frame["bodies"][terminal_body]["linear_velocity"])
            for frame in window
        )
        maximum_allowed_speed = float(
            gold["thresholds"]["maximum_terminal_speed_mps"]
        )
        terminal_margin = maximum_allowed_speed - maximum_terminal_speed

        predicates = {
            predicate["id"]: predicate
            for predicate in validation["atomic_predicates"]
        }
        order_id = item.get("event_order_predicate")
        order_applicable = order_id is not None
        order_pass = bool(predicates[order_id]["pass"]) if order_applicable else False
        for predicate in validation["atomic_predicates"]:
            details = {
                key_name: value
                for key_name, value in predicate.items()
                if key_name not in {"id", "pass", "mandatory"}
            }
            atomic_rows.append({
                "case_id": case_id,
                "generation_seed": generation_seed,
                "physics_seed": physics_seed,
                "predicate_id": predicate["id"],
                "mandatory": int(predicate.get("mandatory", True)),
                "pass": int(bool(predicate["pass"])),
                "details_json": json.dumps(
                    details, sort_keys=True, separators=(",", ":")
                ),
            })

        episode = {
            "case_id": case_id,
            "generation_seed": generation_seed,
            "physics_seed": physics_seed,
            "status": validation["status"],
            "task_predicates_pass": bool(validation["task_predicates_pass"]),
            "safety_pass": bool(validation["safety_pass"]),
            "integrity_pass": bool(validation["integrity_pass"]),
            "dev_reference_success": bool(validation["dev_reference_success"]),
            "forbidden_contact_episode": bool(
                validation["forbidden_contact_episode"]
            ),
            "forbidden_contact_time_ratio": float(
                validation["forbidden_contact_time_ratio"]
            ),
            "joint_limit_violation_episode": bool(
                validation["joint_limit_violation_episode"]
            ),
            "joint_limit_violation_time_ratio": float(
                validation["joint_limit_violation_time_ratio"]
            ),
            "friction_cone_failure_episode": bool(validation["friction_cone_failure_episode"]),
            "friction_cone_failure_time_ratio": float(validation["friction_cone_failure_time_ratio"]),
            "friction_utilization_p95": float(validation["friction_utilization_p95"]),
            "friction_utilization_max": float(validation["friction_utilization_max"]),
            "tangential_slip_speed_p95_mps": float(validation["tangential_slip_speed_p95_mps"]),
            "tangential_slip_speed_max_mps": float(validation["tangential_slip_speed_max_mps"]),
            "friction_contact_sample_count": int(validation["friction_contact_sample_count"]),
            "slip_reporting_sample_count": int(validation["slip_reporting_sample_count"]),
            "mechanics_consistency_pass": bool(validation["mechanics_consistency_pass"]),
            "kinematics_consistency_pass": bool(validation["kinematics_consistency_pass"]),
            "diagnostics_finite": bool(validation["diagnostics_finite"]),
            "generalized_force_residual_relative_max": float(validation["generalized_force_residual_relative_max"]),
            "constraint_force_residual_relative_max": float(validation["constraint_force_residual_relative_max"]),
            "forward_inverse_residual_relative_max": float(validation["generalized_force_residual_relative_max"]),
            "maximum_quaternion_norm_error": float(validation["maximum_quaternion_norm_error"]),
            "maximum_joint_speed": float(validation["maximum_joint_speed"]),
            "maximum_linear_acceleration_mps2": float(validation["maximum_linear_acceleration_mps2"]),
            "maximum_angular_acceleration_radps2": float(validation["maximum_angular_acceleration_radps2"]),
            "maximum_linear_jerk_mps3": float(validation["maximum_linear_jerk_mps3"]),
            "maximum_angular_jerk_radps3": float(validation["maximum_angular_jerk_radps3"]),
            "total_energy_range_J": float(validation["total_energy_range_J"]),
            "kinetic_energy_max_J": float(validation["kinetic_energy_max_J"]),
            "action_bound_violation": bool(validation["action_bound_violation"]),
            "dynamic_penetration_p95_m": float(
                validation["dynamic_penetration_p95_m"]
            ),
            "maximum_penetration_m": float(validation["maximum_penetration_m"]),
            "contact_force_p95_N": float(raw["summary"]["contact_force_p95_N"]),
            "maximum_terminal_speed_mps": maximum_terminal_speed,
            "terminal_speed_threshold_mps": maximum_allowed_speed,
            "terminal_margin_mps": terminal_margin,
            "event_order_applicable": order_applicable,
            "event_order_pass": order_pass,
            "wall_time_s": float(raw["summary"]["wall_time_s"]),
            "raw_event_log_sha256": raw_hash,
        }
        episodes.append(episode)
        replay_fingerprints.setdefault(
            (case_id, physics_seed), set()
        ).add(canonical_validation_fingerprint(validation))

    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        raise RuntimeError("matrix is missing episodes: " + repr(missing))

    episodes.sort(
        key=lambda row: (
            selected_cases.index(row["case_id"]),
            row["generation_seed"],
            row["physics_seed"],
        )
    )
    atomic_rows.sort(
        key=lambda row: (
            selected_cases.index(row["case_id"]),
            row["generation_seed"],
            row["physics_seed"],
            row["predicate_id"],
        )
    )

    replay_checks = []
    for case_id in selected_cases:
        for physics_seed in selected_physics:
            fingerprints = replay_fingerprints[(case_id, physics_seed)]
            replay_checks.append({
                "case_id": case_id,
                "physics_seed": physics_seed,
                "generation_slot_count": len(selected_generations),
                "outcome_fingerprint_count": len(fingerprints),
                "equivalent": len(fingerprints) == 1,
            })
    replay_equivalent = all(item["equivalent"] for item in replay_checks)

    per_case = []
    all_seed_group_count = 0
    all_seed_group_pass_count = 0
    for case_id in selected_cases:
        case_rows = [row for row in episodes if row["case_id"] == case_id]
        metrics = summarize(case_rows)
        generation_groups = []
        for generation_seed in selected_generations:
            group = [
                row for row in case_rows
                if row["generation_seed"] == generation_seed
            ]
            passed = (
                len(group) == len(selected_physics)
                and all(row["dev_reference_success"] for row in group)
            )
            generation_groups.append({
                "generation_seed": generation_seed,
                "physics_seed_count": len(group),
                "all_physics_seeds_pass": passed,
            })
            all_seed_group_count += 1
            all_seed_group_pass_count += int(passed)
        metrics.update({
            "case_id": case_id,
            "generation_groups": generation_groups,
            "all_physics_seeds_pass": all(
                group["all_physics_seeds_pass"] for group in generation_groups
            ),
        })
        per_case.append(metrics)

    overall = summarize(episodes)
    unique_physics_rows = [
        row for row in episodes
        if row["generation_seed"] == selected_generations[0]
    ]
    overall.update({
        "execution_rate": len(episodes) / len(expected_keys),
        "expected_episode_count": len(expected_keys),
        "completed_episode_count": len(episodes),
        "effective_case_physics_trials": len(unique_physics_rows),
        "generation_replay_slot_count": len(selected_generations),
        "all_seed_generation_group_count": all_seed_group_count,
        "all_seed_generation_group_pass_count": all_seed_group_pass_count,
        "all_seed_generation_group_pass_rate": (
            all_seed_group_pass_count / all_seed_group_count
            if all_seed_group_count else None
        ),
        "all_seed_case_pass_count": sum(
            row["all_physics_seeds_pass"] for row in per_case
        ),
        "all_seed_case_count": len(per_case),
        "all_seed_case_pass_rate": (
            sum(row["all_physics_seeds_pass"] for row in per_case) / len(per_case)
            if per_case else None
        ),
        "unique_physics_robust_task_success_rate": rate(
            unique_physics_rows, "dev_reference_success"
        ),
    })

    atomic_summary = []
    for case_id in selected_cases:
        predicate_ids = sorted({
            row["predicate_id"]
            for row in atomic_rows
            if row["case_id"] == case_id
        })
        for predicate_id in predicate_ids:
            rows = [
                row for row in atomic_rows
                if row["case_id"] == case_id
                and row["predicate_id"] == predicate_id
            ]
            atomic_summary.append({
                "case_id": case_id,
                "predicate_id": predicate_id,
                "pass_count": sum(row["pass"] for row in rows),
                "valid_denominator": len(rows),
                "pass_rate": sum(row["pass"] for row in rows) / len(rows),
            })

    output_root.mkdir(parents=True, exist_ok=False)
    episode_fields = list(episodes[0]) if episodes else []
    episode_csv_rows = [
        {
            key: (int(value) if isinstance(value, bool) else value)
            for key, value in row.items()
        }
        for row in episodes
    ]
    write_csv(output_root / "episodes.csv", episode_csv_rows, episode_fields)
    write_csv(
        output_root / "atomic_predicates.csv",
        atomic_rows,
        [
            "case_id", "generation_seed", "physics_seed", "predicate_id",
            "mandatory", "pass", "details_json",
        ],
    )

    per_case_fields = ["case_id"] + [
        key for key in per_case[0]
        if key not in {"case_id", "generation_groups"}
    ] if per_case else []
    per_case_csv_rows = []
    for row in per_case:
        flattened = {
            key: (int(row[key]) if isinstance(row[key], bool) else row[key])
            for key in per_case_fields
        }
        per_case_csv_rows.append(flattened)
    write_csv(
        output_root / "per_case_metrics.csv",
        per_case_csv_rows,
        per_case_fields,
    )
    write_csv(
        output_root / "atomic_summary.csv",
        atomic_summary,
        [
            "case_id", "predicate_id", "pass_count",
            "valid_denominator", "pass_rate",
        ],
    )

    report = {
        "schema_version": "simcode4d.dev15.formal-metrics.v1",
        "suite_id": registry["suite_id"],
        "suite_status": registry["status"],
        "hash_bindings": {
            "suite_registry_sha256": registry_hash,
            "semantic_freeze_sha256": registry["semantic_freeze"]["sha256"],
            "run_manifest_sha256": sha256_file(manifest_path),
            "runtime_sha256": manifest["runtime_sha256"],
            "physics_diagnostics_sha256": manifest["physics_diagnostics_sha256"],
            "runner_sha256": manifest["runner_sha256"],
            "aggregator_sha256": sha256_file(Path(__file__)),
        },
        "denominator_policy": {
            "registered_episode_denominator": len(expected_keys),
            "effective_case_physics_trial_denominator": len(unique_physics_rows),
            "generation_replay_note": registry["replication_note"],
            "missing_measurements": "NA; never removed silently from a denominator",
        },
        "metric_definitions": {
            "robust_task_success": (
                "task predicates AND physical/safety admissibility AND "
                "integrity, with every registered episode in the denominator"
            ),
            "all_physics_seeds_pass": (
                "all five registered physics seeds pass within every "
                "generation replay slot for the case"
            ),
            "dynamic_penetration": (
                "per-episode P95 contact penetration; table value is the "
                "P95 across episode-level P95 values"
            ),
            "mechanics_consistency": "finite diagnostics with MuJoCo qfrc generalized-force residual <= 1e-5; efc constraint residual is reported separately and is not the gate",
            "kinematics_consistency": "finite pose/twist/acceleration/jerk diagnostics and quaternion norm error <= 1e-6",
            "friction_cone": "contact-dimension-aware sliding/torsional/rolling Coulomb utilization; failure requires >1.25 for >=10 ms",
            "friction_contact_coverage": "fraction of episodes with at least one resolved contact sample; zero-contact episodes remain in the denominator",
            "terminal_margin_mps": (
                "maximum allowed terminal speed minus the maximum measured "
                "body speed over the frozen final hold window"
            ),
            "contact_force": (
                "diagnostic only; solver- and timestep-dependent"
            ),
        },
        "artifact_audit": {
            "verified_episode_count": len(episodes),
            "hash_mismatch_count": 0,
            "duplicate_episode_count": 0,
            "missing_episode_count": 0,
        },
        "generation_replay": {
            "all_outcomes_equivalent": replay_equivalent,
            "checks": replay_checks,
        },
        "overall": overall,
        "per_case": per_case,
        "atomic_summary": atomic_summary,
    }
    write_json(output_root / "formal_metrics.json", report)

    markdown = [
        "# SimCode4D Dev15 formal metrics",
        "",
        f"Suite: {registry['suite_id']} ({registry['status']}).",
        "",
        "| Case | Episodes | Task | Physical | Integrity | Mechanics | Kinematics | Friction-cone pass | Friction coverage | RobustTaskSuccess | All 5 seeds | Dyn. pen. P95 (mm) | qfrc residual max | efc residual max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in per_case:
        markdown.append(
            "| {case_id} | {episode_count} | {task} | {physical} | "
            "{integrity} | {mechanics} | {kinematics} | {friction_pass} | "
            "{friction_coverage} | {robust} | {all_seeds} | {penetration} | "
            "{residual} | {constraint_residual} |".format(
                case_id=row["case_id"],
                episode_count=row["episode_count"],
                task=format_rate(row["task_success_rate"]),
                physical=format_rate(row["safety_pass_rate"]),
                integrity=format_rate(row["integrity_pass_rate"]),
                mechanics=format_rate(row["mechanics_consistency_pass_rate"]),
                kinematics=format_rate(row["kinematics_consistency_pass_rate"]),
                friction_pass=format_rate(
                    None if row["friction_cone_failure_episode_rate"] is None
                    else 1.0 - row["friction_cone_failure_episode_rate"]
                ),
                friction_coverage=format_rate(row["friction_contact_coverage_rate"]),
                robust=format_rate(row["robust_task_success_rate"]),
                all_seeds="yes" if row["all_physics_seeds_pass"] else "no",
                penetration=format_number(
                    row["dynamic_penetration_episode_p95_distribution_p95_m"],
                    scale=1000.0,
                ),
                residual=format_number(row["maximum_generalized_force_residual_relative"], digits=6),
                constraint_residual=format_number(row["maximum_constraint_force_residual_relative"], digits=6),
            )
        )
    markdown.extend([
        "",
        "Overall:",
        "",
        f"- Execution: {len(episodes)}/{len(expected_keys)} "
        f"({format_rate(overall['execution_rate'])}).",
        f"- RobustTaskSuccess: "
        f"{format_rate(overall['robust_task_success_rate'])}.",
        f"- Mechanics consistency pass: "
        f"{format_rate(overall['mechanics_consistency_pass_rate'])}.",
        f"- Kinematics consistency pass: "
        f"{format_rate(overall['kinematics_consistency_pass_rate'])}.",
        f"- Friction-cone failure episodes: "
        f"{format_rate(overall['friction_cone_failure_episode_rate'])}.",
        f"- Friction contact coverage: "
        f"{format_rate(overall['friction_contact_coverage_rate'])}.",
        f"- All-seeds case pass: {overall['all_seed_case_pass_count']}/"
        f"{overall['all_seed_case_count']}.",
        f"- Event-order accuracy: "
        f"{format_rate(overall['event_order_accuracy'])} "
        f"(valid denominator {overall['event_order_valid_denominator']}).",
        f"- Generation replay outcomes equivalent: "
        f"{'yes' if replay_equivalent else 'no'}.",
        "",
        f"The {len(episodes)} registered executions contain {len(unique_physics_rows)} distinct case–physics trials "
        f"and {len(selected_generations)} deterministic generation replay slots. The replay slots are "
        "reported for protocol completeness and are not treated as independent "
        "world generations. Contact force, acceleration, jerk, slip, and energy magnitudes are reported; frozen residual, normalization, penetration, contact, joint-limit, friction-cone, and integrity rules determine physical admissibility.",
        "",
    ])
    (output_root / "formal_metrics.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    checksum_files = [
        "episodes.csv",
        "atomic_predicates.csv",
        "per_case_metrics.csv",
        "atomic_summary.csv",
        "formal_metrics.json",
        "formal_metrics.md",
    ]
    write_json(
        output_root / "checksums.sha256.json",
        {
            "schema_version": "simcode4d.dev15.result-checksums.v1",
            "files": {
                name: sha256_file(output_root / name)
                for name in checksum_files
            },
        },
    )
    print(json.dumps({
        "episodes": len(episodes),
        "robust_task_success_rate": overall["robust_task_success_rate"],
        "all_seed_cases": (
            f"{overall['all_seed_case_pass_count']}/"
            f"{overall['all_seed_case_count']}"
        ),
        "generation_replay_equivalent": replay_equivalent,
        "output": str(output_root),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
