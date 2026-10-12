"""Shared read-only helpers for independent case validators."""
from __future__ import annotations

import fnmatch
import math


def planar_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def in_aabb(position, region):
    return all(lo <= value <= hi for value, lo, hi in zip(position, region["min_xyz_m"], region["max_xyz_m"]))


def final_window(frames, hold_s):
    if not frames:
        return []
    start = frames[-1]["t"] - hold_s
    return [frame for frame in frames if frame["t"] >= start]


def body_speed(frame, body_name):
    return norm(frame["bodies"][body_name]["linear_velocity"])


def first_time(frames, predicate):
    for frame in frames:
        if predicate(frame):
            return frame["t"]
    return None


def _contact_endpoint_matches(contact, endpoint, binding):
    geom_name = contact["geom" + str(endpoint)]
    body_name = contact["body" + str(endpoint)]
    return (
        any(fnmatch.fnmatchcase(geom_name, pattern) for pattern in binding.get("geoms", []))
        or any(fnmatch.fnmatchcase(body_name, pattern) for pattern in binding.get("bodies", []))
    )


def semantic_contact_matches(frame, policy, left, right):
    groups = policy["groups"]
    left_binding = groups[left]
    right_binding = groups[right]
    for contact in frame["contacts"]:
        if contact["normal_force_N"] <= 0.0 and contact["distance_m"] >= 0.0:
            continue
        direct = _contact_endpoint_matches(contact, 1, left_binding) and _contact_endpoint_matches(contact, 2, right_binding)
        reverse = _contact_endpoint_matches(contact, 2, left_binding) and _contact_endpoint_matches(contact, 1, right_binding)
        if direct or reverse:
            return True
    return False


def roll_pitch(quaternion):
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


FORCE_BALANCE_RELATIVE_TOLERANCE = 1e-5
QUATERNION_NORM_TOLERANCE = 1e-6


def base_result(raw, predicates):
    initial = raw["initial_admissibility"]
    safety = raw["safety"]
    integrity_audit = raw["integrity_audit"]
    integrity = bool(
        raw["finite_execution"]
        and raw["model_mutation_guard_pass"]
        and not integrity_audit["integrity_violation"]
    )
    task_pass = all(item["pass"] for item in predicates if item.get("mandatory", True))
    summary = raw["summary"]
    diagnostic_values = [
        summary.get("generalized_force_residual_relative_max", math.inf),
        summary.get("constraint_force_residual_relative_max", math.inf),
        summary.get("maximum_quaternion_norm_error", math.inf),
        summary.get("maximum_linear_acceleration_mps2", math.inf),
        summary.get("maximum_angular_acceleration_radps2", math.inf),
        summary.get("maximum_linear_jerk_mps3", math.inf),
        summary.get("maximum_angular_jerk_radps3", math.inf),
        summary.get("total_energy_range_J", math.inf),
    ]
    diagnostics_finite = all(math.isfinite(value) for value in diagnostic_values)
    mechanics_pass = bool(
        diagnostics_finite
        and diagnostic_values[0] <= FORCE_BALANCE_RELATIVE_TOLERANCE
    )
    kinematics_pass = bool(
        diagnostics_finite
        and diagnostic_values[2] <= QUATERNION_NORM_TOLERANCE
    )
    friction_pass = not safety.get("friction_cone_failure_episode", False)
    physical = bool(
        initial["penetration_pass_1mm"]
        and summary["maximum_penetration_m"] <= 0.001
        and not safety["forbidden_contact_episode"]
        and not safety["joint_limit_violation_episode"]
        and mechanics_pass
        and kinematics_pass
        and friction_pass
    )
    return {
        "schema_version": "simcode4d.dev15.validation.v1",
        "case_id": raw["case_id"],
        "generation_seed": raw["generation_seed"],
        "physics_seed": raw["physics_seed"],
        "status": "pass" if task_pass and physical and integrity else "fail",
        "atomic_predicates": predicates,
        "task_predicates_pass": task_pass,
        "physical_admissibility_pass": physical,
        "integrity_pass": integrity,
        "safety_pass": physical,
        "forbidden_contact_episode": safety["forbidden_contact_episode"],
        "forbidden_contact_time_ratio": safety["forbidden_contact_time_ratio"],
        "joint_limit_violation_episode": safety["joint_limit_violation_episode"],
        "joint_limit_violation_time_ratio": safety["joint_limit_violation_time_ratio"],
        "friction_cone_failure_episode": safety.get("friction_cone_failure_episode", False),
        "friction_cone_failure_time_ratio": safety.get("friction_cone_failure_time_ratio", 0.0),
        "friction_utilization_p95": safety.get("friction_utilization_p95", 0.0),
        "friction_utilization_max": safety.get("friction_utilization_max", 0.0),
        "tangential_slip_speed_p95_mps": safety.get("tangential_slip_speed_p95_mps", 0.0),
        "tangential_slip_speed_max_mps": safety.get("tangential_slip_speed_max_mps", 0.0),
        "friction_contact_sample_count": summary.get("friction_contact_sample_count", 0),
        "slip_reporting_sample_count": summary.get("slip_reporting_sample_count", 0),
        "dynamic_penetration_p95_m": safety["dynamic_penetration_p95_m"],
        "mechanics_consistency_pass": mechanics_pass,
        "kinematics_consistency_pass": kinematics_pass,
        "diagnostics_finite": diagnostics_finite,
        "force_balance_relative_tolerance": FORCE_BALANCE_RELATIVE_TOLERANCE,
        "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
        "generalized_force_residual_relative_max": summary.get("generalized_force_residual_relative_max", math.inf),
        "constraint_force_residual_relative_max": summary.get("constraint_force_residual_relative_max", math.inf),
        "forward_inverse_residual_relative_max": summary.get("generalized_force_residual_relative_max", math.inf),
        "maximum_quaternion_norm_error": summary.get("maximum_quaternion_norm_error", math.inf),
        "maximum_joint_speed": summary.get("maximum_joint_speed", math.inf),
        "maximum_linear_acceleration_mps2": summary.get("maximum_linear_acceleration_mps2", math.inf),
        "maximum_angular_acceleration_radps2": summary.get("maximum_angular_acceleration_radps2", math.inf),
        "maximum_linear_jerk_mps3": summary.get("maximum_linear_jerk_mps3", math.inf),
        "maximum_angular_jerk_radps3": summary.get("maximum_angular_jerk_radps3", math.inf),
        "total_energy_range_J": summary.get("total_energy_range_J", math.inf),
        "kinetic_energy_max_J": summary.get("kinetic_energy_max_J", math.inf),
        "action_bound_violation": integrity_audit["action_bound_violation"],
        "undeclared_equality_activation": integrity_audit["undeclared_equality_activation"],
        "dev_reference_success": bool(task_pass and physical and integrity),
        "maximum_penetration_m": raw["summary"]["maximum_penetration_m"],
    }
