"""Benchmark-owned mechanics and kinematics diagnostics for episode logs."""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


def mechanics_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Return energy and forward/inverse consistency without changing state."""
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    mujoco.mj_compareFwdInv(model, data)
    # MuJoCo reports two distinct comparison channels in solver_fwdinv:
    # generalized force (qfrc) and constraint force (efc). Only qfrc is the
    # forward/inverse dynamics consistency residual. The efc channel is kept
    # as a solver diagnostic, but must not be folded into the mechanics gate.
    residual_components = [float(value) for value in data.solver_fwdinv]
    generalized_force_residual = abs(residual_components[0])
    constraint_force_residual = abs(residual_components[1])
    generalized_force_scale = max(1.0, float(np.linalg.norm(data.qfrc_inverse)))
    constraint_force_scale = max(1.0, float(np.linalg.norm(data.efc_force)))
    generalized_force_relative = generalized_force_residual / generalized_force_scale
    constraint_force_relative = constraint_force_residual / constraint_force_scale
    return {
        "potential_energy_J": float(data.energy[0]),
        "kinetic_energy_J": float(data.energy[1]),
        "total_energy_J": float(data.energy[0] + data.energy[1]),
        "forward_inverse_residual_components": residual_components,
        "generalized_force_residual_absolute": generalized_force_residual,
        "generalized_force_residual_relative": generalized_force_relative,
        "constraint_force_residual_absolute": constraint_force_residual,
        "constraint_force_residual_relative": constraint_force_relative,
        # Backward-compatible aliases are explicitly bound to qfrc only.
        "forward_inverse_residual_max": generalized_force_residual,
        "forward_inverse_residual_relative": generalized_force_relative,
    }


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def annotate_kinematics(frames: list[dict[str, Any]]) -> dict[str, float]:
    """Add finite-difference acceleration/jerk to logged body observations."""
    maximum_linear_acceleration = 0.0
    maximum_angular_acceleration = 0.0
    maximum_linear_jerk = 0.0
    maximum_angular_jerk = 0.0
    maximum_quaternion_norm_error = 0.0
    previous_velocity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    previous_acceleration: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    previous_time: float | None = None

    for frame in frames:
        current_time = float(frame["t"])
        dt = current_time - previous_time if previous_time is not None else 0.0
        for body_name, body in frame["bodies"].items():
            linear_velocity = np.asarray(body["linear_velocity"], dtype=float)
            angular_velocity = np.asarray(body["angular_velocity"], dtype=float)
            if dt > 0.0 and body_name in previous_velocity:
                old_linear, old_angular = previous_velocity[body_name]
                linear_acceleration = (linear_velocity - old_linear) / dt
                angular_acceleration = (angular_velocity - old_angular) / dt
            else:
                linear_acceleration = np.zeros(3)
                angular_acceleration = np.zeros(3)
            if dt > 0.0 and body_name in previous_acceleration:
                old_linear_acceleration, old_angular_acceleration = previous_acceleration[body_name]
                linear_jerk = (linear_acceleration - old_linear_acceleration) / dt
                angular_jerk = (angular_acceleration - old_angular_acceleration) / dt
            else:
                linear_jerk = np.zeros(3)
                angular_jerk = np.zeros(3)

            body["linear_acceleration"] = linear_acceleration.tolist()
            body["angular_acceleration"] = angular_acceleration.tolist()
            body["linear_jerk"] = linear_jerk.tolist()
            body["angular_jerk"] = angular_jerk.tolist()
            maximum_linear_acceleration = max(maximum_linear_acceleration, float(np.linalg.norm(linear_acceleration)))
            maximum_angular_acceleration = max(maximum_angular_acceleration, float(np.linalg.norm(angular_acceleration)))
            maximum_linear_jerk = max(maximum_linear_jerk, float(np.linalg.norm(linear_jerk)))
            maximum_angular_jerk = max(maximum_angular_jerk, float(np.linalg.norm(angular_jerk)))
            maximum_quaternion_norm_error = max(
                maximum_quaternion_norm_error,
                abs(_norm(body["quaternion"]) - 1.0),
            )
            previous_velocity[body_name] = (linear_velocity, angular_velocity)
            previous_acceleration[body_name] = (linear_acceleration, angular_acceleration)
        previous_time = current_time

    maximum_joint_speed = max(
        (abs(float(joint["velocity"])) for frame in frames for joint in frame["joints"].values()),
        default=0.0,
    )
    return {
        "maximum_linear_acceleration_mps2": maximum_linear_acceleration,
        "maximum_angular_acceleration_radps2": maximum_angular_acceleration,
        "maximum_linear_jerk_mps3": maximum_linear_jerk,
        "maximum_angular_jerk_radps3": maximum_angular_jerk,
        "maximum_joint_speed": maximum_joint_speed,
        "maximum_quaternion_norm_error": maximum_quaternion_norm_error,
    }


def energy_summary(frames: list[dict[str, Any]]) -> dict[str, float]:
    total = [float(frame["mechanics"]["total_energy_J"]) for frame in frames]
    potential = [float(frame["mechanics"]["potential_energy_J"]) for frame in frames]
    kinetic = [float(frame["mechanics"]["kinetic_energy_J"]) for frame in frames]
    generalized_force_residual = [
        float(frame["mechanics"]["generalized_force_residual_relative"])
        for frame in frames
    ]
    constraint_force_residual = [
        float(frame["mechanics"]["constraint_force_residual_relative"])
        for frame in frames
    ]
    return {
        "initial_total_energy_J": total[0] if total else 0.0,
        "final_total_energy_J": total[-1] if total else 0.0,
        "total_energy_range_J": max(total) - min(total) if total else 0.0,
        "potential_energy_range_J": max(potential) - min(potential) if potential else 0.0,
        "kinetic_energy_max_J": max(kinetic, default=0.0),
        "generalized_force_residual_relative_max": max(generalized_force_residual, default=0.0),
        "constraint_force_residual_relative_max": max(constraint_force_residual, default=0.0),
        # Backward-compatible alias for consumers of the RC1 draft format.
        "forward_inverse_residual_relative_max": max(generalized_force_residual, default=0.0),
    }
