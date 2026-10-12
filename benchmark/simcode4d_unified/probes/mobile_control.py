"""Pure-observation controls shared by the nine Dev15 handover cases."""
from __future__ import annotations

import math

from probe_api import clamp, wrap_angle, yaw_from_quat_wxyz


def smootherstep(value: float) -> float:
    u = clamp(value, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def smootherstep_rate(value: float) -> float:
    u = clamp(value, 0.0, 1.0)
    return 30.0 * u * u * (u - 1.0) * (u - 1.0)


def roll_pitch(quaternion: list[float]) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def planar_reference_controls(
    t: float,
    observation: dict,
    body_name: str,
    start_xyz: list[float],
    end_xyz: list[float],
    move_duration_s: float,
    target_yaw_rad: float,
    force_limit_N: float,
    vertical_force_limit_N: float,
    torque_limit_Nm: float,
    position_kp: float,
    velocity_kd: float,
    attitude_kp: float,
    attitude_kd: float,
) -> dict[str, float]:
    """Track a smooth path using only declared free-joint motors."""
    duration = max(float(move_duration_s), 1e-9)
    phase = clamp(t / duration, 0.0, 1.0)
    blend = smootherstep(phase)
    rate = smootherstep_rate(phase) / duration if t < duration else 0.0
    target = [
        start_xyz[index] + blend * (end_xyz[index] - start_xyz[index])
        for index in range(3)
    ]
    target_velocity = [
        rate * (end_xyz[index] - start_xyz[index])
        for index in range(3)
    ]
    state = observation["bodies"][body_name]
    position = state["position"]
    velocity = state["linear_velocity"]
    angular_velocity = state["angular_velocity"]

    force_x = position_kp * (target[0] - position[0]) + velocity_kd * (
        target_velocity[0] - velocity[0]
    )
    force_y = position_kp * (target[1] - position[1]) + velocity_kd * (
        target_velocity[1] - velocity[1]
    )
    magnitude = math.hypot(force_x, force_y)
    if magnitude > force_limit_N:
        force_x *= force_limit_N / magnitude
        force_y *= force_limit_N / magnitude
    force_z = clamp(
        position_kp * (target[2] - position[2])
        + velocity_kd * (target_velocity[2] - velocity[2]),
        -vertical_force_limit_N,
        vertical_force_limit_N,
    )

    body_roll, body_pitch = roll_pitch(state["quaternion"])
    yaw = yaw_from_quat_wxyz(state["quaternion"])
    return {
        "base_force_x_ctrl": force_x,
        "base_force_y_ctrl": force_y,
        "base_force_z_ctrl": force_z,
        "base_roll_ctrl": clamp(
            -attitude_kp * body_roll - attitude_kd * angular_velocity[0],
            -torque_limit_Nm,
            torque_limit_Nm,
        ),
        "base_pitch_ctrl": clamp(
            -attitude_kp * body_pitch - attitude_kd * angular_velocity[1],
            -torque_limit_Nm,
            torque_limit_Nm,
        ),
        "base_yaw_ctrl": clamp(
            attitude_kp * wrap_angle(target_yaw_rad - yaw)
            - attitude_kd * angular_velocity[2],
            -torque_limit_Nm,
            torque_limit_Nm,
        ),
    }


def go2_posture_controls(
    t: float,
    observation: dict,
    gait_amplitude: float = 0.10,
) -> dict[str, float]:
    """Low-amplitude torque-controlled trot around the standard Go2 pose."""
    controls: dict[str, float] = {}
    phase = 2.0 * math.pi * 1.6 * t
    for leg in ("FL", "FR", "RL", "RR"):
        diagonal_phase = phase if leg in ("FL", "RR") else phase + math.pi
        targets = {
            "hip": 0.0,
            "thigh": 0.90 + gait_amplitude * math.sin(diagonal_phase),
            "calf": -1.80 - 1.35 * gait_amplitude * math.sin(diagonal_phase),
        }
        for part in ("hip", "thigh", "calf"):
            joint_name = f"{leg}_{part}_joint"
            actuator_name = f"{leg}_{part}"
            state = observation["joints"][joint_name]
            kp = 24.0 if part != "calf" else 30.0
            kd = 1.1 if part != "calf" else 1.4
            limit = 23.7 if part != "calf" else 45.43
            controls[actuator_name] = clamp(
                kp * (targets[part] - state["position"]) - kd * state["velocity"],
                -limit,
                limit,
            )
    return controls


def rover_wheel_controls(
    t: float,
    observation: dict,
    move_duration_s: float,
    target_speed_mps: float,
    wheel_radius_m: float = 0.12,
) -> dict[str, float]:
    """Track a zero-endpoint-velocity smootherstep wheel-speed profile."""
    names = (
        ("fl_wheel", "fl_wheel_joint"),
        ("fr_wheel", "fr_wheel_joint"),
        ("rl_wheel", "rl_wheel_joint"),
        ("rr_wheel", "rr_wheel_joint"),
    )
    phase = clamp(t / max(move_duration_s, 1e-9), 0.0, 1.0)
    linear_speed = target_speed_mps * smootherstep_rate(phase) if t < move_duration_s else 0.0
    target = linear_speed / wheel_radius_m
    return {
        actuator: clamp(
            target - 0.08 * observation["joints"][joint]["velocity"], -8.0, 8.0
        )
        for actuator, joint in names
    }
