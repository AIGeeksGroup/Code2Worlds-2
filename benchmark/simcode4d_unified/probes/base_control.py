"""Pure-observation controllers for benchmark-owned mobile-base probes."""
from __future__ import annotations

import math

from probe_api import clamp, wrap_angle, yaw_from_quat_wxyz


def smootherstep(value: float) -> float:
    u = clamp(value, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def roll_pitch_from_quat(quaternion: list[float]) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def tracked_base_controls(
    t: float,
    observation: dict,
    path: list[list[float]],
    yaw_path: list[float],
    body_name: str,
    trajectory_duration: float,
    wheel_radius: float,
    track_width: float,
    wheel_limit: float,
    maximum_linear_speed: float,
    maximum_yaw_rate: float,
    base_force_limit: float,
    assist_limit: float,
    yaw_kp: float,
    yaw_kd: float,
    yaw_limit: float,
    planar_position_kp: float = 0.0,
    planar_velocity_kd: float = 0.0,
) -> dict[str, float]:
    if len(path) < 2 or len(path) != len(yaw_path):
        raise ValueError("reference path/yaw mismatch")
    count = len(path)
    sample_dt = trajectory_duration / (count - 1)
    phase = clamp(t / trajectory_duration, 0.0, 1.0) * (count - 1)
    centre = min(count - 1, int(round(phase)))
    target_index = min(count - 1, int(phase) + 5)
    left_index = max(0, centre - 1)
    right_index = min(count - 1, centre + 1)
    delta_t = max(sample_dt * (right_index - left_index), 1e-9)
    target_velocity = [
        (path[right_index][axis] - path[left_index][axis]) / delta_t
        for axis in range(2)
    ]
    target_yaw_rate = wrap_angle(
        yaw_path[right_index] - yaw_path[left_index]
    ) / delta_t
    if t >= trajectory_duration:
        target_velocity = [0.0, 0.0]
        target_yaw_rate = 0.0

    state = observation["bodies"][body_name]
    position = state["position"]
    velocity = state["linear_velocity"]
    angular_velocity = state["angular_velocity"]
    yaw = yaw_from_quat_wxyz(state["quaternion"])
    target_yaw = yaw_path[target_index]
    heading = (-math.sin(yaw), math.cos(yaw))
    lateral = (math.cos(yaw), math.sin(yaw))
    reference_heading = (-math.sin(target_yaw), math.cos(target_yaw))
    error = (
        path[target_index][0] - position[0],
        path[target_index][1] - position[1],
    )
    forward_error = error[0] * heading[0] + error[1] * heading[1]
    lateral_error = error[0] * lateral[0] + error[1] * lateral[1]
    feedforward = (
        target_velocity[0] * reference_heading[0]
        + target_velocity[1] * reference_heading[1]
    )
    actual_forward = velocity[0] * heading[0] + velocity[1] * heading[1]
    linear = clamp(
        feedforward + 2.5 * forward_error + 0.75 * (feedforward - actual_forward),
        -maximum_linear_speed,
        maximum_linear_speed,
    )
    yaw_error = wrap_angle(target_yaw - yaw)
    yaw_rate = clamp(
        target_yaw_rate + 5.5 * yaw_error - 3.2 * lateral_error,
        -maximum_yaw_rate,
        maximum_yaw_rate,
    )
    left_speed = -(linear - yaw_rate * track_width * 0.5) / wheel_radius
    right_speed = -(linear + yaw_rate * track_width * 0.5) / wheel_radius

    reference_speed = math.hypot(*target_velocity)
    if reference_speed < 0.18 or t >= trajectory_duration:
        force_x = 1800.0 * error[0] - 480.0 * velocity[0]
        force_y = 1800.0 * error[1] - 480.0 * velocity[1]
        magnitude = math.hypot(force_x, force_y)
        if magnitude > base_force_limit:
            force_x *= base_force_limit / magnitude
            force_y *= base_force_limit / magnitude
    else:
        if planar_position_kp > 0.0:
            force_x = planar_position_kp * error[0] + planar_velocity_kd * (target_velocity[0] - velocity[0])
            force_y = planar_position_kp * error[1] + planar_velocity_kd * (target_velocity[1] - velocity[1])
            magnitude = math.hypot(force_x, force_y)
            if magnitude > assist_limit:
                force_x *= assist_limit / magnitude
                force_y *= assist_limit / magnitude
        else:
            assist = clamp(
                160.0 * (feedforward - actual_forward) + 80.0 * forward_error,
                -assist_limit,
                assist_limit,
            )
            force_x = heading[0] * assist
            force_y = heading[1] * assist

    roll, pitch = roll_pitch_from_quat(state["quaternion"])
    roll_torque = clamp(-8000.0 * roll - 900.0 * angular_velocity[0], -5500.0, 5500.0)
    pitch_torque = clamp(-8000.0 * pitch - 900.0 * angular_velocity[1], -5500.0, 5500.0)
    yaw_torque = clamp(
        yaw_kp * yaw_error + yaw_kd * (yaw_rate - angular_velocity[2]),
        -yaw_limit,
        yaw_limit,
    )
    return {
        "wheel_left_ctrl": clamp(left_speed, -wheel_limit, wheel_limit),
        "wheel_right_ctrl": clamp(right_speed, -wheel_limit, wheel_limit),
        "base_force_x_ctrl": force_x,
        "base_force_y_ctrl": force_y,
        "base_roll_ctrl": roll_torque,
        "base_pitch_ctrl": pitch_torque,
        "base_yaw_ctrl": yaw_torque,
    }
