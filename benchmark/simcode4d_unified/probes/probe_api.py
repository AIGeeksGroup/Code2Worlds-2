"""Narrow, benchmark-owned probe interface.

Probe modules receive only copied observations and return actuator commands plus
typed benchmark API requests. They never receive MuJoCo model/data handles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class ProbeCommand:
    controls: dict[str, float] = field(default_factory=dict)
    grasp_request: bool = False
    markers: tuple[str, ...] = ()


class Probe:
    probe_id = "abstract"
    version = "1"

    def reset(self, spec: dict[str, Any], gold: dict[str, Any]) -> None:
        self.spec = spec
        self.gold = gold

    def command(self, t: float, observation: dict[str, Any]) -> ProbeCommand:
        raise NotImplementedError


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quat_wxyz(quat: list[float]) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body(observation: dict[str, Any], name: str) -> dict[str, Any]:
    return observation["bodies"][name]


def joint(observation: dict[str, Any], name: str) -> dict[str, float]:
    return observation["joints"][name]


def time_waypoint(path: list[list[float]], t: float, duration: float, lookahead: int = 5) -> tuple[float, float]:
    if not path:
        raise ValueError("empty probe path")
    index = int(clamp(t / max(duration, 1e-9), 0.0, 1.0) * (len(path) - 1))
    index = min(len(path) - 1, index + lookahead)
    return float(path[index][0]), float(path[index][1])


def differential_velocity_targets(
    current_xy: tuple[float, float],
    current_yaw: float,
    target_xy: tuple[float, float],
    forward_axis: str,
    wheel_radius: float,
    track_width: float,
    maximum_linear_speed: float,
    maximum_yaw_rate: float,
) -> tuple[float, float, float]:
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    distance = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx)
    if forward_axis == "y":
        desired_yaw -= math.pi / 2.0
    yaw_error = wrap_angle(desired_yaw - current_yaw)
    linear = clamp(1.25 * distance, 0.0, maximum_linear_speed)
    linear *= max(0.0, math.cos(yaw_error))
    yaw_rate = clamp(2.2 * yaw_error, -maximum_yaw_rate, maximum_yaw_rate)
    left_linear = linear - yaw_rate * track_width * 0.5
    right_linear = linear + yaw_rate * track_width * 0.5
    sign = 1.0 if forward_axis == "x" else -1.0
    return sign * left_linear / wheel_radius, sign * right_linear / wheel_radius, distance
