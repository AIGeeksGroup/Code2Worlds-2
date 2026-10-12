"""Read-only metric helpers for the nine Dev15 handover validators."""
from __future__ import annotations

import math

from common import body_speed, final_window, in_aabb, norm, roll_pitch


def planar_travel(frames, body_name: str) -> float:
    first = frames[0]["bodies"][body_name]["position"]
    last = frames[-1]["bodies"][body_name]["position"]
    return math.hypot(last[0] - first[0], last[1] - first[1])


def signed_travel(frames, body_name: str, axis: int) -> float:
    first = frames[0]["bodies"][body_name]["position"][axis]
    last = frames[-1]["bodies"][body_name]["position"][axis]
    return last - first


def lateral_span(frames, body_name: str, travel_axis: int) -> float:
    other = 1 - travel_axis
    values = [frame["bodies"][body_name]["position"][other] for frame in frames]
    return max(values) - min(values)


def height_span(frames, body_name: str) -> float:
    values = [frame["bodies"][body_name]["position"][2] for frame in frames]
    return max(values) - min(values)


def maximum_tilt(frames, body_name: str) -> float:
    return max(
        max(abs(value) for value in roll_pitch(frame["bodies"][body_name]["quaternion"]))
        for frame in frames
    )


def terminal_stopped(frames, body_name: str, hold_s: float, threshold: float) -> bool:
    window = final_window(frames, hold_s)
    return bool(window) and all(body_speed(frame, body_name) <= threshold for frame in window)


def terminal_in_region(frames, body_name: str, hold_s: float, region: dict) -> bool:
    window = final_window(frames, hold_s)
    return bool(window) and all(
        in_aabb(frame["bodies"][body_name]["position"], region) for frame in window
    )


def first_region_time(frames, body_name: str, region: dict):
    for frame in frames:
        if in_aabb(frame["bodies"][body_name]["position"], region):
            return frame["t"]
    return None


def first_axis_progress_time(frames, body_name: str, axis: int, threshold: float, sign: float):
    start = frames[0]["bodies"][body_name]["position"][axis]
    for frame in frames:
        progress = sign * (frame["bodies"][body_name]["position"][axis] - start)
        if progress >= threshold:
            return frame["t"]
    return None


def interpolate_points(points: list[list[float]], phase: float) -> list[float]:
    phase = max(0.0, min(1.0, phase)) * (len(points) - 1)
    left = int(phase)
    right = min(len(points) - 1, left + 1)
    blend = phase - left
    return [
        (1.0 - blend) * points[left][index] + blend * points[right][index]
        for index in range(3)
    ]


def relative_position_in_body_frame(frame, body_name: str, target_name: str) -> list[float]:
    body = frame["bodies"][body_name]
    target = frame["bodies"][target_name]
    vx, vy, vz = [target["position"][i] - body["position"][i] for i in range(3)]
    w, x, y, z = body["quaternion"]
    return [
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + z * w) * vy + 2 * (x * z - y * w) * vz,
        2 * (x * y - z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + x * w) * vz,
        2 * (x * z + y * w) * vx + 2 * (y * z - x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ]


def relative_orientation_in_body_frame(frame, body_name: str, target_name: str) -> list[float]:
    aw, ax, ay, az = frame["bodies"][body_name]["quaternion"]
    bw, bx, by, bz = frame["bodies"][target_name]["quaternion"]
    relative = [
        aw * bw + ax * bx + ay * by + az * bz,
        aw * bx - ax * bw - ay * bz + az * by,
        aw * by + ax * bz - ay * bw - az * bx,
        aw * bz - ax * by + ay * bx - az * bw,
    ]
    magnitude = norm(relative)
    return [value / magnitude for value in relative]


def quaternion_distance(a, b) -> float:
    dot = abs(sum(float(a[index]) * float(b[index]) for index in range(4)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def point_distance(a, b) -> float:
    return norm([a[index] - b[index] for index in range(3)])
