"""Independent validator for NAT-02."""
from __future__ import annotations

import math

from common import base_result, body_speed, final_window, first_time, in_aabb, roll_pitch


class HillsTurnStopValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        regions = gold["regions"]
        hill_t = first_time(frames, lambda f: in_aabb(f["bodies"]["jackal_base"]["position"], regions["hill_checkpoint"]))
        goal_t = first_time(frames, lambda f: in_aabb(f["bodies"]["jackal_base"]["position"], regions["goal"]))
        initial_q = frames[0]["bodies"]["jackal_base"]["quaternion"]
        final_q = frames[-1]["bodies"]["jackal_base"]["quaternion"]
        initial_yaw = math.atan2(2 * (initial_q[0] * initial_q[3] + initial_q[1] * initial_q[2]), 1 - 2 * (initial_q[2] ** 2 + initial_q[3] ** 2))
        final_yaw = math.atan2(2 * (final_q[0] * final_q[3] + final_q[1] * final_q[2]), 1 - 2 * (final_q[2] ** 2 + final_q[3] ** 2))
        yaw_change = abs(math.atan2(math.sin(final_yaw - initial_yaw), math.cos(final_yaw - initial_yaw)))
        maximum_tilt = max(max(abs(v) for v in roll_pitch(frame["bodies"]["jackal_base"]["quaternion"])) for frame in frames)
        window = final_window(frames, gold["thresholds"]["terminal_hold_s"])
        stopped = bool(window) and all(body_speed(frame, "jackal_base") <= gold["thresholds"]["maximum_terminal_speed_mps"] for frame in window)
        goal_hold = bool(window) and all(in_aabb(frame["bodies"]["jackal_base"]["position"], regions["goal"]) for frame in window)
        predicates = [
            {"id": "hill_checkpoint", "pass": hill_t is not None, "value_s": hill_t, "mandatory": True},
            {"id": "left_turn", "pass": yaw_change >= gold["thresholds"]["minimum_heading_change_rad"], "value_rad": yaw_change, "mandatory": True},
            {"id": "goal_hold", "pass": goal_hold, "value_s": goal_t, "mandatory": True},
            {"id": "terminal_stop", "pass": stopped, "mandatory": True},
            {"id": "tilt_limit", "pass": maximum_tilt <= gold["thresholds"]["maximum_tilt_rad"], "value_rad": maximum_tilt, "mandatory": True},
            {"id": "event_order", "pass": hill_t is not None and goal_t is not None and hill_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = HillsTurnStopValidator
