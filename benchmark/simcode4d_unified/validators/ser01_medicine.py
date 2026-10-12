"""Independent validator for SER-01."""
from __future__ import annotations

from common import base_result, body_speed, final_window, first_time, in_aabb, planar_distance


class MedicineValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        start = frames[0]["bodies"]["task_item"]["position"]
        grasp_t = next((event["t"] for event in raw["events"] if event["event"] == "grasp_activated"), None)
        goal = gold["regions"]["goal"]
        goal_t = first_time(frames, lambda f: in_aabb(f["bodies"]["task_item"]["position"], goal))
        window = final_window(frames, gold["thresholds"]["terminal_hold_s"])
        goal_hold = bool(window) and all(in_aabb(frame["bodies"]["task_item"]["position"], goal) for frame in window)
        settled = bool(window) and all(body_speed(frame, "task_item") <= gold["thresholds"]["maximum_terminal_speed_mps"] for frame in window)
        displacement = planar_distance(start, frames[-1]["bodies"]["task_item"]["position"])
        predicates = [
            {"id": "legal_grasp", "pass": grasp_t is not None, "value_s": grasp_t, "mandatory": True},
            {"id": "transport_distance", "pass": displacement >= gold["thresholds"]["minimum_transport_m"], "value_m": displacement, "mandatory": True},
            {"id": "goal_hold", "pass": goal_hold, "value_s": goal_t, "mandatory": True},
            {"id": "terminal_settled", "pass": settled, "mandatory": True},
            {"id": "grasp_before_goal", "pass": grasp_t is not None and goal_t is not None and grasp_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = MedicineValidator
