"""Independent validator for SER-03."""
from __future__ import annotations

from common import base_result, body_speed, final_window, first_time, in_aabb


class HotelDrawerValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        goal = gold["regions"]["robot_goal"]
        arrival_t = first_time(frames, lambda f: in_aabb(f["bodies"]["robot"]["position"], goal))
        open_t = first_time(frames, lambda f: f["joints"]["drawer_joint"]["position"] >= gold["thresholds"]["minimum_drawer_extension_m"])
        window = final_window(frames, gold["thresholds"]["terminal_hold_s"])
        drawer_hold = bool(window) and all(frame["joints"]["drawer_joint"]["position"] >= gold["thresholds"]["minimum_drawer_extension_m"] for frame in window)
        parcel_supported = bool(window) and all(
            frame["bodies"]["meal_parcel"]["position"][2] >= gold["thresholds"]["minimum_parcel_height_m"]
            and abs(frame["bodies"]["meal_parcel"]["position"][2] - frame["bodies"]["upper_drawer"]["position"][2]) <= gold["thresholds"]["maximum_parcel_drawer_z_offset_m"]
            for frame in window
        )
        stopped = bool(window) and all(body_speed(frame, "robot") <= gold["thresholds"]["maximum_terminal_speed_mps"] for frame in window)
        predicates = [
            {"id": "robot_goal_arrival", "pass": arrival_t is not None, "value_s": arrival_t, "mandatory": True},
            {"id": "drawer_open", "pass": open_t is not None, "value_s": open_t, "mandatory": True},
            {"id": "arrival_before_open", "pass": arrival_t is not None and open_t is not None and arrival_t <= open_t, "mandatory": True},
            {"id": "drawer_terminal_hold", "pass": drawer_hold, "mandatory": True},
            {"id": "parcel_supported", "pass": parcel_supported, "mandatory": True},
            {"id": "terminal_stop", "pass": stopped, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = HotelDrawerValidator
