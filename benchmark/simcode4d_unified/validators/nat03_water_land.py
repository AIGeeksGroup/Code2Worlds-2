"""Independent validator for NAT-03."""
from __future__ import annotations

from common import base_result, body_speed, final_window, first_time, in_aabb, semantic_contact_matches


class WaterLandValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        water_t = first_time(frames, lambda f: f["field"].get("submerged_fraction", 0.0) >= gold["thresholds"]["minimum_water_submergence"])
        bank_t = next(
            (event["t"] for event in raw["events"] if event["event"] == "bank_contact"), None
        )
        if bank_t is None:
            bank_t = first_time(frames, lambda f: semantic_contact_matches(f, gold["contact_policy"], "rover_wheels", "shore"))
        dry_t = first_time(frames, lambda f: f["field"].get("submerged_fraction", 1.0) <= gold["thresholds"]["maximum_terminal_submergence"])
        window = final_window(frames, gold["thresholds"]["terminal_hold_s"])
        goal = gold["regions"]["goal"]
        goal_hold = bool(window) and all(in_aabb(frame["bodies"]["robot"]["position"], goal) for frame in window)
        stopped = bool(window) and all(body_speed(frame, "robot") <= gold["thresholds"]["maximum_terminal_speed_mps"] for frame in window)
        order = water_t is not None and bank_t is not None and dry_t is not None and water_t <= bank_t <= dry_t
        predicates = [
            {"id": "water_state", "pass": water_t is not None, "value_s": water_t, "mandatory": True},
            {"id": "bank_contact", "pass": bank_t is not None, "value_s": bank_t, "mandatory": True},
            {"id": "dry_land_state", "pass": dry_t is not None, "value_s": dry_t, "mandatory": True},
            {"id": "event_order", "pass": order, "mandatory": True},
            {"id": "goal_hold", "pass": goal_hold, "mandatory": True},
            {"id": "terminal_stop", "pass": stopped, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = WaterLandValidator
