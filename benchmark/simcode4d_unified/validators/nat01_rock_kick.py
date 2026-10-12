"""Independent validator for NAT-01."""
from __future__ import annotations

from common import base_result, body_speed, final_window, first_time, planar_distance, semantic_contact_matches


class RockKickValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        start = frames[0]["bodies"]["stone"]["position"]
        end = frames[-1]["bodies"]["stone"]["position"]
        displacement = planar_distance(start, end)
        contact_t = next((event["t"] for event in raw["events"] if event["event"] == "semantic_contact" and event["rule"] == "go2_feet/rock"), None)
        if contact_t is None:
            contact_t = first_time(frames, lambda f: semantic_contact_matches(f, gold["contact_policy"], "go2_feet", "rock"))
        move_t = first_time(frames, lambda f: planar_distance(start, f["bodies"]["stone"]["position"]) >= gold["thresholds"]["minimum_planar_displacement_m"])
        window = final_window(frames, gold["thresholds"]["terminal_hold_s"])
        settled = bool(window) and all(body_speed(frame, "stone") <= gold["thresholds"]["maximum_terminal_speed_mps"] for frame in window)
        order = contact_t is not None and move_t is not None and contact_t <= move_t
        predicates = [
            {"id": "foot_stone_contact", "pass": contact_t is not None, "value_s": contact_t, "mandatory": True},
            {"id": "planar_displacement", "pass": displacement >= gold["thresholds"]["minimum_planar_displacement_m"], "value_m": displacement, "mandatory": True},
            {"id": "contact_before_motion", "pass": order, "mandatory": True},
            {"id": "terminal_settled", "pass": settled, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = RockKickValidator
