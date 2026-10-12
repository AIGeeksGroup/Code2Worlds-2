"""Independent validator for URB-02, including the previously omitted start check."""
from common import base_result, in_aabb
from dev15_common import (
    first_region_time, height_span, lateral_span, maximum_tilt,
    terminal_in_region, terminal_stopped,
)


class IntersectionValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        start_ok = in_aabb(frames[0]["bodies"]["base"]["position"], gold["regions"]["west_start"])
        road_t = first_region_time(frames, "base", gold["regions"]["road_crossing"])
        goal_t = first_region_time(frames, "base", gold["regions"]["east_sidewalk"])
        lateral = lateral_span(frames, "base", 0)
        height = height_span(frames, "base")
        tilt = maximum_tilt(frames, "base")
        predicates = [
            {"id": "west_sidewalk_start", "pass": start_ok, "mandatory": True},
            {"id": "road_crossing", "pass": road_t is not None, "value_s": road_t, "mandatory": True},
            {"id": "east_sidewalk_hold", "pass": terminal_in_region(frames, "base", threshold["terminal_hold_s"], gold["regions"]["east_sidewalk"]), "value_s": goal_t, "mandatory": True},
            {"id": "crosswalk_corridor", "pass": lateral <= threshold["maximum_lateral_span_m"], "value_m": lateral, "mandatory": True},
            {"id": "height_stability", "pass": height <= threshold["maximum_height_span_m"], "value_m": height, "mandatory": True},
            {"id": "tilt_limit", "pass": tilt <= threshold["maximum_tilt_rad"], "value_rad": tilt, "mandatory": True},
            {"id": "terminal_stop", "pass": terminal_stopped(frames, "base", threshold["terminal_hold_s"], threshold["maximum_terminal_speed_mps"]), "mandatory": True},
            {"id": "event_order", "pass": start_ok and road_t is not None and goal_t is not None and road_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = IntersectionValidator
