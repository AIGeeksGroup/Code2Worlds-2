"""Independent validator for INS-01."""
from common import base_result
from dev15_common import (
    first_axis_progress_time, first_region_time, lateral_span, maximum_tilt,
    signed_travel, terminal_in_region, terminal_stopped,
)


class AgvAisleValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        travel = signed_travel(frames, "rover_base", 0)
        lateral = lateral_span(frames, "rover_base", 0)
        tilt = maximum_tilt(frames, "rover_base")
        progress_t = first_axis_progress_time(frames, "rover_base", 0, threshold["minimum_travel_m"] * 0.5, 1.0)
        goal_t = first_region_time(frames, "rover_base", gold["regions"]["goal"])
        predicates = [
            {"id": "aisle_travel", "pass": travel >= threshold["minimum_travel_m"], "value_m": travel, "mandatory": True},
            {"id": "lane_hold", "pass": lateral <= threshold["maximum_lateral_span_m"], "value_m": lateral, "mandatory": True},
            {"id": "tilt_limit", "pass": tilt <= threshold["maximum_tilt_rad"], "value_rad": tilt, "mandatory": True},
            {"id": "goal_hold", "pass": terminal_in_region(frames, "rover_base", threshold["terminal_hold_s"], gold["regions"]["goal"]), "value_s": goal_t, "mandatory": True},
            {"id": "terminal_stop", "pass": terminal_stopped(frames, "rover_base", threshold["terminal_hold_s"], threshold["maximum_terminal_speed_mps"]), "mandatory": True},
            {"id": "event_order", "pass": progress_t is not None and goal_t is not None and progress_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = AgvAisleValidator
