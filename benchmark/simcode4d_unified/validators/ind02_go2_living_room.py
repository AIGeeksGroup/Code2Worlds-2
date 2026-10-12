"""Independent validator for IND-02."""
from common import base_result
from dev15_common import (
    first_axis_progress_time, first_region_time, height_span, lateral_span,
    maximum_tilt, signed_travel, terminal_in_region, terminal_stopped,
)


class IndoorGo2Validator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        travel = signed_travel(frames, "base", 0)
        lateral = lateral_span(frames, "base", 0)
        height = height_span(frames, "base")
        tilt = maximum_tilt(frames, "base")
        progress_t = first_axis_progress_time(frames, "base", 0, threshold["minimum_travel_m"] * 0.5, 1.0)
        goal_t = first_region_time(frames, "base", gold["regions"]["goal"])
        predicates = [
            {"id": "forward_travel", "pass": travel >= threshold["minimum_travel_m"], "value_m": travel, "mandatory": True},
            {"id": "lateral_corridor", "pass": lateral <= threshold["maximum_lateral_span_m"], "value_m": lateral, "mandatory": True},
            {"id": "height_stability", "pass": height <= threshold["maximum_height_span_m"], "value_m": height, "mandatory": True},
            {"id": "tilt_limit", "pass": tilt <= threshold["maximum_tilt_rad"], "value_rad": tilt, "mandatory": True},
            {"id": "goal_hold", "pass": terminal_in_region(frames, "base", threshold["terminal_hold_s"], gold["regions"]["goal"]), "value_s": goal_t, "mandatory": True},
            {"id": "terminal_stop", "pass": terminal_stopped(frames, "base", threshold["terminal_hold_s"], threshold["maximum_terminal_speed_mps"]), "mandatory": True},
            {"id": "event_order", "pass": progress_t is not None and goal_t is not None and progress_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = IndoorGo2Validator
