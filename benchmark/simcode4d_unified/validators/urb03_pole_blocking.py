"""Independent validator for URB-03 using actual target-pole contact."""
from common import base_result, semantic_contact_matches
from dev15_common import (
    first_axis_progress_time, lateral_span, maximum_tilt, signed_travel,
    terminal_stopped,
)


class PoleBlockingValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        contact_t = next(
            (
                float(event["t"])
                for event in raw["events"]
                if event["event"] == "semantic_contact"
                and event.get("rule") == "go2_all/target_pole"
            ),
            None,
        )
        if contact_t is None:
            for frame in frames:
                if semantic_contact_matches(frame, gold["contact_policy"], "go2_all", "target_pole"):
                    contact_t = frame["t"]
        approach_t = first_axis_progress_time(frames, "base", 0, threshold["minimum_approach_m"], 1.0)
        travel = signed_travel(frames, "base", 0)
        max_x = max(frame["bodies"]["base"]["position"][0] for frame in frames)
        lateral = lateral_span(frames, "base", 0)
        tilt = maximum_tilt(frames, "base")
        predicates = [
            {"id": "approach_distance", "pass": travel >= threshold["minimum_approach_m"], "value_m": travel, "mandatory": True},
            {"id": "target_pole_contact", "pass": contact_t is not None, "value_s": contact_t, "mandatory": True},
            {"id": "no_pass_through", "pass": max_x <= threshold["maximum_base_x_m"], "value_m": max_x, "mandatory": True},
            {"id": "sidewalk_corridor", "pass": lateral <= threshold["maximum_lateral_span_m"], "value_m": lateral, "mandatory": True},
            {"id": "tilt_limit", "pass": tilt <= threshold["maximum_tilt_rad"], "value_rad": tilt, "mandatory": True},
            {"id": "terminal_stop", "pass": terminal_stopped(frames, "base", threshold["terminal_hold_s"], threshold["maximum_terminal_speed_mps"]), "mandatory": True},
            {"id": "event_order", "pass": approach_t is not None and contact_t is not None and approach_t <= contact_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = PoleBlockingValidator
