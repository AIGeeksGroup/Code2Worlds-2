"""Independent validator for INS-02."""
from common import base_result
from dev15_common import (
    first_axis_progress_time, first_region_time, maximum_tilt, point_distance,
    quaternion_distance, relative_orientation_in_body_frame,
    relative_position_in_body_frame, signed_travel, terminal_in_region, terminal_stopped,
)


class AgvCargoValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        travel = signed_travel(frames, "rover_base", 0)
        initial_relative = relative_position_in_body_frame(frames[0], "rover_base", "cargo")
        initial_orientation = relative_orientation_in_body_frame(frames[0], "rover_base", "cargo")
        relative_errors = [
            point_distance(relative_position_in_body_frame(frame, "rover_base", "cargo"), initial_relative)
            for frame in frames
        ]
        retention = max(relative_errors)
        orientation_retention = max(
            quaternion_distance(relative_orientation_in_body_frame(frame, "rover_base", "cargo"), initial_orientation)
            for frame in frames
        )
        tilt = maximum_tilt(frames, "rover_base")
        progress_t = first_axis_progress_time(frames, "rover_base", 0, threshold["minimum_travel_m"] * 0.5, 1.0)
        goal_t = first_region_time(frames, "cargo", gold["regions"]["cargo_goal"])
        predicates = [
            {"id": "cargo_transport", "pass": travel >= threshold["minimum_travel_m"], "value_m": travel, "mandatory": True},
            {"id": "weld_retention", "pass": retention <= threshold["maximum_relative_pose_error_m"], "value_m": retention, "mandatory": True},
            {"id": "weld_orientation_retention", "pass": orientation_retention <= threshold["maximum_relative_orientation_error_rad"], "value_rad": orientation_retention, "mandatory": True},
            {"id": "tilt_limit", "pass": tilt <= threshold["maximum_tilt_rad"], "value_rad": tilt, "mandatory": True},
            {"id": "cargo_goal_hold", "pass": terminal_in_region(frames, "cargo", threshold["terminal_hold_s"], gold["regions"]["cargo_goal"]), "value_s": goal_t, "mandatory": True},
            {"id": "terminal_stop", "pass": terminal_stopped(frames, "rover_base", threshold["terminal_hold_s"], threshold["maximum_terminal_speed_mps"]), "mandatory": True},
            {"id": "event_order", "pass": progress_t is not None and goal_t is not None and progress_t <= goal_t, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = AgvCargoValidator
