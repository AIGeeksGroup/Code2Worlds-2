"""Independent validator for INS-03."""
from common import base_result, body_speed, final_window
from dev15_common import interpolate_points, point_distance


class WeldSeamValidator:
    def validate(self, raw, spec, gold):
        frames = raw["frames"]
        threshold = gold["thresholds"]
        reference = gold["regions"]["seam_reference"]["points_xyz_m"]
        duration = spec["probe"]["move_duration_s"]
        deviations = []
        steps = []
        previous = None
        for frame in frames:
            actual = frame["bodies"]["weld_tool_tip"]["position"]
            target = interpolate_points(reference, frame["t"] / duration)
            deviations.append(point_distance(actual, target))
            if previous is not None:
                steps.append(point_distance(actual, previous))
            previous = actual
        maximum_deviation = max(deviations)
        endpoint_error = point_distance(
            frames[-1]["bodies"]["weld_tool_tip"]["position"], reference[-1]
        )
        window = final_window(frames, threshold["terminal_hold_s"])
        stopped = bool(window) and all(
            body_speed(frame, "weld_tool_tip") <= threshold["maximum_terminal_speed_mps"]
            for frame in window
        )
        start_ok = deviations[0] <= threshold["maximum_path_error_m"]
        end_ok = endpoint_error <= threshold["maximum_path_error_m"]
        predicates = [
            {"id": "seam_tracking", "pass": maximum_deviation <= threshold["maximum_path_error_m"], "value_m": maximum_deviation, "mandatory": True},
            {"id": "endpoint_coverage", "pass": end_ok, "value_m": endpoint_error, "mandatory": True},
            {"id": "continuous_motion", "pass": max(steps or [0.0]) <= threshold["maximum_logged_step_m"], "value_m": max(steps or [0.0]), "mandatory": True},
            {"id": "terminal_stop", "pass": stopped, "mandatory": True},
            {"id": "event_order", "pass": start_ok and end_ok, "mandatory": True},
        ]
        return base_result(raw, predicates)


VALIDATOR_CLASS = WeldSeamValidator
