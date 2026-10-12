"""Benchmark probe for SER-02: legal actuator-only luggage delivery."""
from __future__ import annotations

from base_control import smootherstep, tracked_base_controls
from probe_api import Probe, ProbeCommand, body


class AirportLuggageProbe(Probe):
    probe_id = "ser02.airport_luggage_delivery"
    version = "2.3.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.path = spec["probe"]["waypoints_xy"]
        self.yaws = spec["probe"]["reference_yaw_rad"]

    def command(self, t, observation):
        controls = tracked_base_controls(
            t, observation, self.path, self.yaws, "robot",
            trajectory_duration=6.9, wheel_radius=0.18, track_width=0.78,
            wheel_limit=20.0, maximum_linear_speed=4.2, maximum_yaw_rate=3.2,
            base_force_limit=900.0, assist_limit=420.0,
            yaw_kp=350.0, yaw_kd=60.0, yaw_limit=520.0,
        )
        lift = 0.03 + 0.30 * smootherstep((t - 1.60) / 0.80)
        lift -= 0.03 * smootherstep((t - 3.10) / 0.60)
        controls["lift_ctrl"] = lift
        item = body(observation, "task_item")["position"]
        grasp = body(observation, "grasp_frame")["position"]
        distance = sum((a - b) ** 2 for a, b in zip(item, grasp)) ** 0.5
        return ProbeCommand(
            controls=controls,
            grasp_request=t >= 0.80 and distance <= 0.06,
        )


PROBE_CLASS = AirportLuggageProbe
