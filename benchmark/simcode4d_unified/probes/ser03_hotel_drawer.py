"""Benchmark probe for SER-03: legal actuator-only hotel delivery."""
from __future__ import annotations

from base_control import smootherstep, tracked_base_controls
from probe_api import Probe, ProbeCommand, body


class HotelDrawerProbe(Probe):
    probe_id = "ser03.hotel_payload_drawer"
    version = "2.3.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.path = spec["probe"]["waypoints_xy"]
        self.yaws = spec["probe"]["reference_yaw_rad"]

    def command(self, t, observation):
        controls = tracked_base_controls(
            t, observation, self.path, self.yaws, "robot",
            trajectory_duration=6.8, wheel_radius=0.16, track_width=0.68,
            wheel_limit=20.0, maximum_linear_speed=4.2, maximum_yaw_rate=3.2,
            base_force_limit=900.0, assist_limit=420.0,
            yaw_kp=350.0, yaw_kd=60.0, yaw_limit=520.0,
        )
        position = body(observation, "robot")["position"]
        at_terminal = 5.05 <= position[0] <= 6.25 and 4.0 <= position[1] <= 5.2
        drawer = 0.06 + 0.23 * smootherstep((t - 6.85) / 0.35) if at_terminal else 0.06
        controls["drawer_ctrl"] = drawer
        markers = ("drawer_commanded",) if drawer > 0.07 else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = HotelDrawerProbe
