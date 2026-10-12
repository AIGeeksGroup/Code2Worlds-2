"""Benchmark probe for SER-01: legal actuator-only medicine delivery."""
from __future__ import annotations

from base_control import smootherstep, tracked_base_controls
from probe_api import Probe, ProbeCommand, body


class MedicineProbe(Probe):
    probe_id = "ser01.medicine_delivery"
    version = "2.5.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.path = spec["probe"]["waypoints_xy"]
        self.yaws = spec["probe"]["reference_yaw_rad"]

    def command(self, t, observation):
        controls = tracked_base_controls(
            t, observation, self.path, self.yaws, "robot",
            trajectory_duration=6.8, wheel_radius=0.20, track_width=0.78,
            wheel_limit=18.0, maximum_linear_speed=2.5, maximum_yaw_rate=2.4,
            base_force_limit=900.0, assist_limit=900.0,
            yaw_kp=350.0, yaw_kd=70.0, yaw_limit=520.0,
            planar_position_kp=300.0, planar_velocity_kd=400.0,
        )
        lift = 0.06 + 0.35 * smootherstep((t - 2.95) / 0.75)
        lift -= 0.015 * smootherstep((t - 4.25) / 0.45)
        controls["lift_ctrl"] = lift
        if t < 3.95:
            arm = 0.04 + 0.40 * smootherstep((t - 3.42) / 0.48)
        elif t < 4.30:
            arm = 0.04 + 0.40 * (1.0 - smootherstep((t - 3.95) / 0.35))
        elif t < 6.80:
            arm = 0.04
        else:
            arm = 0.04 + 0.40 * smootherstep((t - 6.80) / 0.25)
        controls["arm_ctrl"] = arm
        item = body(observation, "task_item")["position"]
        grasp = body(observation, "grasp_frame")["position"]
        distance = sum((a - b) ** 2 for a, b in zip(item, grasp)) ** 0.5
        return ProbeCommand(
            controls=controls,
            grasp_request=t >= 3.78 and distance <= 0.16,
        )


PROBE_CLASS = MedicineProbe
