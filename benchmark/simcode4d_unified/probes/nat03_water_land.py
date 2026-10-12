"""Benchmark probe for NAT-03: declared thrusters plus wheel actuators."""
from __future__ import annotations

from probe_api import Probe, ProbeCommand, body, clamp, joint


WHEELS = ("fl", "fr", "rl", "rr")


class WaterLandProbe(Probe):
    probe_id = "nat03.amphibious_water_land"
    version = "2.3.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.bank_contacted = False
        self.stop_target = None

    def command(self, t, observation):
        robot = body(observation, "robot")
        x = robot["position"][0]
        vx = robot["linear_velocity"][0]
        stop_x = 3.35
        controls = {}
        if self.stop_target is None and x >= 3.25:
            self.stop_target = stop_x
        if self.stop_target is None:
            thrust = clamp(650.0 * (1.55 - vx), -1600.0, 1600.0)
        else:
            thrust = clamp(2000.0 * (self.stop_target - x) - 1600.0 * vx, -2800.0, 2800.0)
        controls["propeller_x_ctrl"] = thrust
        controls["heave_ctrl"] = clamp(-600.0 * robot["linear_velocity"][2], -1300.0, 1300.0)
        pitch = joint(observation, "shore_pitch")
        controls["pitch_ctrl"] = clamp(-650.0 * pitch["position"] - 160.0 * pitch["velocity"], -900.0, 900.0)
        wheel_target = 5.5 if self.stop_target is None else 0.0
        for name in WHEELS:
            controls["steer_" + name + "_ctrl"] = 0.0
            controls["wheel_" + name + "_ctrl"] = wheel_target
        for contact in observation["contacts"]:
            names = (contact["geom1"], contact["geom2"])
            if "shore_hfield" in names and any(name.startswith("wheel_") for name in names):
                self.bank_contacted = True
                break
        markers = ("bank_contact",) if self.bank_contacted else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = WaterLandProbe
