"""Benchmark probe for NAT-01: actuator-only Go2 rock kick."""
from __future__ import annotations

from base_control import smootherstep
from probe_api import Probe, ProbeCommand, clamp, joint


ACTUATORS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
JOINTS = [name + "_joint" for name in ACTUATORS]
HOME = [0.0, 0.9, -1.8] * 4
PRE_STRIKE = [-0.25, 1.10, -2.05]
FOLLOW_THROUGH = [0.50, 0.20, -1.55]


def blend(left, right, value):
    weight = smootherstep(value)
    return [a + weight * (b - a) for a, b in zip(left, right)]


class RockKickProbe(Probe):
    probe_id = "nat01.actuator_only_rock_kick"
    version = "2.3.0"

    def command(self, t, observation):
        target = list(HOME)
        if 0.55 <= t < 0.77:
            target[:3] = blend(HOME[:3], PRE_STRIKE, (t - 0.55) / 0.22)
        elif 0.77 <= t < 0.93:
            target[:3] = PRE_STRIKE
        elif 0.93 <= t < 1.08:
            target[:3] = blend(PRE_STRIKE, FOLLOW_THROUGH, (t - 0.93) / 0.15)
        elif 1.08 <= t < 1.25:
            target[:3] = FOLLOW_THROUGH
        elif 1.25 <= t < 1.60:
            target[:3] = blend(FOLLOW_THROUGH, HOME[:3], (t - 1.25) / 0.35)
        controls = {}
        for index, (actuator, joint_name) in enumerate(zip(ACTUATORS, JOINTS)):
            state = joint(observation, joint_name)
            kp = 52.0 if "calf" not in actuator else 62.0
            kd = 2.4
            limit = 23.7 if "calf" not in actuator else 45.43
            torque = kp * (target[index] - state["position"]) - kd * state["velocity"]
            controls[actuator] = clamp(torque, -limit, limit)
        marker = ("strike",) if 0.93 <= t < 1.25 else ()
        return ProbeCommand(controls=controls, markers=marker)


PROBE_CLASS = RockKickProbe
