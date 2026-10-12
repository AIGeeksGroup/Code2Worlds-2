"""Reference probe for INS-03: replay frozen UR5e joint targets via actuators."""
from probe_api import Probe, ProbeCommand, clamp


class WeldSeamProbe(Probe):
    probe_id = "ins03.ur5e_weld"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.targets = spec["probe"]["reference_joint_positions"]
        self.duration = spec["probe"]["move_duration_s"]
        self.actuators = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

    def command(self, t, observation):
        phase = clamp(t / self.duration, 0.0, 1.0) * (len(self.targets) - 1)
        left = int(phase)
        right = min(len(self.targets) - 1, left + 1)
        blend = phase - left
        target = [
            (1.0 - blend) * self.targets[left][index] + blend * self.targets[right][index]
            for index in range(6)
        ]
        markers = ("seam_complete",) if t >= self.duration else ()
        return ProbeCommand(
            controls=dict(zip(self.actuators, target)),
            markers=markers,
        )


PROBE_CLASS = WeldSeamProbe
