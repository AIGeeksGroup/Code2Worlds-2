"""Reference probe for URB-02: west-to-east Go2 crosswalk traversal."""
from probe_api import Probe, ProbeCommand
from mobile_control import go2_posture_controls, planar_reference_controls


class IntersectionProbe(Probe):
    probe_id = "urb02.go2_intersection"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]

    def command(self, t, observation):
        duration = self.spec["probe"]["move_duration_s"]
        controls = go2_posture_controls(t, observation, gait_amplitude=0.08)
        controls.update(planar_reference_controls(
            t, observation, "base", self.start, self.end, duration, 0.0,
            165.0, 120.0, 70.0, 800.0, 120.0, 45.0, 8.0,
        ))
        markers = ("crossing_complete",) if t >= duration else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = IntersectionProbe
