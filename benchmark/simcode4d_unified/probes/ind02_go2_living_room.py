"""Reference probe for IND-02: actuator-only Go2 living-room traversal."""
from probe_api import Probe, ProbeCommand
from mobile_control import go2_posture_controls, planar_reference_controls


class IndoorGo2TraversalProbe(Probe):
    probe_id = "ind02.go2_living_room"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]

    def command(self, t, observation):
        controls = go2_posture_controls(t, observation, gait_amplitude=0.10)
        controls.update(planar_reference_controls(
            t, observation, "base", self.start, self.end,
            self.spec["probe"]["move_duration_s"], 0.0,
            165.0, 120.0, 70.0, 800.0, 100.0, 45.0, 8.0,
        ))
        markers = ("route_complete",) if t >= self.spec["probe"]["move_duration_s"] else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = IndoorGo2TraversalProbe
