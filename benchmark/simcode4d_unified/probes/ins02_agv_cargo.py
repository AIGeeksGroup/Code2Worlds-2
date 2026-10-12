"""Reference probe for INS-02: transport an equality-secured cargo box."""
from probe_api import Probe, ProbeCommand
from mobile_control import planar_reference_controls, rover_wheel_controls


class AgvCargoProbe(Probe):
    probe_id = "ins02.agv_cargo"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]

    def command(self, t, observation):
        duration = self.spec["probe"]["move_duration_s"]
        controls = rover_wheel_controls(t, observation, duration, abs(self.end[0] - self.start[0]) / duration)
        controls.update(planar_reference_controls(
            t, observation, "rover_base", self.start, self.end, duration, 0.0,
            170.0, 110.0, 120.0, 180.0, 70.0, 110.0, 18.0,
        ))
        markers = ("cargo_transport_complete",) if t >= duration else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = AgvCargoProbe
