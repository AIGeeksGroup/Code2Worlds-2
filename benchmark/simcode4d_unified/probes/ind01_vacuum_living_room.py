"""Reference probe for IND-01: one eastward living-room sweep."""
from probe_api import Probe, ProbeCommand
from mobile_control import planar_reference_controls


class IndoorVacuumSweepProbe(Probe):
    probe_id = "ind01.vacuum_living_room"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]

    def command(self, t, observation):
        controls = planar_reference_controls(
            t, observation, "robot", self.start, self.end,
            self.spec["probe"]["move_duration_s"], 0.0,
            120.0, 60.0, 28.0, 500.0, 80.0, 20.0, 5.0,
        )
        markers = ("sweep_complete",) if t >= self.spec["probe"]["move_duration_s"] else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = IndoorVacuumSweepProbe
