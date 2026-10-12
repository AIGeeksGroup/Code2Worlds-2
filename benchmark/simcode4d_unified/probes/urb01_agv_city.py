"""Reference probe for URB-01: AGV traversal along the authored pavement."""
from probe_api import Probe, ProbeCommand
from mobile_control import planar_reference_controls, rover_wheel_controls


class UrbanAgvProbe(Probe):
    probe_id = "urb01.agv_city"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]

    def command(self, t, observation):
        duration = self.spec["probe"]["move_duration_s"]
        controls = rover_wheel_controls(t, observation, duration, abs(self.end[0] - self.start[0]) / duration)
        controls.update(planar_reference_controls(
            t, observation, "rover_base", self.start, self.end, duration, 0.0,
            150.0, 100.0, 65.0, 75.0, 42.0, 55.0, 8.0,
        ))
        markers = ("pavement_complete",) if t >= duration else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = UrbanAgvProbe
