"""Reference probe for URB-03: approach and be physically blocked by a pole."""
from probe_api import Probe, ProbeCommand
from mobile_control import go2_posture_controls, planar_reference_controls


TARGET_POLE = "prop_convex_small_017_lt_n4_pole"


class PoleBlockingProbe(Probe):
    probe_id = "urb03.go2_pole_blocking"
    version = "1.0.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.start, self.end = spec["probe"]["waypoints_xyz"]
        self.stop = None

    def command(self, t, observation):
        for contact in observation["contacts"]:
            if TARGET_POLE in (contact["geom1"], contact["geom2"]):
                if self.stop is None:
                    position = observation["bodies"]["base"]["position"]
                    self.stop = [position[0], position[1], self.start[2]]
                break
        endpoint = self.stop if self.stop is not None else self.end
        duration = 0.35 if self.stop is not None else self.spec["probe"]["move_duration_s"]
        local_t = 0.0 if self.stop is not None else t
        controls = go2_posture_controls(t, observation, gait_amplitude=0.07)
        controls.update(planar_reference_controls(
            local_t, observation, "base", self.start if self.stop is None else endpoint,
            endpoint, duration, 0.0, 165.0, 120.0, 70.0,
            800.0, 120.0, 45.0, 8.0,
        ))
        markers = ("pole_contact",) if self.stop is not None else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = PoleBlockingProbe
