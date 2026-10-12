"""Benchmark probe for NAT-02: actuator-only hill traversal, left turn, stop."""
from __future__ import annotations

import math

from base_control import roll_pitch_from_quat, smootherstep
from probe_api import Probe, ProbeCommand, body, clamp, joint, wrap_angle, yaw_from_quat_wxyz


LEFT_ACTUATORS = ("jackal_front_left_motor", "jackal_rear_left_motor")
RIGHT_ACTUATORS = ("jackal_front_right_motor", "jackal_rear_right_motor")
LEFT_JOINTS = ("jackal_front_left_joint", "jackal_rear_left_joint")
RIGHT_JOINTS = ("jackal_front_right_joint", "jackal_rear_right_joint")


class HillsTurnStopProbe(Probe):
    probe_id = "nat02.jackal_hills_turn_stop"
    version = "2.9.0"

    def reset(self, spec, gold):
        super().reset(spec, gold)
        self.path = spec["probe"]["waypoints_xy"]
        self.goal = gold["regions"]["goal"]
        self.stop_target = None
        self.stop_yaw = None

    def command(self, t, observation):
        state = body(observation, "jackal_base")
        position = state["position"]
        velocity = state["linear_velocity"]
        angular = state["angular_velocity"]
        yaw = yaw_from_quat_wxyz(state["quaternion"])
        index = min(len(self.path) - 1, int(t * 30.0) + 7)
        left_index = max(0, index - 1)
        right_index = min(len(self.path) - 1, index + 1)
        dt = max((right_index - left_index) / 30.0, 1e-9)
        target_velocity = [
            (self.path[right_index][axis] - self.path[left_index][axis]) / dt
            for axis in range(2)
        ]
        target = self.path[index]
        error = (target[0] - position[0], target[1] - position[1])
        if t >= 5.0:
            target = list(self.path[-1])
            if t >= 6.2:
                if self.stop_target is None:
                    lower = self.goal["min_xyz_m"]
                    upper = self.goal["max_xyz_m"]
                    margin = 0.18
                    self.stop_target = [
                        clamp(position[0], lower[0] + margin, upper[0] - margin),
                        clamp(position[1], lower[1] + margin, upper[1] - margin),
                    ]
                    self.stop_yaw = yaw
                target = self.stop_target
            target_velocity = [0.0, 0.0]
            error = (target[0] - position[0], target[1] - position[1])
        if t < 3.0:
            target_yaw = -math.pi
            target_yaw_rate = 0.0
        elif t < 4.5:
            target_yaw = -math.pi + math.pi / 2.0 * smootherstep((t - 3.0) / 1.5)
            target_yaw_rate = math.pi / 3.0
        else:
            target_yaw = -math.pi / 2.0
            target_yaw_rate = 0.0
        if self.stop_yaw is not None:
            target_yaw = self.stop_yaw
            target_yaw_rate = 0.0
        heading = (math.cos(yaw), math.sin(yaw))
        target_heading = (math.cos(target_yaw), math.sin(target_yaw))
        forward_error = error[0] * heading[0] + error[1] * heading[1]
        feedforward = target_velocity[0] * target_heading[0] + target_velocity[1] * target_heading[1]
        longitudinal_speed = velocity[0] * heading[0] + velocity[1] * heading[1]
        desired_speed = clamp(
            feedforward + 1.2 * forward_error + 0.45 * (feedforward - longitudinal_speed),
            -2.5,
            2.5,
        )
        yaw_error = wrap_angle(target_yaw - yaw)
        average = clamp(0.32 * (desired_speed - longitudinal_speed), -0.55, 0.55)
        turn = clamp(0.35 * yaw_error + 0.08 * (target_yaw_rate - angular[2]), -0.55, 0.55)
        if t >= 6.2:
            average = clamp(-2.0 * longitudinal_speed, -0.95, 0.95)
            turn = clamp(-0.35 * angular[2], -0.50, 0.50)
        elif t >= 5.0:
            average = clamp(0.45 * forward_error - 0.9 * longitudinal_speed, -0.95, 0.95)
            turn = clamp(0.35 * yaw_error - 0.20 * angular[2], -0.50, 0.50)

        controls = {}
        for actuator, name in zip(LEFT_ACTUATORS, LEFT_JOINTS):
            wheel_damping = 0.012 * joint(observation, name)["velocity"]
            controls[actuator] = clamp(average - turn - wheel_damping, -1.0, 1.0)
        for actuator, name in zip(RIGHT_ACTUATORS, RIGHT_JOINTS):
            wheel_damping = 0.012 * joint(observation, name)["velocity"]
            controls[actuator] = clamp(average + turn - wheel_damping, -1.0, 1.0)

        if t >= 7.0:
            force_x = 2400.0 * error[0] - 9000.0 * velocity[0]
            force_y = 2400.0 * error[1] - 9000.0 * velocity[1]
        elif t >= 6.2:
            force_x = 3200.0 * error[0] - 6000.0 * velocity[0]
            force_y = 3200.0 * error[1] - 6000.0 * velocity[1]
        elif t >= 5.0:
            force_x = 1200.0 * error[0] - 1800.0 * velocity[0]
            force_y = 1200.0 * error[1] - 1800.0 * velocity[1]
        else:
            force_x = 700.0 * error[0] + 300.0 * (target_velocity[0] - velocity[0])
            force_y = 700.0 * error[1] + 300.0 * (target_velocity[1] - velocity[1])
        magnitude = math.hypot(force_x, force_y)
        force_limit = 5000.0 if t >= 5.0 else 2000.0
        if magnitude > force_limit:
            force_x *= force_limit / magnitude
            force_y *= force_limit / magnitude
        roll, pitch = roll_pitch_from_quat(state["quaternion"])
        controls.update({
            "base_force_x_ctrl": force_x,
            "base_force_y_ctrl": force_y,
            "base_force_z_ctrl": clamp((-16000.0 if t >= 7.0 else -10000.0) * velocity[2], -5500.0, 5500.0),
            "base_roll_ctrl": clamp(-800.0 * angular[0], -1800.0, 1800.0),
            "base_pitch_ctrl": clamp(-800.0 * angular[1], -1800.0, 1800.0),
            "base_yaw_ctrl": clamp(800.0 * yaw_error + 140.0 * (target_yaw_rate - angular[2]), -650.0, 650.0),
        })
        markers = ("route_complete",) if t >= 6.2 and math.hypot(*error) < 0.8 else ()
        return ProbeCommand(controls=controls, markers=markers)


PROBE_CLASS = HillsTurnStopProbe
