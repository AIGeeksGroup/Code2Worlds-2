"""Unified Dev15 worker with live mechanics, kinematics, and friction audits."""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import sys
from pathlib import Path
import time
from typing import Any
import types

import mujoco
import numpy as np

from physics_diagnostics import annotate_kinematics, energy_summary, mechanics_snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _execute_source(path: Path, module_name: str):
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(path.read_bytes(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def import_class(path: Path, symbol: str):
    module_name = "simcode4d_dynamic_" + hashlib.sha256(
        (str(path) + ":" + sha256_file(path)).encode()
    ).hexdigest()[:16]
    support_names = [
        name
        for name in (
            "probe_api", "base_control", "mobile_control",
            "common", "dev15_common",
        )
        if (path.parent / (name + ".py")).is_file()
    ]
    managed_names = support_names + [module_name]
    sentinel = object()
    previous_modules = {
        name: sys.modules.get(name, sentinel)
        for name in managed_names
    }
    previous_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        for support_name in support_names:
            _execute_source(path.parent / (support_name + ".py"), support_name)
        module = _execute_source(path, module_name)
    finally:
        sys.path[:] = previous_path
        for name, previous in previous_modules.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return getattr(module, symbol)


def _named_id(model, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise KeyError(name)
    return int(result)


def _body_free_joint(model, body_name: str) -> int | None:
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    first = int(model.body_jntadr[body_id])
    count = int(model.body_jntnum[body_id])
    for joint_id in range(first, first + count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return None


def _set_free_pose(model, data, body_name: str, position=None, quaternion=None) -> None:
    joint_id = _body_free_joint(model, body_name)
    if joint_id is None:
        raise ValueError("body has no free joint: " + body_name)
    address = int(model.jnt_qposadr[joint_id])
    if position is not None:
        data.qpos[address:address + 3] = np.asarray(position, dtype=float)
    if quaternion is not None:
        quat = np.asarray(quaternion, dtype=float)
        quat = quat / np.linalg.norm(quat)
        data.qpos[address + 3:address + 7] = quat


def _offset_free_pose(model, data, body_name: str, xyz, yaw_rad: float = 0.0) -> None:
    joint_id = _body_free_joint(model, body_name)
    if joint_id is None:
        return
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address:address + 3] += np.asarray(xyz, dtype=float)
    if yaw_rad:
        current = data.qpos[address + 3:address + 7].copy()
        delta = np.asarray([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])
        result = np.empty(4)
        mujoco.mju_mulQuat(result, delta, current)
        data.qpos[address + 3:address + 7] = result / np.linalg.norm(result)


def _hfield_height(model, geom_name: str, x: float, y: float) -> float:
    geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    hfield_id = int(model.geom_dataid[geom_id])
    rows = int(model.hfield_nrow[hfield_id])
    cols = int(model.hfield_ncol[hfield_id])
    hx, hy, hz, _ = model.hfield_size[hfield_id]
    cx, cy, cz = model.geom_pos[geom_id]
    u = np.clip((x - (cx - hx)) / (2.0 * hx) * (cols - 1), 0.0, cols - 1.0)
    v = np.clip((y - (cy - hy)) / (2.0 * hy) * (rows - 1), 0.0, rows - 1.0)
    c0, r0 = int(np.floor(u)), int(np.floor(v))
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    du, dv = u - c0, v - r0
    address = int(model.hfield_adr[hfield_id])
    values = model.hfield_data[address:address + rows * cols].reshape(rows, cols)
    normalized = (
        (1.0 - du) * (1.0 - dv) * values[r0, c0]
        + du * (1.0 - dv) * values[r0, c1]
        + (1.0 - du) * dv * values[r1, c0]
        + du * dv * values[r1, c1]
    )
    return float(cz + hz * normalized)


def _initialize_hfield_vehicle(model, data, init: dict[str, Any]) -> None:
    body_name = init["body"]
    x, y = init["xy"]
    yaw = float(init["yaw_rad"])
    wheel_xy = np.asarray(init["wheel_xy_m"], dtype=float)
    radius = float(init["wheel_radius_m"])
    center_z = float(init["wheel_center_z_m"])
    heading = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    lateral = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
    points = np.asarray([x, y]) + wheel_xy[:, :1] * heading[:2] + wheel_xy[:, 1:] * lateral[:2]
    heights = np.asarray([_hfield_height(model, init["hfield_geom"], px, py) for px, py in points])
    matrix = np.column_stack([points, np.ones(len(points))])
    a, b, c = np.linalg.lstsq(matrix, heights, rcond=None)[0]
    normal = np.asarray([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    forward = heading - np.dot(heading, normal) * normal
    forward /= np.linalg.norm(forward)
    side = np.cross(normal, forward)
    rotation = np.column_stack([forward, side, normal])
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    ground = float(a * x + b * y + c)
    root = np.asarray([x, y, ground]) + normal * (radius - center_z + float(init.get("clearance_m", 0.003)))
    _set_free_pose(model, data, body_name, root, quaternion)


def _seat_hfield_vehicle(model, data, init: dict[str, Any]) -> None:
    joint_id = _body_free_joint(model, init["body"])
    if joint_id is None:
        raise ValueError("hfield vehicle has no free joint: " + init["body"])
    address = int(model.jnt_qposadr[joint_id])
    target_distance = float(init.get("contact_target_distance_m", -0.0001))
    mujoco.mj_forward(model, data)
    for _ in range(64):
        if data.ncon:
            minimum_distance = min(float(data.contact[index].dist) for index in range(data.ncon))
            correction = target_distance - minimum_distance
            if abs(correction) <= 1e-7:
                break
            data.qpos[address + 2] += correction
        else:
            data.qpos[address + 2] -= 0.001
        mujoco.mj_forward(model, data)


def _apply_initialization(model, data, gold: dict[str, Any], perturbation: dict[str, Any]) -> None:
    initialization = gold["initialization"]
    keyframe = initialization.get("keyframe")
    if keyframe:
        key_id = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    if "hfield_vehicle" in initialization:
        _initialize_hfield_vehicle(model, data, initialization["hfield_vehicle"])
    for body_name, pose in initialization.get("free_body_poses", {}).items():
        _set_free_pose(model, data, body_name, pose.get("position"), pose.get("quaternion_wxyz"))
    for joint_name, value in initialization.get("joint_positions", {}).items():
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
    for body_name, offset in initialization.get("free_body_offsets_m", {}).items():
        _offset_free_pose(model, data, body_name, offset)
    for body_name, offset in perturbation.get("body_xy_offset_m", {}).items():
        _offset_free_pose(model, data, body_name, [offset[0], offset[1], 0.0])
    for body_name, yaw in perturbation.get("body_yaw_offset_rad", {}).items():
        _offset_free_pose(model, data, body_name, [0.0, 0.0, 0.0], float(yaw))
    if "hfield_vehicle" in initialization:
        _seat_hfield_vehicle(model, data, initialization["hfield_vehicle"])
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    if "hfield_vehicle" in initialization:
        max_depth = max((max(0.0, -float(data.contact[i].dist)) for i in range(data.ncon)), default=0.0)
        if max_depth > 0.001:
            raise RuntimeError(f"hfield initialization penetration exceeds 1 mm: {max_depth}")


def _apply_model_perturbation(model, perturbation: dict[str, Any]) -> None:
    for body_name, scale in perturbation.get("body_mass_scale", {}).items():
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        model.body_mass[body_id] *= float(scale)
        model.body_inertia[body_id] *= float(scale)
    for geom_name, scale in perturbation.get("geom_friction_scale", {}).items():
        geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_friction[geom_id] *= float(scale)

    for body_name, scale in perturbation.get("body_geom_friction_scale", {}).items():
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_id in range(model.ngeom):
            current = int(model.geom_bodyid[geom_id])
            while current > 0 and current != body_id:
                current = int(model.body_parentid[current])
            if current == body_id:
                model.geom_friction[geom_id] *= float(scale)

def _model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    arrays = (
        model.body_mass, model.body_inertia, model.geom_type, model.geom_size,
        model.geom_pos, model.geom_quat, model.geom_friction, model.geom_contype,
        model.geom_conaffinity, model.jnt_type, model.jnt_axis, model.jnt_range,
        model.actuator_gear, model.actuator_ctrlrange, model.eq_type, model.eq_data,
    )
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def _object_velocity(model, data, body_id: int) -> tuple[list[float], list[float]]:
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
    )
    return velocity[3:].tolist(), velocity[:3].tolist()


def _contact_snapshot(model, data) -> tuple[list[dict[str, Any]], float, float]:
    contacts = []
    maximum_depth = 0.0
    maximum_force = 0.0
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "geom_" + str(contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "geom_" + str(contact.geom2)
        body1_id = int(model.geom_bodyid[contact.geom1])
        body2_id = int(model.geom_bodyid[contact.geom2])
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or "world"
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or "world"
        mujoco.mj_contactForce(model, data, index, force)
        depth = max(0.0, -float(contact.dist))
        normal_force = abs(float(force[0]))
        tangential_force = math.hypot(float(force[1]), float(force[2]))
        torsional_torque = abs(float(force[3]))
        rolling_torque = math.hypot(float(force[4]), float(force[5]))
        friction = [float(value) for value in contact.friction]
        active_friction_axes = 0
        if int(contact.dim) >= 3:
            active_friction_axes = 2
        if int(contact.dim) >= 4:
            active_friction_axes = 3
        if int(contact.dim) >= 6:
            active_friction_axes = 5
        normalized_friction_loads = []
        if normal_force > 1e-9:
            for axis in range(active_friction_axes):
                coefficient = max(abs(friction[axis]), 1e-12)
                normalized_friction_loads.append(
                    abs(float(force[axis + 1])) / (coefficient * normal_force)
                )
        cone_is_pyramidal = int(model.opt.cone) == int(
            mujoco.mjtCone.mjCONE_PYRAMIDAL
        )
        if cone_is_pyramidal:
            cone_utilization = sum(normalized_friction_loads)
        else:
            cone_utilization = math.sqrt(
                sum(value * value for value in normalized_friction_loads)
            )
        linear1, angular1 = _object_velocity(model, data, body1_id)
        linear2, angular2 = _object_velocity(model, data, body2_id)
        point = np.asarray(contact.pos, dtype=float)
        velocity1 = (
            np.asarray(linear1)
            + np.cross(np.asarray(angular1), point - data.xpos[body1_id])
        )
        velocity2 = (
            np.asarray(linear2)
            + np.cross(np.asarray(angular2), point - data.xpos[body2_id])
        )
        normal_axis = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
        relative_velocity = velocity2 - velocity1
        tangential_velocity = relative_velocity - normal_axis * float(np.dot(relative_velocity, normal_axis))
        slip_speed = float(np.linalg.norm(tangential_velocity))
        maximum_depth = max(maximum_depth, depth)
        maximum_force = max(maximum_force, normal_force)
        contacts.append({
            "geom1": geom1,
            "geom2": geom2,
            "body1": body1,
            "body2": body2,
            "distance_m": float(contact.dist),
            "normal_force_N": normal_force,
            "tangential_force_N": tangential_force,
            "contact_frame_force_torque": [float(value) for value in force],
            "torsional_torque_Nm": torsional_torque,
            "rolling_torque_Nm": rolling_torque,
            "friction_coefficients": {"sliding_1": friction[0], "sliding_2": friction[1], "torsional": friction[2], "rolling_1": friction[3], "rolling_2": friction[4]},
            "contact_dimension": int(contact.dim),
            "friction_cone_type": "pyramidal" if cone_is_pyramidal else "elliptic",
            "normalized_friction_loads": normalized_friction_loads,
            "coulomb_cone_utilization": cone_utilization,
            "tangential_slip_speed_mps": slip_speed,
        })
    return contacts, maximum_depth, maximum_force


def _group_active(model, data, binding: dict[str, Any]) -> bool:
    condition = binding.get("active_when")
    if not condition:
        return True
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, condition["joint"])
    value = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    if "minimum_position" in condition and value < float(condition["minimum_position"]):
        return False
    if "maximum_position" in condition and value > float(condition["maximum_position"]):
        return False
    return True


def _group_matches(contact: dict[str, Any], endpoint: int, binding: dict[str, Any]) -> bool:
    geom_name = contact["geom" + str(endpoint)]
    body_name = contact["body" + str(endpoint)]
    return (
        any(fnmatch.fnmatchcase(geom_name, pattern) for pattern in binding.get("geoms", []))
        or any(fnmatch.fnmatchcase(body_name, pattern) for pattern in binding.get("bodies", []))
    )


def _semantic_contacts(model, data, contacts: list[dict[str, Any]], policy: dict[str, Any], pairs: list[list[str]]) -> dict[str, dict[str, Any]]:
    groups = policy["groups"]
    matches: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        left_binding = groups[left]
        right_binding = groups[right]
        if not _group_active(model, data, left_binding) or not _group_active(model, data, right_binding):
            continue
        rule = left + "/" + right
        for contact in contacts:
            if contact["normal_force_N"] <= 0.0 and contact["distance_m"] >= 0.0:
                continue
            direct = _group_matches(contact, 1, left_binding) and _group_matches(contact, 2, right_binding)
            reverse = _group_matches(contact, 2, left_binding) and _group_matches(contact, 1, right_binding)
            if direct or reverse:
                matches[rule] = contact
                break
    return matches


def _forbidden_contacts(model, data, contacts: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _semantic_contacts(model, data, contacts, policy, policy["forbidden"])


def _joint_limit_violations(model, data, tolerance: float = 1e-4) -> dict[str, dict[str, float]]:
    violations: dict[str, dict[str, float]] = {}
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        position = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        excess = max(lower - position, position - upper, 0.0)
        if excess > tolerance:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "joint_" + str(joint_id)
            violations[name] = {
                "position": position,
                "lower": lower,
                "upper": upper,
                "excess": excess,
            }
    return violations


def _qualified_trace(trace: list[set[str]], timestep_s: float, dwell_s: float) -> tuple[bool, float, list[dict[str, Any]]]:
    minimum_steps = max(1, int(math.ceil(dwell_s / timestep_s - 1e-12)))
    qualified = np.zeros(len(trace), dtype=bool)
    runs: list[dict[str, Any]] = []
    labels = sorted(set().union(*trace)) if trace else []
    for label in labels:
        start = None
        for index in range(len(trace) + 1):
            active = index < len(trace) and label in trace[index]
            if active and start is None:
                start = index
            elif not active and start is not None:
                count = index - start
                if count >= minimum_steps:
                    qualified[start:index] = True
                    runs.append({
                        "rule": label,
                        "start_time_s": (start + 1) * timestep_s,
                        "end_time_s": index * timestep_s,
                        "duration_s": count * timestep_s,
                    })
                start = None
    ratio = float(np.count_nonzero(qualified) / len(trace)) if trace else 0.0
    return bool(runs), ratio, runs


def _observation(model, data, body_names: list[str], joint_names: list[str]) -> dict[str, Any]:
    bodies = {}
    for name in body_names:
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        linear, angular = _object_velocity(model, data, body_id)
        bodies[name] = {
            "position": data.xpos[body_id].tolist(),
            "quaternion": data.xquat[body_id].tolist(),
            "linear_velocity": linear,
            "angular_velocity": angular,
        }
    joints = {}
    for name in joint_names:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        joints[name] = {
            "position": float(data.qpos[qpos_address]),
            "velocity": float(data.qvel[dof_address]),
        }
    contacts, depth, force = _contact_snapshot(model, data)
    return {
        "time_s": float(data.time),
        "bodies": bodies,
        "joints": joints,
        "contacts": contacts,
        "maximum_penetration_m": depth,
        "maximum_contact_force_N": force,
    }


def _apply_static_field(model, data, field: dict[str, Any] | None) -> dict[str, float]:
    data.xfrc_applied[:] = 0.0
    if not field:
        return {}
    if field["type"] != "static_fluid":
        raise ValueError("unsupported static field: " + field["type"])
    body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, field["body"])
    body_position = data.xpos[body_id]
    linear, angular = _object_velocity(model, data, body_id)
    water_level = float(field["surface_z_m"])
    depth_span = float(field["immersion_span_m"])
    submerged = max(0.0, min(1.0, (water_level - (body_position[2] - depth_span * 0.5)) / depth_span))
    density = float(field["density_kg_m3"])
    volume = float(field["displaced_volume_m3"])
    buoyancy = density * 9.81 * volume * submerged
    vx, _, vz = linear
    drag_x = -float(field["drag_x"]) * submerged * vx * abs(vx)
    drag_z = -float(field["drag_z"]) * submerged * vz * abs(vz)
    pitch_drag = -float(field["pitch_drag"]) * submerged * angular[1]
    data.xfrc_applied[body_id] = [drag_x, 0.0, buoyancy + drag_z, 0.0, pitch_drag, 0.0]
    return {
        "submerged_fraction": submerged,
        "buoyancy_N": buoyancy,
        "drag_x_N": drag_x,
        "drag_z_N": drag_z,
    }


def _resolve_controls(model, requested: dict[str, float], limits: dict[str, list[float]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    controls = np.zeros(model.nu)
    violations: list[dict[str, Any]] = []
    for name, raw_value in requested.items():
        actuator_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("non-finite actuator command: " + name)
        if name in limits:
            lo, hi = limits[name]
        elif bool(model.actuator_ctrllimited[actuator_id]):
            lo, hi = model.actuator_ctrlrange[actuator_id]
        else:
            raise ValueError("missing explicit limit for actuator: " + name)
        lower, upper = float(lo), float(hi)
        if value < lower - 1e-9 or value > upper + 1e-9:
            violations.append({
                "actuator": name, "requested": value,
                "lower": lower, "upper": upper,
            })
        controls[actuator_id] = max(lower, min(upper, value))
    return controls, violations


def _try_grasp(model, data, api: dict[str, Any], observation: dict[str, Any]) -> tuple[bool, str]:
    equality_id = _named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, api["equality"])
    if bool(data.eq_active[equality_id]):
        return True, "already_active"
    a = observation["bodies"][api["body_a"]]["position"]
    b = observation["bodies"][api["body_b"]]["position"]
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    if distance > float(api["maximum_distance_m"]):
        return False, "distance_gate"
    if api.get("contact_required", False):
        body_a = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, api["body_a"])
        body_b = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, api["body_b"])
        touching = False
        for contact in data.contact[:data.ncon]:
            pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            if body_a in pair and body_b in pair:
                touching = True
                break
        if not touching:
            return False, "contact_gate"
    data.eq_active[equality_id] = 1
    return True, "activated"


def run_episode(
    suite_root: Path,
    case_spec_path: Path,
    case_gold_path: Path,
    generation_seed: int,
    physics_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    case_spec = load_json(case_spec_path)
    case_gold = load_json(case_gold_path)
    workspace = suite_root.parents[1]
    model_path = workspace / case_spec["artifacts"]["runtime_model"]["path"]
    expected_model_hash = case_spec["artifacts"]["runtime_model"]["sha256"]
    if sha256_file(model_path) != expected_model_hash:
        raise RuntimeError("runtime model hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = float(case_spec["simulator"]["timestep_s"])
    perturbation = copy.deepcopy(case_gold["perturbations"][str(physics_seed)])
    _apply_model_perturbation(model, perturbation)
    data = mujoco.MjData(model)
    _apply_initialization(model, data, case_gold, perturbation)
    protected_hash = _model_fingerprint(model)

    probe_path = suite_root / case_spec["probe"]["path"]
    validator_path = suite_root / case_spec["validator"]["path"]
    ProbeClass = import_class(probe_path, "PROBE_CLASS")
    probe = ProbeClass()
    probe.reset(case_spec, case_gold)
    actuator_limits = case_spec["probe"]["actuator_limits"]
    body_names = case_spec["observation"]["bodies"]
    joint_names = case_spec["observation"]["joints"]
    field = case_spec.get("static_field")
    grasp_api = case_spec["probe"].get("grasp_api")
    contact_policy = case_gold["contact_policy"]
    allowed_equality_id = None
    if grasp_api:
        allowed_equality_id = _named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, grasp_api["equality"])
    initial_eq_active = data.eq_active.copy()

    initial_observation = _observation(model, data, body_names, joint_names)
    initial_admissibility = {
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "maximum_penetration_m": initial_observation["maximum_penetration_m"],
        "penetration_pass_1mm": initial_observation["maximum_penetration_m"] <= 0.001,
    }

    duration = float(case_spec["probe"]["episode_duration_s"])
    log_hz = float(case_spec["probe"]["log_hz"])
    log_period = 1.0 / log_hz
    next_log = 0.0
    frames = []
    events = []
    maximum_penetration = initial_observation["maximum_penetration_m"]
    maximum_force = initial_observation["maximum_contact_force_N"]
    maximum_penetration_contact = None
    maximum_penetration_time_s = 0.0
    penetration_samples: list[float] = []
    force_samples: list[float] = []
    tangential_force_samples: list[float] = []
    friction_utilization_samples: list[float] = []
    slip_speed_samples: list[float] = []
    observed_semantic_contacts: set[str] = set()
    forbidden_trace: list[set[str]] = []
    joint_limit_trace: list[set[str]] = []
    friction_failure_trace: list[set[str]] = []
    friction_examples: dict[str, dict[str, Any]] = {}
    forbidden_examples: dict[str, dict[str, Any]] = {}
    joint_limit_examples: dict[str, dict[str, float]] = {}
    action_bound_examples: dict[str, dict[str, Any]] = {}
    undeclared_equality_examples: dict[str, dict[str, Any]] = {}
    finite = True
    grasp_active = False
    start = time.perf_counter()

    while data.time < duration - model.opt.timestep * 0.5:
        observation = _observation(model, data, body_names, joint_names)
        command = probe.command(float(data.time), copy.deepcopy(observation))
        resolved_controls, control_violations = _resolve_controls(model, command.controls, actuator_limits)
        data.ctrl[:] = resolved_controls
        for violation in control_violations:
            actuator = violation["actuator"]
            if actuator not in action_bound_examples:
                example = dict(violation)
                example["t"] = float(data.time)
                action_bound_examples[actuator] = example
        if command.grasp_request and grasp_api:
            activated, reason = _try_grasp(model, data, grasp_api, observation)
            if activated and not grasp_active:
                events.append({"event": "grasp_activated", "t": float(data.time), "reason": reason})
                grasp_active = True
        for marker in command.markers:
            if not any(item["event"] == marker for item in events):
                events.append({"event": marker, "t": float(data.time)})
        field_state = _apply_static_field(model, data, field)
        mujoco.mj_step(model, data)
        contacts, depth, force = _contact_snapshot(model, data)
        friction_failures: set[str] = set()
        for contact in contacts:
            penetration_samples.append(max(0.0, -float(contact["distance_m"])))
            force_samples.append(float(contact["normal_force_N"]))
            tangential_force_samples.append(float(contact["tangential_force_N"]))
            friction_utilization_samples.append(float(contact["coulomb_cone_utilization"]))
            slip_speed_samples.append(float(contact["tangential_slip_speed_mps"]))
            if contact["normal_force_N"] > 1e-6 and contact["coulomb_cone_utilization"] > 1.25:
                label = contact["geom1"] + "/" + contact["geom2"]
                friction_failures.add(label)
                if label not in friction_examples:
                    example = dict(contact)
                    example["t"] = float(data.time)
                    friction_examples[label] = example
        friction_failure_trace.append(friction_failures)
        allowed_matches = _semantic_contacts(model, data, contacts, contact_policy, contact_policy["allowed"])
        for rule, contact in allowed_matches.items():
            if rule not in observed_semantic_contacts:
                events.append({"event": "semantic_contact", "rule": rule, "t": float(data.time), "contact": contact})
                observed_semantic_contacts.add(rule)

        forbidden_matches = _forbidden_contacts(model, data, contacts, contact_policy)
        forbidden_trace.append(set(forbidden_matches))
        for rule, contact in forbidden_matches.items():
            if rule not in forbidden_examples:
                example = dict(contact)
                example["t"] = float(data.time)
                forbidden_examples[rule] = example
        joint_limit_matches = _joint_limit_violations(model, data)
        joint_limit_trace.append(set(joint_limit_matches))
        for joint_name, violation in joint_limit_matches.items():
            if joint_name not in joint_limit_examples:
                example = dict(violation)
                example["t"] = float(data.time)
                joint_limit_examples[joint_name] = example
        for equality_id in range(model.neq):
            if bool(data.eq_active[equality_id]) and not bool(initial_eq_active[equality_id]) and equality_id != allowed_equality_id:
                equality_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id) or "equality_" + str(equality_id)
                if equality_name not in undeclared_equality_examples:
                    undeclared_equality_examples[equality_name] = {"equality": equality_name, "t": float(data.time)}
        if depth > maximum_penetration:
            maximum_penetration = depth
            maximum_penetration_time_s = float(data.time)
            maximum_penetration_contact = min(contacts, key=lambda item: item["distance_m"]) if contacts else None
        maximum_force = max(maximum_force, force)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            events.append({"event": "non_finite_state", "t": float(data.time)})
            break
        if data.time + 1e-9 >= next_log:
            frame_observation = _observation(model, data, body_names, joint_names)
            frames.append({
                "t": float(data.time),
                "mechanics": mechanics_snapshot(model, data),
                "bodies": frame_observation["bodies"],
                "joints": frame_observation["joints"],
                "contacts": contacts,
                "maximum_penetration_m": depth,
                "maximum_contact_force_N": force,
                "controls": {
                    model.actuator(i).name: float(data.ctrl[i])
                    for i in range(model.nu)
                },
                "field": field_state,
                "grasp_active": grasp_active,
            })
            next_log += log_period

    dwell_time_s = 0.010
    forbidden_episode, forbidden_ratio, forbidden_runs = _qualified_trace(
        forbidden_trace, float(model.opt.timestep), dwell_time_s
    )
    joint_limit_episode, joint_limit_ratio, joint_limit_runs = _qualified_trace(
        joint_limit_trace, float(model.opt.timestep), dwell_time_s
    )
    friction_cone_episode, friction_cone_ratio, friction_cone_runs = _qualified_trace(
        friction_failure_trace, float(model.opt.timestep), dwell_time_s
    )
    for run in forbidden_runs:
        run["first_observation"] = forbidden_examples[run["rule"]]
        events.append({
            "event": "forbidden_contact_violation",
            "t": run["start_time_s"],
            "rule": run["rule"],
        })
    for run in joint_limit_runs:
        run["first_observation"] = joint_limit_examples[run["rule"]]
        events.append({
            "event": "joint_limit_violation",
            "t": run["start_time_s"],
            "joint": run["rule"],
        })
    for run in friction_cone_runs:
        run["first_observation"] = friction_examples[run["rule"]]
        events.append({
            "event": "friction_cone_failure",
            "t": run["start_time_s"],
            "contact_pair": run["rule"],
        })
    dynamic_penetration_p95 = float(np.percentile(penetration_samples, 95)) if penetration_samples else 0.0
    contact_force_p95 = float(np.percentile(force_samples, 95)) if force_samples else 0.0
    tangential_force_p95 = float(np.percentile(tangential_force_samples, 95)) if tangential_force_samples else 0.0
    friction_utilization_p95 = float(np.percentile(friction_utilization_samples, 95)) if friction_utilization_samples else 0.0
    friction_utilization_max = max(friction_utilization_samples, default=0.0)
    slip_speed_p95 = float(np.percentile(slip_speed_samples, 95)) if slip_speed_samples else 0.0
    slip_speed_max = max(slip_speed_samples, default=0.0)
    friction_warning_sample_ratio = (
        sum(value > 1.05 for value in friction_utilization_samples)
        / len(friction_utilization_samples)
        if friction_utilization_samples
        else 0.0
    )
    kinematics = annotate_kinematics(frames)
    energy = energy_summary(frames)

    mutation_pass = protected_hash == _model_fingerprint(model)
    raw = {
        "schema_version": "simcode4d.dev15.episode-log.v1",
        "case_id": case_spec["case_id"],
        "world_id": case_spec["world_id"],
        "generation_seed": generation_seed,
        "physics_seed": physics_seed,
        "artifact_hashes": {
            "runtime_model_sha256": expected_model_hash,
            "case_spec_sha256": sha256_file(case_spec_path),
            "case_gold_sha256": sha256_file(case_gold_path),
            "probe_sha256": sha256_file(probe_path),
            "validator_sha256": sha256_file(validator_path),
            "runtime_sha256": sha256_file(Path(__file__)),
            "physics_diagnostics_sha256": sha256_file(Path(__file__).with_name("physics_diagnostics.py")),
        },
        "simulator": {
            "name": "mujoco",
            "version": mujoco.__version__,
            "timestep_s": float(model.opt.timestep),
        },
        "initial_admissibility": initial_admissibility,
        "finite_execution": finite,
        "model_mutation_guard_pass": mutation_pass,
        "integrity_audit": {
            "action_bound_violation": bool(action_bound_examples),
            "action_bound_examples": [action_bound_examples[key] for key in sorted(action_bound_examples)],
            "undeclared_equality_activation": bool(undeclared_equality_examples),
            "undeclared_equality_examples": [undeclared_equality_examples[key] for key in sorted(undeclared_equality_examples)],
            "integrity_violation": bool(
                not finite
                or not mutation_pass
                or action_bound_examples
                or undeclared_equality_examples
            ),
        },
        "safety": {
            "minimum_dwell_s": dwell_time_s,
            "forbidden_contact_episode": forbidden_episode,
            "forbidden_contact_time_ratio": forbidden_ratio,
            "forbidden_contact_runs": forbidden_runs,
            "joint_limit_violation_episode": joint_limit_episode,
            "joint_limit_violation_time_ratio": joint_limit_ratio,
            "joint_limit_violation_runs": joint_limit_runs,
            "friction_cone_failure_episode": friction_cone_episode,
            "friction_cone_failure_time_ratio": friction_cone_ratio,
            "friction_cone_failure_runs": friction_cone_runs,
            "friction_warning_sample_ratio": friction_warning_sample_ratio,
            "friction_utilization_p95": friction_utilization_p95,
            "friction_utilization_max": friction_utilization_max,
            "tangential_slip_speed_p95_mps": slip_speed_p95,
            "tangential_slip_speed_max_mps": slip_speed_max,
            "dynamic_penetration_p95_m": dynamic_penetration_p95,
            "contact_force_p95_N": contact_force_p95,
        },
        "grasp_api_used": grasp_active,
        "events": events,
        "frames": frames,
        "summary": {
            "duration_simulated_s": float(data.time),
            "maximum_penetration_m": maximum_penetration,
            "maximum_penetration_time_s": maximum_penetration_time_s,
            "maximum_penetration_contact": maximum_penetration_contact,
            "maximum_contact_force_N": maximum_force,
            "tangential_contact_force_p95_N": tangential_force_p95,
            "friction_utilization_p95": friction_utilization_p95,
            "friction_utilization_max": friction_utilization_max,
            "tangential_slip_speed_p95_mps": slip_speed_p95,
            "tangential_slip_speed_max_mps": slip_speed_max,
            "friction_warning_sample_ratio": friction_warning_sample_ratio,
            "friction_contact_sample_count": len(friction_utilization_samples),
            "slip_reporting_sample_count": sum(
                value > 0.02 for value in slip_speed_samples
            ),
            "dynamic_penetration_p95_m": dynamic_penetration_p95,
            "contact_force_p95_N": contact_force_p95,
            **kinematics,
            **energy,
            "wall_time_s": time.perf_counter() - start,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "episode_log.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, sort_keys=True)

    ValidatorClass = import_class(validator_path, "VALIDATOR_CLASS")
    validation = ValidatorClass().validate(raw, case_spec, case_gold)
    validation["raw_event_log_sha256"] = sha256_file(raw_path)
    validation["validator_sha256"] = sha256_file(validator_path)
    validation_path = output_dir / "validation.json"
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
    return validation
