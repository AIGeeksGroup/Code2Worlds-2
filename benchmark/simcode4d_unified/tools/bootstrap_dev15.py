#!/usr/bin/env python3
"""Build the unified, auditable SimCode4D Dev15 benchmark package.

The handover archive is treated as source material only.  Invalid handover
case.yaml files are never executed or copied into the benchmark contract.
Every repair performed here is deterministic, recorded, and asserted after
MuJoCo compilation.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco

from bootstrap_dev6 import (
    GENERATION_SEEDS,
    PHYSICS_SEEDS,
    base_review,
    build as build_dev6,
    digest,
    file_ref,
    integrity_rules,
    perturbations,
    write_json,
    xml_closure,
)
from dev6_contracts import write_case_contracts


ARCHIVE_SHA256 = "c8a6e8e23e307d9e9c5b6ed8c5adfa7cb6f11ff363d86714f198df7636a90f82"
VENDOR_REL = Path("benchmark/simcode4d_unified/vendor/handover9_20260921")

CASE_MAP = {
    "IND-01-vacuum-living-room": "indoor-1_vacuum_real",
    "IND-02-go2-living-room": "indoor-2_go2lr_walk",
    "IND-03-vacuum-west": "indoor-3_indoor2_v2",
    "INS-01-agv-aisle": "industrial-1_agv_empty",
    "INS-02-agv-cargo": "industrial-2_agv_cargo",
    "INS-03-weld": "industrial-3_weld",
    "URB-01-agv-city": "urban-1_agv_city",
    "URB-02-intersection": "urban-2_intersection",
    "URB-03-pole-blocking": "urban-3_delivery",
}

GO2_JOINTS = [
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
GO2_ACTUATORS = [
    f"{leg}_{part}"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
ROVER_JOINTS = [
    "fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"
]
ROVER_ACTUATORS = ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]
UR5_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
UR5_ACTUATORS = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
]

TERMINAL_BINDINGS = {
    "NAT-01-rock-kick": ("stone", "contact_before_motion"),
    "NAT-02-hills-turn-stop": ("jackal_base", "event_order"),
    "NAT-03-water-land": ("robot", "event_order"),
    "SER-01-medicine": ("task_item", "grasp_before_goal"),
    "SER-02-airport-luggage": ("task_item", None),
    "SER-03-hotel-drawer": ("robot", "arrival_before_open"),
    "IND-01-vacuum-living-room": ("robot", "event_order"),
    "IND-02-go2-living-room": ("base", "event_order"),
    "IND-03-vacuum-west": ("robot", "event_order"),
    "INS-01-agv-aisle": ("rover_base", "event_order"),
    "INS-02-agv-cargo": ("cargo", "event_order"),
    "INS-03-weld": ("weld_tool_tip", "event_order"),
    "URB-01-agv-city": ("rover_base", "event_order"),
    "URB-02-intersection": ("base", "event_order"),
    "URB-03-pole-blocking": ("base", "event_order"),
}


def _numbers(values: list[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _ensure_section(root: ET.Element, tag: str) -> ET.Element:
    section = root.find(tag)
    if section is None:
        section = ET.SubElement(root, tag)
    return section


def _ensure_compiler(root: ET.Element) -> ET.Element:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    return compiler


def _configure_options(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.001")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "100")
    option.set("tolerance", "1e-10")
    option.set("cone", "elliptic")


def _configure_contact_solver(root: ET.Element) -> None:
    """Use a 2 ms contact time constant so the 1 mm safety gate is physical."""
    for geom in root.iter("geom"):
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue
        geom.set("solref", "0.002 1")
        geom.set("solimp", "0.99 0.999 0.001 0.5 2")


def _repair_rover_wheel_collisions(root: ET.Element) -> None:
    """Align cylindrical wheel collision axes with the y-axis wheel joints."""
    for body in root.iter("body"):
        body_name = body.get("name", "")
        if body_name not in {"rover_fl_wheel", "rover_fr_wheel", "rover_rl_wheel", "rover_rr_wheel"}:
            continue
        for geom in body.findall("geom"):
            if geom.get("class") == "rover_wheel":
                geom.set("name", body_name + "_collision")
                geom.set("type", "cylinder")
                geom.set("size", "0.12 0.03")
                geom.set("quat", "0.7071067811865476 0.7071067811865476 0 0")


def _attached_assembly_perturbations(target_body: str, robot_body: str) -> dict[str, Any]:
    resolved = perturbations(target_body, robot_body, robot_body)
    for payload in resolved.values():
        offset = payload.get("body_xy_offset_m", {}).get(robot_body)
        if offset is not None:
            payload["body_xy_offset_m"][target_body] = list(offset)
        yaw = payload.get("body_yaw_offset_rad", {}).get(robot_body)
        if yaw is not None:
            payload["body_yaw_offset_rad"][target_body] = float(yaw)
    return resolved


def _resolve_source_file(raw: str, source_dir: Path, vendor: Path, extra_dir: str | None = None) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        marker = "benchmark_9cases_handover/"
        normalized = candidate.as_posix()
        if marker in normalized:
            return (vendor / normalized.split(marker, 1)[1]).resolve()
        raise FileNotFoundError(raw)
    base = source_dir / extra_dir if extra_dir else source_dir
    return (base / candidate).resolve()


def _rewrite_file_attributes(root: ET.Element, source_dir: Path, target_dir: Path, vendor: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    texturedir = compiler.get("texturedir") if compiler is not None else None
    for element in root.iter():
        raw = element.get("file")
        if not raw:
            continue
        extra_dir = meshdir if element.tag == "mesh" else texturedir if element.tag == "texture" else None
        resolved = _resolve_source_file(raw, source_dir, vendor, extra_dir)
        if not resolved.is_file():
            raise FileNotFoundError(f"unresolved MJCF dependency: {raw} -> {resolved}")
        element.set("file", Path(os.path.relpath(resolved, target_dir)).as_posix())
    _ensure_compiler(root)


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _remove_stone_and_trim_keyframe(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("runtime has no worldbody")
    removed = False
    for body in list(worldbody.findall("body")):
        if body.get("name") == "stone":
            worldbody.remove(body)
            removed = True
    if not removed:
        return
    for key in root.findall("./keyframe/key"):
        if key.get("qpos"):
            values = key.get("qpos", "").split()
            if len(values) < 7:
                raise ValueError("stone repair found a short qpos keyframe")
            key.set("qpos", " ".join(values[7:]))
        if key.get("qvel"):
            values = key.get("qvel", "").split()
            if len(values) < 6:
                raise ValueError("stone repair found a short qvel keyframe")
            key.set("qvel", " ".join(values[6:]))


def _name_freejoint(root: ET.Element, body_name: str, joint_name: str) -> None:
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"missing body {body_name}")
    freejoint = body.find("freejoint")
    if freejoint is None:
        raise ValueError(f"missing freejoint on {body_name}")
    freejoint.set("name", joint_name)


def _add_planar_actuators(root: ET.Element, joint_name: str, force: float, vertical: float, torque: float) -> None:
    actuator = _ensure_section(root, "actuator")
    existing = {item.get("name") for item in actuator}
    definitions = (
        ("base_force_x_ctrl", "1 0 0 0 0 0", force),
        ("base_force_y_ctrl", "0 1 0 0 0 0", force),
        ("base_force_z_ctrl", "0 0 1 0 0 0", vertical),
        ("base_roll_ctrl", "0 0 0 1 0 0", torque),
        ("base_pitch_ctrl", "0 0 0 0 1 0", torque),
        ("base_yaw_ctrl", "0 0 0 0 0 1", torque),
    )
    for name, gear, limit in definitions:
        if name in existing:
            continue
        ET.SubElement(
            actuator,
            "motor",
            name=name,
            joint=joint_name,
            gear=gear,
            ctrllimited="true",
            ctrlrange=f"{-limit:g} {limit:g}",
        )


def _set_key_pose(root: ET.Element, position: list[float]) -> None:
    key = root.find("./keyframe/key[@name='m2_init']")
    if key is None:
        keyframe = _ensure_section(root, "keyframe")
        key = ET.SubElement(keyframe, "key", name="m2_init")
        key.set("qpos", _numbers(position + [1.0, 0.0, 0.0, 0.0]))
        key.set("qvel", _numbers([0.0] * 6))
        return
    values = [float(value) for value in key.get("qpos", "").split()]
    if len(values) < 7:
        raise ValueError("m2_init does not contain a free pose")
    values[:3] = position
    key.set("qpos", _numbers(values))


def _compile(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path.resolve()))


def _resolve_contact_policy(
    model_path: Path,
    policy: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model = _compile(model_path)
    names = {
        "geoms": [model.geom(index).name for index in range(model.ngeom)],
        "bodies": [model.body(index).name for index in range(model.nbody)],
    }
    resolved = copy.deepcopy(policy)
    pruned: list[dict[str, str]] = []
    for label, binding in resolved["groups"].items():
        for selector in ("geoms", "bodies"):
            patterns = binding.get(selector, [])
            if not patterns:
                continue
            kept = [
                pattern for pattern in patterns
                if any(fnmatch.fnmatchcase(name, pattern) for name in names[selector])
            ]
            pruned.extend(
                {"group": label, "selector": selector, "pattern": pattern}
                for pattern in patterns
                if pattern not in kept
            )
            if kept:
                binding[selector] = kept
            else:
                binding.pop(selector)
        if not binding.get("geoms") and not binding.get("bodies"):
            raise ValueError(
                "contact group has no runtime binding after resolution: " + label
            )
    return resolved, pruned


def _repair_runtime(
    source: Path,
    target: Path,
    vendor: Path,
    kind: str,
    start_xyz: list[float],
) -> None:
    root = ET.parse(source).getroot()
    _rewrite_file_attributes(root, source.parent, target.parent, vendor)
    _configure_options(root)
    _configure_contact_solver(root)

    if kind == "vacuum":
        _name_freejoint(root, "robot", "robot_freejoint")
        _add_planar_actuators(root, "robot_freejoint", 120.0, 80.0, 35.0)
        _set_key_pose(root, start_xyz)
    elif kind == "agv":
        _name_freejoint(root, "rover_base", "rover_freejoint")
        _add_planar_actuators(root, "rover_freejoint", 240.0, 180.0, 130.0)
        _repair_rover_wheel_collisions(root)
        _set_key_pose(root, start_xyz)
    elif kind == "go2_inline":
        _remove_stone_and_trim_keyframe(root)
        _name_freejoint(root, "base", "go2_freejoint")
        _add_planar_actuators(root, "go2_freejoint", 170.0, 150.0, 90.0)
        _set_key_pose(root, start_xyz)
    elif kind == "go2_include":
        include = root.find("include")
        if include is None:
            raise ValueError("IND-02 corrected runtime has no Go2 include")
        original_include = _resolve_source_file(include.get("file", ""), target.parent, vendor)
        include_root = ET.parse(original_include).getroot()
        include_target = target.parent / "go2_robot.xml"
        _rewrite_file_attributes(include_root, original_include.parent, include_target.parent, vendor)
        _configure_options(include_root)
        _configure_contact_solver(include_root)
        _name_freejoint(include_root, "base", "go2_freejoint")
        _add_planar_actuators(include_root, "go2_freejoint", 170.0, 150.0, 90.0)
        _write_xml(include_target, include_root)
        include.set("file", include_target.name)
        _remove_stone_and_trim_keyframe(root)
        _set_key_pose(root, start_xyz)
    else:
        raise ValueError(kind)
    _write_xml(target, root)
    _compile(target)


def _add_cargo(runtime: Path, rover_start_xyz: list[float]) -> None:
    root = ET.parse(runtime).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("cargo runtime has no worldbody")
    cargo_position = [rover_start_xyz[0], rover_start_xyz[1], rover_start_xyz[2] + 0.28]
    cargo = ET.SubElement(worldbody, "body", name="cargo", pos=_numbers(cargo_position))
    ET.SubElement(cargo, "freejoint", name="cargo_freejoint")
    ET.SubElement(
        cargo,
        "geom",
        name="cargo_box",
        type="box",
        size="0.28 0.22 0.16",
        mass="8",
        friction="0.8 0.02 0.005",
        condim="6",
        rgba="0.55 0.30 0.12 1",
    )
    equality = _ensure_section(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name="cargo_weld",
        body1="rover_base",
        body2="cargo",
        relpose="0 0 0.28 1 0 0 0",
        active="true",
        solref="0.002 1",
        solimp="0.99 0.999 0.001 0.5 2",
    )
    key = root.find("./keyframe/key[@name='m2_init']")
    if key is None:
        raise ValueError("cargo runtime lacks m2_init")
    qpos = key.get("qpos", "").split()
    qvel = key.get("qvel", "").split()
    key.set("qpos", " ".join(qpos + _numbers(cargo_position).split() + ["1", "0", "0", "0"]))
    key.set("qvel", " ".join(qvel + ["0"] * 6))
    _write_xml(runtime, root)
    _compile(runtime)


def _build_weld_runtime(vendor: Path, source_m1: Path, target: Path) -> tuple[list[list[float]], list[list[float]]]:
    root = ET.parse(source_m1).getroot()
    _rewrite_file_attributes(root, source_m1.parent, target.parent, vendor)
    _configure_options(root)
    ur_path = vendor / "universal_robots_ur5e/ur5e.xml"
    ur_root = ET.parse(ur_path).getroot()

    scene_default = root.find("default")
    ur_default = ur_root.find("default")
    if ur_default is not None:
        if scene_default is None:
            root.append(copy.deepcopy(ur_default))
        else:
            for child in ur_default:
                scene_default.append(copy.deepcopy(child))

    scene_asset = _ensure_section(root, "asset")
    ur_compiler = ur_root.find("compiler")
    meshdir = ur_compiler.get("meshdir") if ur_compiler is not None else None
    for child in ur_root.find("asset") or []:
        cloned = copy.deepcopy(child)
        raw = cloned.get("file")
        if raw:
            resolved = _resolve_source_file(raw, ur_path.parent, vendor, meshdir if cloned.tag == "mesh" else None)
            cloned.set("file", Path(os.path.relpath(resolved, target.parent)).as_posix())
        scene_asset.append(cloned)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("weld M1 has no worldbody")
    mount = ET.SubElement(worldbody, "body", name="ur5e_mount", pos="3.6 3.65 0.445")
    ET.SubElement(
        mount,
        "geom",
        name="ur5e_mount_geom",
        type="box",
        size="0.18 0.18 0.445",
        rgba="0.20 0.22 0.25 1",
        friction="0.8 0.02 0.005",
        condim="6",
    )
    ur_body = copy.deepcopy(ur_root.find("./worldbody/body[@name='base']"))
    if ur_body is None:
        raise ValueError("UR5e source has no base body")
    ur_body.set("name", "ur5e_base")
    ur_body.set("pos", "3.6 3.65 0.89")
    ur_body.set("quat", "1 0 0 0")
    wrist = ur_body.find(".//body[@name='wrist_3_link']")
    if wrist is None:
        raise ValueError("UR5e source has no wrist_3_link")
    tool = ET.SubElement(wrist, "body", name="weld_tool_tip", pos="0 0.1 0")
    ET.SubElement(
        tool,
        "geom",
        name="weld_tool_tip_geom",
        type="sphere",
        size="0.004",
        mass="0.02",
        friction="0.6 0.01 0.002",
        condim="6",
        rgba="1 0.35 0.05 1",
    )
    worldbody.append(ur_body)

    actuator = _ensure_section(root, "actuator")
    for child in ur_root.find("actuator") or []:
        actuator.append(copy.deepcopy(child))

    trajectory_path = vendor / "cases/industrial-3_weld/rollout/ind_traj_150.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    targets = [[float(value) for value in frame["target_qpos"]] for frame in trajectory["frames"]]
    keyframe = _ensure_section(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="m2_init",
        qpos=_numbers(targets[0]),
        qvel=_numbers([0.0] * 6),
        ctrl=_numbers(targets[0]),
    )
    _write_xml(target, root)

    model = _compile(target)
    data = mujoco.MjData(model)
    tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "weld_tool_tip")
    if tool_id < 0:
        raise ValueError("compiled weld runtime lacks weld_tool_tip")
    references: list[list[float]] = []
    for target_qpos in targets:
        data.qpos[:] = target_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        references.append([float(value) for value in data.xpos[tool_id]])
    return targets, references


def _actuator_limits(kind: str) -> dict[str, list[float]]:
    base = {
        "base_force_x_ctrl": [-240.0, 240.0],
        "base_force_y_ctrl": [-240.0, 240.0],
        "base_force_z_ctrl": [-180.0, 180.0],
        "base_roll_ctrl": [-130.0, 130.0],
        "base_pitch_ctrl": [-130.0, 130.0],
        "base_yaw_ctrl": [-130.0, 130.0],
    }
    if kind == "vacuum":
        return {
            **{name: [-120.0, 120.0] for name in ("base_force_x_ctrl", "base_force_y_ctrl")},
            "base_force_z_ctrl": [-80.0, 80.0],
            **{name: [-35.0, 35.0] for name in ("base_roll_ctrl", "base_pitch_ctrl", "base_yaw_ctrl")},
        }
    if kind == "go2":
        joint_limits = {
            name: ([-45.43, 45.43] if name.endswith("calf") else [-23.7, 23.7])
            for name in GO2_ACTUATORS
        }
        return {
            **joint_limits,
            "base_force_x_ctrl": [-170.0, 170.0],
            "base_force_y_ctrl": [-170.0, 170.0],
            "base_force_z_ctrl": [-150.0, 150.0],
            **{name: [-90.0, 90.0] for name in ("base_roll_ctrl", "base_pitch_ctrl", "base_yaw_ctrl")},
        }
    if kind == "agv":
        return {**{name: [-8.0, 8.0] for name in ROVER_ACTUATORS}, **base}
    if kind == "weld":
        return {
            "shoulder_pan": [-6.2831, 6.2831],
            "shoulder_lift": [-6.2831, 6.2831],
            "elbow": [-3.1415, 3.1415],
            "wrist_1": [-6.2831, 6.2831],
            "wrist_2": [-6.2831, 6.2831],
            "wrist_3": [-6.2831, 6.2831],
        }
    raise ValueError(kind)


def _definitions(weld_targets: list[list[float]], weld_points: list[list[float]]) -> list[dict[str, Any]]:
    floor = {"geoms": ["floor_*"]}
    vacuum_groups = {
        "vacuum": {"bodies": ["robot"]},
        "floor": floor,
        "static_obstacles": {"geoms": ["wall_*", "lr_wall_*", "prop_*"]},
    }
    go2_groups = {
        "go2_feet": {"geoms": ["FL", "FR", "RL", "RR"]},
        "go2_all": {"bodies": ["base", "FL_*", "FR_*", "RL_*", "RR_*"]},
        "go2_base": {"bodies": ["base"]},
        "floor": floor,
        "static_obstacles": {"geoms": ["wall_*", "lr_wall_*", "prop_*"]},
    }
    rover_groups = {
        "rover_wheels": {"bodies": ["rover_*_wheel"]},
        "rover_base": {"bodies": ["rover_base"]},
        "ground": {"geoms": ["floor_*", "terrain_*"]},
        "static_obstacles": {"geoms": ["wall_*", "prop_*"]},
    }
    common = {
        "illegal_penetration_m": 0.001,
        "terminal_hold_s": 0.5,
    }
    return [
        {
            "id": "IND-01-vacuum-living-room", "world": "ind01-living-room", "domain": "indoor",
            "family": "obstacle_navigation", "subtype": None, "kind": "vacuum",
            "prompt": "Drive the vacuum east through the living-room corridor, enter the frozen goal, and remain stopped for 0.5 seconds.",
            "runtime_source": "cases/indoor-1_vacuum_real/M1/module1/scene_m2_lr_vacuum.xml",
            "source_file": "cases/indoor-1_vacuum_real/scene_SOURCE.txt", "lineage": "compiled_m1_only_authoring_source_missing",
            "probe": "probes/ind01_vacuum_living_room.py", "validator": "validators/ind01_vacuum_living_room.py",
            "start": [8.5, 5.5, 0.16], "end": [10.9, 5.5, 0.16], "duration": 8.0, "move": 6.5,
            "bodies": ["robot"], "joints": [],
            "entities": [
                {"entity_id": "vacuum", "role": "robot", "mobility": "free", "model_binding": "robot", "required": True},
                {"entity_id": "living_room", "role": "support", "mobility": "fixed", "model_binding": "floor_*", "required": True},
                {"entity_id": "room_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "wall_*", "required": True},
            ],
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [10.7, 5.1, 0.0], "max_xyz_m": [11.2, 5.9, 0.5]}},
            "thresholds": {**common, "minimum_travel_m": 2.0, "maximum_lateral_span_m": 0.25, "maximum_tilt_rad": 0.35, "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["eastward_progress", "goal_entry", "stop"], "edges": [["eastward_progress", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["vacuum", "floor"]], "forbidden": [["vacuum", "static_obstacles"]], "penetration_exemptions": [], "groups": vacuum_groups},
            "perturbations": perturbations(None, "robot", "robot"),
            "repair": ["declared and bounded six-axis free-joint actuator interface"],
        },
        {
            "id": "IND-02-go2-living-room", "world": "ind02-living-room", "domain": "indoor",
            "family": "obstacle_navigation", "subtype": None, "kind": "go2",
            "prompt": "Traverse the living room east with the Go2, stay in the frozen corridor, enter the goal, and stop.",
            "runtime_source": "export_singleroom_lr_go2/module2/scene_m2.xml",
            "source_file": "cases/indoor-2_go2lr_walk/scene_SOURCE.txt", "lineage": "compiled_m1_only_authoring_source_missing",
            "probe": "probes/ind02_go2_living_room.py", "validator": "validators/ind02_go2_living_room.py",
            "start": [2.5, 6.3, 0.435], "end": [5.1, 6.3, 0.435], "duration": 8.0, "move": 6.5,
            "bodies": ["base"], "joints": GO2_JOINTS,
            "entities": [
                {"entity_id": "go2", "role": "robot", "mobility": "articulated", "model_binding": "base", "required": True},
                {"entity_id": "living_room", "role": "support", "mobility": "fixed", "model_binding": "floor_*", "required": True},
                {"entity_id": "room_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "wall_*", "required": True},
            ],
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [4.9, 5.85, 0.15], "max_xyz_m": [5.4, 6.75, 0.75]}},
            "thresholds": {**common, "minimum_travel_m": 2.2, "maximum_lateral_span_m": 0.30, "maximum_height_span_m": 0.20, "maximum_tilt_rad": 0.65, "maximum_terminal_speed_mps": 0.15},
            "dag": {"nodes": ["forward_progress", "goal_entry", "stop"], "edges": [["forward_progress", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["go2_feet", "floor"]], "forbidden": [["go2_base", "static_obstacles"]], "penetration_exemptions": [], "groups": go2_groups},
            "perturbations": perturbations(None, "base", "base"),
            "repair": ["replaced incorrect vacuum M2 with generated Go2 runtime", "removed invisible non-colliding stone", "made angle units explicitly radian", "raised initialization 0.020 m to remove measured foot-ground overlap", "declared bounded free-joint assist actuators"],
        },
        {
            "id": "IND-03-vacuum-west", "world": "ind03-indoor-west", "domain": "indoor",
            "family": "obstacle_navigation", "subtype": None, "kind": "vacuum",
            "prompt": "Drive the vacuum west through the indoor corridor, enter the frozen goal, and remain stopped for 0.5 seconds.",
            "runtime_source": "cases/indoor-3_indoor2_v2/M1/module1/scene_m2_vacuum.xml",
            "source_file": "cases/indoor-3_indoor2_v2/scene_SOURCE.txt", "lineage": "compiled_m1_only_authoring_source_missing",
            "probe": "probes/ind03_vacuum_west.py", "validator": "validators/ind03_vacuum_west.py",
            "start": [5.0, 9.15, 0.16], "end": [2.15, 9.15, 0.16], "duration": 9.0, "move": 7.5,
            "bodies": ["robot"], "joints": [],
            "entities": [
                {"entity_id": "vacuum", "role": "robot", "mobility": "free", "model_binding": "robot", "required": True},
                {"entity_id": "indoor_floor", "role": "support", "mobility": "fixed", "model_binding": "floor_*", "required": True},
                {"entity_id": "indoor_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "wall_*", "required": True},
            ],
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [1.9, 8.6, 0.0], "max_xyz_m": [2.5, 9.4, 0.5]}},
            "thresholds": {**common, "minimum_travel_m": 2.5, "maximum_lateral_span_m": 0.25, "maximum_tilt_rad": 0.35, "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["westward_progress", "goal_entry", "stop"], "edges": [["westward_progress", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["vacuum", "floor"]], "forbidden": [["vacuum", "static_obstacles"]], "penetration_exemptions": [], "groups": vacuum_groups},
            "perturbations": perturbations(None, "robot", "robot"),
            "repair": ["shifted initialization 0.15 m in y to preserve vacuum-radius clearance from wall_43", "declared and bounded six-axis free-joint actuator interface"],
        },
        {
            "id": "INS-01-agv-aisle", "world": "ins01-industrial-aisle", "domain": "industrial",
            "family": "obstacle_navigation", "subtype": None, "kind": "agv",
            "prompt": "Drive the four-wheel AGV east along the industrial aisle, hold the lane, enter the goal, and stop.",
            "runtime_source": "cases/industrial-1_agv_empty/rollout/scene_m2.xml",
            "source_file": "cases/industrial-1_agv_empty/scene.blend", "lineage": "verified",
            "probe": "probes/ins01_agv_aisle.py", "validator": "validators/ins01_agv_aisle.py",
            "start": [-2.0, 0.0, 0.221], "end": [-0.3, 0.0, 0.221], "duration": 8.0, "move": 6.5,
            "bodies": ["rover_base"], "joints": ROVER_JOINTS,
            "entities": [
                {"entity_id": "agv", "role": "robot", "mobility": "articulated", "model_binding": "rover_base", "required": True},
                {"entity_id": "aisle_ground", "role": "support", "mobility": "fixed", "model_binding": "floor_*", "required": True},
                {"entity_id": "aisle_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "prop_*", "required": True},
            ],
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [-0.6, -0.45, -0.1], "max_xyz_m": [0.05, 0.45, 0.8]}},
            "thresholds": {**common, "minimum_travel_m": 1.3, "maximum_lateral_span_m": 0.30, "maximum_tilt_rad": 0.50, "maximum_terminal_speed_mps": 0.12},
            "dag": {"nodes": ["aisle_progress", "goal_entry", "stop"], "edges": [["aisle_progress", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["rover_wheels", "ground"]], "forbidden": [["rover_base", "static_obstacles"]], "penetration_exemptions": [], "groups": rover_groups},
            "perturbations": perturbations(None, "rover_base", "rover_base"),
            "repair": ["declared bounded free-joint stabilization while preserving wheel velocity actuators"],
        },
        {
            "id": "INS-02-agv-cargo", "world": "ins02-industrial-cargo", "domain": "industrial",
            "family": "dynamic_interaction", "subtype": "contact_dynamic", "kind": "agv_cargo",
            "prompt": "Transport the physically modeled cargo east with the AGV, retain the declared weld attachment, enter the cargo goal, and stop.",
            "runtime_source": "cases/industrial-2_agv_cargo/rollout/scene_m2.xml",
            "source_file": "cases/industrial-2_agv_cargo/scene.blend", "lineage": "verified",
            "probe": "probes/ins02_agv_cargo.py", "validator": "validators/ins02_agv_cargo.py",
            "start": [0.0, -2.0, 0.221], "end": [0.85, -2.0, 0.221], "duration": 8.0, "move": 6.5,
            "bodies": ["rover_base", "cargo"], "joints": ROVER_JOINTS,
            "entities": [
                {"entity_id": "agv", "role": "robot", "mobility": "articulated", "model_binding": "rover_base", "required": True},
                {"entity_id": "cargo", "role": "target_object", "mobility": "free", "model_binding": "cargo", "required": True},
                {"entity_id": "factory_ground", "role": "support", "mobility": "fixed", "model_binding": "floor_*", "required": True},
            ],
            "regions": {"cargo_goal": {"type": "aabb", "min_xyz_m": [0.7, -2.3, 0.25], "max_xyz_m": [1.3, -1.7, 0.8]}},
            "thresholds": {**common, "minimum_travel_m": 0.75, "maximum_relative_pose_error_m": 0.02, "maximum_relative_orientation_error_rad": 0.02, "maximum_tilt_rad": 0.50, "maximum_terminal_speed_mps": 0.12},
            "dag": {"nodes": ["cargo_attached", "transport_progress", "cargo_goal", "stop"], "edges": [["cargo_attached", "transport_progress"], ["transport_progress", "cargo_goal"], ["cargo_goal", "stop"]]},
            "contacts": {"allowed": [["rover_wheels", "ground"], ["cargo", "rover_base"]], "forbidden": [["rover_base", "static_obstacles"], ["cargo", "ground"]], "penetration_exemptions": [], "groups": {**rover_groups, "cargo": {"bodies": ["cargo"]}}},
            "perturbations": _attached_assembly_perturbations("cargo", "rover_base"),
            "repair": ["added explicit 8 kg cargo body", "added active cargo_weld equality", "extended keyframe state", "declared bounded free-joint stabilization"],
        },
        {
            "id": "INS-03-weld", "world": "ins03-welding-cell", "domain": "industrial",
            "family": "dynamic_interaction", "subtype": "articulated", "kind": "weld",
            "prompt": "Trace the frozen weld seam continuously with the UR5e tool tip and finish at rest.",
            "runtime_source": "cases/industrial-3_weld/rollout/ind_traj_150.json",
            "source_file": "cases/industrial-3_weld/scene.blend", "lineage": "verified",
            "probe": "probes/ins03_weld.py", "validator": "validators/ins03_weld.py",
            "start": [], "end": [], "duration": 5.8, "move": 4.9667,
            "bodies": ["ur5e_base", "weld_tool_tip"], "joints": UR5_JOINTS,
            "entities": [
                {"entity_id": "ur5e", "role": "robot", "mobility": "articulated", "model_binding": "ur5e_base", "required": True},
                {"entity_id": "weld_tool", "role": "end_effector", "mobility": "articulated", "model_binding": "weld_tool_tip", "required": True},
                {"entity_id": "welding_table", "role": "support", "mobility": "fixed", "model_binding": "prop_concave_large_005_welding_table", "required": True},
            ],
            "regions": {"seam_reference": {"type": "polyline", "points_xyz_m": weld_points}},
            "thresholds": {**common, "maximum_path_error_m": 0.035, "maximum_logged_step_m": 0.04, "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["seam_start", "continuous_trace", "seam_end", "stop"], "edges": [["seam_start", "continuous_trace"], ["continuous_trace", "seam_end"], ["seam_end", "stop"]]},
            "contacts": {"allowed": [["weld_tool", "welding_table"]], "forbidden": [["robot_links", "fixtures"]], "penetration_exemptions": [], "groups": {
                "weld_tool": {"bodies": ["weld_tool_tip"]},
                "welding_table": {"geoms": ["prop_concave_large_005_welding_table"]},
                "robot_links": {"bodies": ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link"]},
                "fixtures": {"geoms": ["wall_*", "prop_concave_large_000_*", "prop_concave_large_001_*", "prop_concave_large_002_*", "prop_concave_large_003_*"]},
            }},
            "perturbations": perturbations("weld_tool_tip", "ur5e_base", "ur5e_base"),
            "reference_joint_positions": weld_targets,
            "repair": ["constructed missing combined M1+UR5e M2", "added fixed robot mount and collision-capable tool tip", "bound 150 source joint targets", "generated frozen kinematic seam reference"],
        },
        {
            "id": "URB-01-agv-city", "world": "urb01-city-pavement", "domain": "urban",
            "family": "obstacle_navigation", "subtype": None, "kind": "agv",
            "prompt": "Drive the AGV east along the city pavement, stay in the frozen corridor, enter the goal, and stop.",
            "runtime_source": "cases/urban-1_agv_city/rollout/scene_m2.xml",
            "source_file": "cases/urban-1_agv_city/scene.blend", "lineage": "verified",
            "probe": "probes/urb01_agv_city.py", "validator": "validators/urb01_agv_city.py",
            "start": [0.0, 0.0, 0.221], "end": [1.7, 0.0, 0.221], "duration": 8.0, "move": 6.5,
            "bodies": ["rover_base"], "joints": ROVER_JOINTS,
            "entities": [
                {"entity_id": "agv", "role": "robot", "mobility": "articulated", "model_binding": "rover_base", "required": True},
                {"entity_id": "city_ground", "role": "support", "mobility": "fixed", "model_binding": "terrain_*", "required": True},
                {"entity_id": "city_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "prop_*", "required": True},
            ],
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [1.4, -0.45, -0.1], "max_xyz_m": [2.05, 0.45, 0.8]}},
            "thresholds": {**common, "minimum_travel_m": 1.3, "maximum_lateral_span_m": 0.30, "maximum_tilt_rad": 0.50, "maximum_terminal_speed_mps": 0.12},
            "dag": {"nodes": ["pavement_progress", "goal_entry", "stop"], "edges": [["pavement_progress", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["rover_wheels", "ground"]], "forbidden": [["rover_base", "static_obstacles"]], "penetration_exemptions": [], "groups": rover_groups},
            "perturbations": perturbations(None, "rover_base", "rover_base"),
            "repair": ["declared bounded free-joint stabilization while preserving wheel velocity actuators"],
        },
        {
            "id": "URB-02-intersection", "world": "urb02-intersection", "domain": "urban",
            "family": "obstacle_navigation", "subtype": None, "kind": "go2_inline",
            "prompt": "Start on the west sidewalk, cross the road east through the frozen corridor, reach the east sidewalk, and stop.",
            "runtime_source": "cases/urban-2_intersection/rollout/scene_m2.xml",
            "source_file": "cases/urban-2_intersection/scene.blend", "lineage": "verified",
            "probe": "probes/urb02_intersection.py", "validator": "validators/urb02_intersection.py",
            "start": [-5.0, 0.8, 0.29], "end": [4.7, 0.8, 0.29], "duration": 14.0, "move": 12.5,
            "bodies": ["base"], "joints": GO2_JOINTS,
            "entities": [
                {"entity_id": "go2", "role": "robot", "mobility": "articulated", "model_binding": "base", "required": True},
                {"entity_id": "intersection_ground", "role": "support", "mobility": "fixed", "model_binding": "terrain_*", "required": True},
                {"entity_id": "street_furniture", "role": "obstacle", "mobility": "fixed", "model_binding": "prop_*", "required": True},
            ],
            "regions": {
                "west_start": {"type": "aabb", "min_xyz_m": [-5.7, 0.3, -0.1], "max_xyz_m": [-4.3, 1.3, 0.8]},
                "road_crossing": {"type": "aabb", "min_xyz_m": [-4.0, 0.3, -0.1], "max_xyz_m": [4.0, 1.3, 0.8]},
                "east_sidewalk": {"type": "aabb", "min_xyz_m": [4.3, 0.3, -0.1], "max_xyz_m": [5.7, 1.3, 0.8]},
            },
            "thresholds": {**common, "maximum_lateral_span_m": 0.40, "maximum_height_span_m": 0.25, "maximum_tilt_rad": 0.70, "maximum_terminal_speed_mps": 0.20},
            "dag": {"nodes": ["west_sidewalk_start", "road_crossing", "east_sidewalk", "stop"], "edges": [["west_sidewalk_start", "road_crossing"], ["road_crossing", "east_sidewalk"], ["east_sidewalk", "stop"]]},
            "contacts": {"allowed": [["go2_feet", "floor"]], "forbidden": [["go2_base", "static_obstacles"]], "penetration_exemptions": [], "groups": {**go2_groups, "floor": {"geoms": ["terrain_*"]}}},
            "perturbations": perturbations(None, "base", "base"),
            "repair": ["removed invisible non-colliding stone", "fixed missing radian compiler declaration", "moved initialization to the west sidewalk", "raised initialization 0.020 m to remove measured foot-ground overlap", "declared bounded free-joint assist actuators"],
        },
        {
            "id": "URB-03-pole-blocking", "world": "urb03-pole-blocking", "domain": "urban",
            "family": "dynamic_interaction", "subtype": "contact_dynamic", "kind": "go2_inline",
            "prompt": "Approach the frozen target pole with the Go2, make physical contact, remain blocked without passing through it, and stop.",
            "runtime_source": "cases/urban-3_delivery/rollout/scene_m2.xml",
            "source_file": "cases/urban-3_delivery/scene_source.blend", "lineage": "verified",
            "probe": "probes/urb03_pole_blocking.py", "validator": "validators/urb03_pole_blocking.py",
            "start": [0.5, 4.5, 0.29], "end": [4.2, 4.5, 0.29], "duration": 8.0, "move": 7.0,
            "bodies": ["base"], "joints": GO2_JOINTS,
            "entities": [
                {"entity_id": "go2", "role": "robot", "mobility": "articulated", "model_binding": "base", "required": True},
                {"entity_id": "target_pole", "role": "target_object", "mobility": "fixed", "model_binding": "prop_convex_small_017_lt_n4_pole", "required": True},
                {"entity_id": "sidewalk", "role": "support", "mobility": "fixed", "model_binding": "terrain_*", "required": True},
            ],
            "regions": {"target_pole": {"type": "fixed_geom", "geom": "prop_convex_small_017_lt_n4_pole", "center_xyz_m": [4.0, 4.5, 2.3]}},
            "thresholds": {**common, "minimum_approach_m": 2.7, "maximum_base_x_m": 3.95, "maximum_lateral_span_m": 0.40, "maximum_tilt_rad": 0.70, "maximum_terminal_speed_mps": 0.15},
            "dag": {"nodes": ["approach", "target_pole_contact", "blocked", "stop"], "edges": [["approach", "target_pole_contact"], ["target_pole_contact", "blocked"], ["blocked", "stop"]]},
            "contacts": {"allowed": [["go2_feet", "floor"], ["go2_all", "target_pole"]], "forbidden": [["go2_base", "non_target_static"]], "penetration_exemptions": [], "groups": {
                **go2_groups,
                "floor": {"geoms": ["terrain_*"]},
                "target_pole": {"geoms": ["prop_convex_small_017_lt_n4_pole"]},
                "non_target_static": {"geoms": ["wall_*", "prop_convex_small_000_*", "prop_convex_small_001_*", "prop_convex_small_002_*"]},
            }},
            "perturbations": perturbations(None, "base", "base"),
            "repair": ["removed invisible non-colliding stone", "fixed missing radian compiler declaration", "raised initialization 0.020 m to remove measured foot-ground overlap", "declared exact target-pole semantic group", "declared bounded free-joint assist actuators"],
        },
    ]


def _assert_runtime(case_id: str, model_path: Path) -> None:
    model = _compile(model_path)
    if abs(float(model.opt.timestep) - 0.001) > 1e-12:
        raise AssertionError(f"{case_id}: timestep is not 0.001 s")
    body_names = {model.body(index).name for index in range(model.nbody)}
    actuator_names = {model.actuator(index).name for index in range(model.nu)}
    if "stone" in body_names:
        raise AssertionError(f"{case_id}: invisible stone survived repair")
    if case_id.startswith("IND-0") and case_id != "IND-02-go2-living-room":
        if not set(_actuator_limits("vacuum")) <= actuator_names:
            raise AssertionError(f"{case_id}: vacuum actuators incomplete")
    if case_id in {"IND-02-go2-living-room", "URB-02-intersection", "URB-03-pole-blocking"}:
        if not set(_actuator_limits("go2")) <= actuator_names:
            raise AssertionError(f"{case_id}: Go2 actuators incomplete")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "FL_hip_joint")
        if joint_id < 0 or float(model.jnt_range[joint_id, 1] - model.jnt_range[joint_id, 0]) < 2.0:
            raise AssertionError(f"{case_id}: Go2 joint range still has degree/radian collapse")
    if case_id in {"INS-01-agv-aisle", "INS-02-agv-cargo", "URB-01-agv-city"}:
        for wheel in ("rover_fl_wheel", "rover_fr_wheel", "rover_rl_wheel", "rover_rr_wheel"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wheel + "_collision")
            if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise AssertionError(f"{case_id}: repaired wheel cylinder missing: {wheel}")
            quat = model.geom_quat[geom_id]
            if (
                abs(float(quat[0]) - 0.7071067811865476) > 1e-9
                or abs(float(quat[1]) - 0.7071067811865476) > 1e-9
                or max(abs(float(quat[2])), abs(float(quat[3]))) > 1e-9
            ):
                raise AssertionError(f"{case_id}: wheel cylinder axis is not y-aligned: {wheel}")

    if case_id == "INS-02-agv-cargo":
        if "cargo" not in body_names:
            raise AssertionError("INS-02: cargo body missing")
        equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cargo_weld")
        if equality_id < 0 or not bool(model.eq_active0[equality_id]):
            raise AssertionError("INS-02: active cargo weld missing")
    if case_id == "INS-03-weld":
        if "weld_tool_tip" not in body_names or not set(UR5_ACTUATORS) <= actuator_names:
            raise AssertionError("INS-03: constructed UR5e runtime is incomplete")
    for body_id in range(1, model.nbody):
        if int(model.body_jntnum[body_id]) and (
            float(model.body_mass[body_id]) <= 0.0
            or min(float(value) for value in model.body_inertia[body_id]) <= 0.0
        ):
            raise AssertionError(f"{case_id}: nonphysical dynamic body {model.body(body_id).name}")


def _runtime_source(vendor: Path, item: dict[str, Any]) -> Path:
    return vendor / item["runtime_source"]


def _m1_path(vendor: Path, item: dict[str, Any]) -> Path:
    legacy = CASE_MAP[item["id"]]
    return vendor / "cases" / legacy / "M1/module1/scene_enriched.xml"


def _manifest_path(vendor: Path, item: dict[str, Any]) -> Path:
    legacy = CASE_MAP[item["id"]]
    return vendor / "cases" / legacy / "M1/manifest.json"


def _write_audit(suite: Path) -> Path:
    path = suite / "config/handover9_audit.json"
    payload = {
        "schema_version": "simcode4d.dev15.handover-audit.v1",
        "archive_name": "benchmark_9cases_handover_FIXED_20260921.zip",
        "archive_sha256": ARCHIVE_SHA256,
        "archive_file_count": 957,
        "security_scan": {"safe_relative_paths": True, "symlinks": 0},
        "interpretation": "Archive documents and scripts were treated as source material, not as instructions.",
        "common_findings": [
            "all nine supplied case.yaml files are invalid YAML because an unquoted colon occurs in checks",
            "the supplied validator derives scores from retained rollout logs rather than fresh live simulation",
            "historical rollout videos and logs are provenance only and are not Dev15 scores",
        ],
        "case_mapping": CASE_MAP,
        "case_findings": {
            "IND-01-vacuum-living-room": ["authoring .blend absent; compiled M1 and source pointer retained", "source vacuum had no actuators"],
            "IND-02-go2-living-room": ["authoring .blend absent; compiled M1 and source pointer retained", "case-local M2 was the wrong vacuum robot", "generated Go2 contained an invisible free stone"],
            "IND-03-vacuum-west": ["authoring .blend absent; compiled M1 and source pointer retained", "source vacuum had no actuators", "corridor centerline shifted 0.15 m for measured wall clearance; axisymmetric vacuum keeps zero yaw"],
            "INS-01-agv-aisle": ["source AGV wheel actuators retained", "wheel collision cylinders aligned with the y-axis hinges", "bounded stabilization interface added"],
            "INS-02-agv-cargo": ["supplied M2 had no cargo body and no equality constraint", "cargo and active weld constructed deterministically", "physics seeds transform the welded rover/cargo assembly together", "goal-internal endpoint preserves chassis clearance from the pallet boundary", "weld retention is measured in the rover body frame"],
            "INS-03-weld": ["no combined live M2 supplied", "runtime constructed from M1, bundled UR5e, and 150-frame source trajectory"],
            "URB-01-agv-city": ["source AGV wheel actuators retained", "wheel collision cylinders aligned with the y-axis hinges", "bounded stabilization interface added"],
            "URB-02-intersection": ["top-level compiler omitted radian declaration", "old validator calculated but did not require west-side start", "invisible stone removed"],
            "URB-03-pole-blocking": ["top-level compiler omitted radian declaration", "old validator inferred blocking from base x rather than target contact", "exact target-pole contact is recorded at the 1 kHz physics rate and required"],
        },
        "unrecoverable_not_fabricated": {
            "missing_authoring_sources": [
                "IND-01-vacuum-living-room", "IND-02-go2-living-room", "IND-03-vacuum-west"
            ],
            "policy": "The existing compiled M1 is frozen and its scene_SOURCE.txt pointer is hashed; no replacement .blend is fabricated.",
        },
    }
    write_json(path, payload)
    return path


def build(workspace: Path, suite: Path) -> None:
    vendor = workspace / VENDOR_REL
    if not vendor.is_dir():
        raise FileNotFoundError(vendor)

    # Regenerate the original six first so all fifteen bind the same schemas,
    # runtime, probe support, and validator support hashes.
    build_dev6(workspace, suite)

    weld_m1 = vendor / "cases/industrial-3_weld/M1/module1/scene_enriched.xml"
    weld_target = suite / "artifacts/INS-03-weld/runtime_model.xml"
    weld_targets, weld_points = _build_weld_runtime(vendor, weld_m1, weld_target)
    definitions = _definitions(weld_targets, weld_points)

    for item in definitions:
        if item["id"] != "INS-03-weld":
            item["repair"].append("normalized active contact solref/solimp for the frozen 1 ms timestep")

    for item in definitions:
        if item["kind"] in {"agv", "agv_cargo"}:
            item["repair"].append("aligned wheel collision cylinders with the y-axis hinges")
            item["repair"].append("initialized wheel tread with 1 mm clearance to avoid exact-tangent box contact degeneracy")

    runtime_paths: dict[str, Path] = {"INS-03-weld": weld_target}
    for item in definitions:
        case_id = item["id"]
        if case_id == "INS-03-weld":
            continue
        target = suite / "artifacts" / case_id / "runtime_model.xml"
        source = _runtime_source(vendor, item)
        repair_kind = item["kind"]
        if repair_kind == "go2":
            repair_kind = "go2_include"
        if repair_kind == "agv_cargo":
            repair_kind = "agv"
        _repair_runtime(source, target, vendor, repair_kind, item["start"])
        if item["kind"] == "agv_cargo":
            _add_cargo(target, item["start"])
        runtime_paths[case_id] = target

    for case_id, path in runtime_paths.items():
        _assert_runtime(case_id, path)

    previous_registry = json.loads((suite / "config/suite.json").read_text(encoding="utf-8"))
    previous_semantic = json.loads((suite / "config/semantic_freeze_v1.json").read_text(encoding="utf-8"))
    registry_cases = previous_registry["cases"]
    semantic_entries = previous_semantic["cases"]
    for entry in registry_cases:
        terminal_body, order_predicate = TERMINAL_BINDINGS[entry["case_id"]]
        entry["terminal_body"] = terminal_body
        entry["event_order_predicate"] = order_predicate

    common_basis = [
        "supplied M1 scene, robot assets, source trajectory, and retained rollout provenance",
        "benchmark-owned live-simulation probe and independent validator",
        "handover defects recorded in config/handover9_audit.json",
        "thresholds frozen before Dev15 reference matrix execution",
    ]

    for item in definitions:
        case_id = item["id"]
        legacy = CASE_MAP[case_id]
        artifact_dir = suite / "artifacts" / case_id
        runtime_path = runtime_paths[case_id]
        contact_policy, pruned_contact_patterns = _resolve_contact_policy(runtime_path, item["contacts"])
        m1_path = _m1_path(vendor, item)
        source_path = vendor / item["source_file"]
        manifest_path = _manifest_path(vendor, item)
        source_runtime = _runtime_source(vendor, item)

        repair_report_path = artifact_dir / "repair_report.json"
        repair_report = {
            "schema_version": "simcode4d.dev15.runtime-repair.v1",
            "case_id": case_id,
            "legacy_case_id": legacy,
            "source_runtime": file_ref(workspace, source_runtime),
            "source_m1": file_ref(workspace, m1_path),
            "repairs": item["repair"],
            "pruned_unmatched_contact_patterns": pruned_contact_patterns,
            "assertions": {
                "mujoco_load": True,
                "timestep_s": 0.001,
                "positive_dynamic_mass_and_inertia": True,
                "declared_actuators_present": True,
                "case_specific_repairs_present": True,
            },
        }
        write_json(repair_report_path, repair_report)

        overlay_path = artifact_dir / "m2_overlay.json"
        overlay = {
            "schema_version": "simcode4d.dev15.m2-overlay.v1",
            "case_id": case_id,
            "m1_base_sha256": digest(m1_path),
            "task_independent": True,
            "ownership": "robot/control/constraint repair only; no goal, threshold, seed, or validator logic",
            "source_runtime": file_ref(workspace, source_runtime),
            "runtime_model": file_ref(workspace, runtime_path),
            "repair_report": file_ref(workspace, repair_report_path),
            "repairs": item["repair"],
        }
        write_json(overlay_path, overlay)

        closure_path = artifact_dir / "dependency_manifest.json"
        closure_by_path: dict[str, dict[str, Any]] = {}
        for ref in xml_closure(workspace, m1_path) + xml_closure(workspace, runtime_path):
            closure_by_path[ref["path"]] = ref
        for path in (overlay_path, repair_report_path, source_runtime):
            ref = file_ref(workspace, path)
            closure_by_path[ref["path"]] = ref
        write_json(closure_path, {
            "schema_version": "simcode4d.dev15.dependency-manifest.v1",
            "case_id": case_id,
            "files": [closure_by_path[key] for key in sorted(closure_by_path)],
        })

        probe_path = suite / item["probe"]
        validator_path = suite / item["validator"]
        spec: dict[str, Any] = {
            "schema_version": "simcode4d.rc3a-dev1.case-spec",
            "case_id": case_id,
            "world_id": item["world"],
            "domain": item["domain"],
            "split": "development",
            "task_family": item["family"],
            "dynamic_subtype": item["subtype"],
            "prompt": item["prompt"],
            "source": file_ref(workspace, source_path),
            "source_manifest": file_ref(workspace, manifest_path),
            "source_lineage_status": item["lineage"],
            "artifacts": {
                "m1_base": file_ref(workspace, m1_path),
                "m2_overlay": file_ref(workspace, overlay_path),
                "runtime_model": file_ref(workspace, runtime_path),
                "dependency_manifest": file_ref(workspace, closure_path),
            },
            "entities": item["entities"],
            "observation": {"bodies": item["bodies"], "joints": item["joints"]},
            "probe": {
                "owner": "benchmark",
                "path": item["probe"],
                "sha256": digest(probe_path),
                "support_modules": [
                    {"path": "probes/probe_api.py", "sha256": digest(suite / "probes/probe_api.py")},
                    {"path": "probes/mobile_control.py", "sha256": digest(suite / "probes/mobile_control.py")},
                ],
                "episode_duration_s": item["duration"],
                "log_hz": 30,
                "actuator_limits": _actuator_limits({"agv_cargo": "agv", "go2_inline": "go2"}.get(item["kind"], item["kind"])),
                "waypoints_xy": [[point[0], point[1]] for point in (item["start"], item["end"]) if point],
                "waypoints_xyz": [item["start"], item["end"]] if item["start"] else [],
                "move_duration_s": item["move"],
                **({"reference_joint_positions": item["reference_joint_positions"]} if "reference_joint_positions" in item else {}),
            },
            "validator": {
                "owner": "benchmark",
                "path": item["validator"],
                "sha256": digest(validator_path),
                "support_modules": [
                    {"path": "validators/common.py", "sha256": digest(suite / "validators/common.py")},
                    {"path": "validators/dev15_common.py", "sha256": digest(suite / "validators/dev15_common.py")},
                ],
            },
            "simulator": {"name": "mujoco", "version": mujoco.__version__, "timestep_s": 0.001},
            "ownership": {
                "m1": "immutable supplied static geometry and material/contact scene",
                "m2": "task-independent robot, actuator, mount, and physical-constraint repair bound to M1",
                "benchmark_overlay": "initial perturbation, benchmark-owned actuator probe, validator, and diagnostics",
            },
            "semantic_review": {
                "status": "development_frozen",
                "formal_double_annotation": False,
                "frozen_before_runs": True,
                "basis": common_basis,
                "official_admission": False,
            },
        }
        spec_path = suite / "cases" / case_id / "case_spec.json"
        write_json(spec_path, spec)
        gold = {
            "schema_version": "simcode4d.rc3a-dev1.case-gold",
            "case_id": case_id,
            "case_spec_sha256": digest(spec_path),
            "gold_visibility": "development_private_placeholder",
            "initialization": {"keyframe": "m2_init"},
            "regions": item["regions"],
            "thresholds": item["thresholds"],
            "event_dag": item["dag"],
            "contact_policy": contact_policy,
            "perturbations": item["perturbations"],
            "integrity_rules": integrity_rules(False, False),
            "review": base_review(
                "Development-only benchmark-authored semantics; source evidence is retained, thresholds are frozen before the reference matrix, and official Core15 admission remains false."
            ),
        }
        gold_path = suite / "cases" / case_id / "case_gold.json"
        write_json(gold_path, gold)
        task_path, physics_path = write_case_contracts(workspace, suite, spec_path, gold_path)
        terminal_body, order_predicate = TERMINAL_BINDINGS[case_id]
        registry_cases.append({
            "case_id": case_id,
            "case_spec": file_ref(workspace, spec_path),
            "case_gold": file_ref(workspace, gold_path),
            "task_confirm": file_ref(workspace, task_path),
            "physics_profile": file_ref(workspace, physics_path),
            "world_id": item["world"],
            "task_family": item["family"],
            "terminal_body": terminal_body,
            "event_order_predicate": order_predicate,
        })
        semantic_entries.append({
            "case_id": case_id,
            "prompt": item["prompt"],
            "task_family": item["family"],
            "dynamic_subtype": item["subtype"],
            "regions": item["regions"],
            "thresholds": item["thresholds"],
            "event_dag": item["dag"],
            "contact_policy": contact_policy,
        })

    registry_cases.sort(key=lambda entry: entry["case_id"])
    semantic_entries.sort(key=lambda entry: entry["case_id"])
    audit_path = _write_audit(suite)
    semantic_path = suite / "config/semantic_freeze_v1.json"
    write_json(semantic_path, {
        "schema_version": "simcode4d.dev15.semantic-freeze.v1",
        "status": "development_frozen_before_reference_runs",
        "case_count": 15,
        "cases": semantic_entries,
    })
    contract_schemas = {
        "case_spec": file_ref(workspace, suite / "schemas/case_spec.schema.json"),
        "case_gold": file_ref(workspace, suite / "schemas/case_gold.schema.json"),
        "task_confirm": file_ref(workspace, suite / "schemas/task_confirm.schema.json"),
        "physics_profile": file_ref(workspace, suite / "schemas/physics_profile.schema.json"),
    }
    toolchain = {
        name: file_ref(workspace, suite / "tools" / filename)
        for name, filename in {
            "runtime": "runtime.py",
            "physics_diagnostics": "physics_diagnostics.py",
            "matrix_runner": "run_matrix.py",
            "result_aggregator": "aggregate_results.py",
            "strict_preflight": "validate_suite.py",
            "suite_bootstrap": "bootstrap_dev15.py",
            "dev6_bootstrap": "bootstrap_dev6.py",
            "contract_generator": "dev6_contracts.py",
        }.items()
    }
    suite_payload = {
        "schema_version": "simcode4d.dev15.suite.v1",
        "suite_id": "SimCode4D-Dev15-RC1",
        "status": "development_only_not_official_core15",
        "generation_seeds": GENERATION_SEEDS,
        "physics_seeds": PHYSICS_SEEDS,
        "nominal_physics_seed": 401,
        "semantic_freeze": file_ref(workspace, semantic_path),
        "handover_audit": file_ref(workspace, audit_path),
        "contract_schemas": contract_schemas,
        "toolchain": toolchain,
        "cases": registry_cases,
        "replication_note": "Three rows are deterministic rebuild/replay slots unless a generating method consumes the generation seed; identical hashes are disclosed and are not independent generation replicates.",
        "score_scope": "benchmark-owned reference probes validate Dev15 self-consistency only; these are not Code2Worlds method or ablation scores.",
    }
    suite_path = suite / "config/suite.json"
    write_json(suite_path, suite_payload)

    lock_files = [semantic_path, suite_path, audit_path, *[suite / "schemas" / name for name in (
        "case_spec.schema.json", "case_gold.schema.json", "task_confirm.schema.json", "physics_profile.schema.json"
    )]]
    lock_files.extend(workspace / ref["path"] for ref in toolchain.values())
    for entry in registry_cases:
        for label in ("case_spec", "case_gold", "task_confirm", "physics_profile"):
            lock_files.append(workspace / entry[label]["path"])
    write_json(suite / "config/freeze_lock.json", {
        "schema_version": "simcode4d.dev15.freeze-lock.v1",
        "suite_id": suite_payload["suite_id"],
        "files": [file_ref(workspace, path) for path in sorted(lock_files)],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    build(args.workspace.resolve(), args.suite.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
