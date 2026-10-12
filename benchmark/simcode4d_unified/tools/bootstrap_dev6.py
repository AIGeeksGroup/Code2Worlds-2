#!/usr/bin/env python3
"""Generate and hash the six RC3A development case specs and gold files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from dev6_contracts import write_case_contracts


GENERATION_SEEDS = [1103, 2207, 3301]
PHYSICS_SEEDS = [401, 409, 419, 421, 431]
COMMON_BASIS = [
    "source Blender animation and source object hierarchy",
    "retained legacy rollout used only to recover intent, not to tune post-freeze scores",
    "task_semantics_confirmation_2026-09-08.csv development draft",
]

CONTACT_GROUPS = {
    "NAT-01-rock-kick": {"go2_feet": {"geoms": ["FL", "FR", "RL", "RR"]}, "rock": {"bodies": ["stone"]}, "canyon_terrain": {"geoms": ["terrain_*"]}, "go2_base": {"bodies": ["base"]}},
    "NAT-02-hills-turn-stop": {"jackal_wheels": {"geoms": ["jackal_front_left_collision", "jackal_front_right_collision", "jackal_rear_left_collision", "jackal_rear_right_collision"]}, "hill_terrain": {"geoms": ["terrain_hfield"]}, "jackal_chassis": {"geoms": ["jackal_chassis_collision"]}, "route_obstacles": {"geoms": ["prop_concave_large_*"]}},
    "NAT-03-water-land": {"rover_wheels": {"geoms": ["wheel_*_geom"]}, "shore": {"geoms": ["shore_hfield"]}, "rover_hull": {"geoms": ["rov_hull", "rov_pontoon_*"]}},
    "SER-01-medicine": {"gripper": {"geoms": ["gripper_*"]}, "medicine_pouch": {"bodies": ["task_item"]}, "robot_wheels": {"geoms": ["wheel_*_geom", "caster_*_geom"]}, "floor": {"geoms": ["floor"]}, "robot_body": {"geoms": ["robot_base_collision", "robot_body_collision"]}, "counter": {"geoms": ["counter_*"]}},
    "SER-02-airport-luggage": {"luggage_lift": {"geoms": ["lift_fork_*", "clamp_*"]}, "blue_luggage": {"bodies": ["task_item"]}, "robot_wheels": {"geoms": ["wheel_*_geom", "caster_*_geom"]}, "floor": {"geoms": ["floor"]}, "robot_body": {"geoms": ["robot_base_collision", "robot_body_collision"]}, "seats": {"geoms": ["seat_*"]}, "floor_after_lift": {"geoms": ["floor"], "active_when": {"joint": "lift_joint", "minimum_position": 0.20}}},
    "SER-03-hotel-drawer": {"meal_parcel": {"bodies": ["meal_parcel"]}, "payload_drawer": {"bodies": ["upper_drawer"]}, "robot_wheels": {"geoms": ["wheel_*_geom", "caster_*_geom"]}, "floor": {"geoms": ["floor"]}, "robot_body": {"geoms": ["robot_base_collision", "robot_body_collision"]}, "room_door": {"geoms": ["room_door_1208"]}},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_ref(workspace: Path, path: Path) -> dict:
    resolved = path.resolve()
    relative = resolved.relative_to(workspace.resolve())
    return {"path": relative.as_posix(), "sha256": digest(resolved), "bytes": resolved.stat().st_size}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def xml_closure(workspace: Path, root_xml: Path) -> list[dict]:
    found: dict[Path, dict] = {}

    def visit(xml_path: Path) -> None:
        xml_path = xml_path.resolve()
        if xml_path in found:
            return
        found[xml_path] = file_ref(workspace, xml_path)
        tree = ET.parse(xml_path)
        for element in tree.getroot().iter():
            raw = element.get("file")
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = xml_path.parent / candidate
            candidate = candidate.resolve()
            found[candidate] = file_ref(workspace, candidate)
            if element.tag == "include":
                visit(candidate)

    visit(root_xml)
    return [found[key] for key in sorted(found, key=lambda path: str(path))]


def source_trajectory(workspace: Path, scene_id: str) -> tuple[list[list[float]], list[float]]:
    path = workspace / "benchmark/simcode4d_unified/source_reexports" / scene_id / "source_motion.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    positions = [[float(item["root"]["pos"][0]), float(item["root"]["pos"][1])] for item in samples]
    yaws = [float(item["root"]["yaw_rad"]) for item in samples]
    return positions, yaws


def service_executable_trajectory(workspace: Path) -> tuple[list[list[float]], list[float]]:
    path = workspace / "deliverables/final_real8s_physical_2026-09-01/module1/scene_indoor_service/physics_adjusted_action.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    count = len(samples)
    knots = (
        (0, -4.00, -3.80, -0.35),
        (18, -3.35, -3.55, -1.20),
        (34, -3.10, -3.40, -1.5707963267948966),
        (50, -3.10, -3.40, 0.0),
        (65, -3.10, -2.25, 0.0),
        (110, -3.10, 0.72, 0.0),
        (140, -3.10, 0.72, 0.0),
        (195, -3.10, -3.35, 0.0),
        (217, -4.80, -3.35, 1.5707963267948966),
        (239, -4.80, -2.25, 0.0),
    )
    positions = [[0.0, 0.0] for _ in range(count)]
    yaws = [0.0 for _ in range(count)]
    for left, right in zip(knots, knots[1:]):
        start_index, start_x, start_y, start_yaw = left
        end_index, end_x, end_y, end_yaw = right
        for index in range(start_index, end_index + 1):
            u = (index - start_index) / max(end_index - start_index, 1)
            s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
            positions[index] = [start_x + s * (end_x - start_x), start_y + s * (end_y - start_y)]
            yaws[index] = start_yaw + s * (end_yaw - start_yaw)
    return positions, yaws


def nat02_path(workspace: Path) -> list[list[float]]:
    path = workspace / "export_scene3/module2/.ipynb_checkpoints/rollout_log_blender-checkpoint.json"
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    raw = [frame["go2_base_pos"][:2] for frame in frames]
    first = raw[0]
    scaled = [[first[0] + 0.45 * (item[0] - first[0]), first[1] + 0.45 * (item[1] - first[1])] for item in raw]
    return [[float(value) for value in item] for item in scaled]


def perturbations(target_body: str | None, robot_body: str, friction_body: str | None = None) -> dict:
    nominal = {}
    seed_409 = {"body_xy_offset_m": {robot_body: [0.01, -0.01]}, "body_yaw_offset_rad": {robot_body: 0.015}}
    seed_419 = {"body_xy_offset_m": {robot_body: [-0.012, 0.008]}, "body_yaw_offset_rad": {robot_body: -0.02}}
    seed_421 = {"body_xy_offset_m": {robot_body: [0.0, 0.015]}}
    seed_431 = {"body_xy_offset_m": {robot_body: [0.008, 0.012]}, "body_yaw_offset_rad": {robot_body: 0.012}}
    if target_body:
        seed_409["body_mass_scale"] = {target_body: 0.95}
        seed_419["body_mass_scale"] = {target_body: 1.05}
        seed_421["body_mass_scale"] = {target_body: 0.90}
        seed_431["body_mass_scale"] = {target_body: 1.10}
    if friction_body:
        seed_409["body_geom_friction_scale"] = {friction_body: 0.95}
        seed_419["body_geom_friction_scale"] = {friction_body: 1.05}
        seed_421["body_geom_friction_scale"] = {friction_body: 1.08}
        seed_431["body_geom_friction_scale"] = {friction_body: 0.92}
    return {"401": nominal, "409": seed_409, "419": seed_419, "421": seed_421, "431": seed_431}


def integrity_rules(allow_grasp: bool, allow_field: bool) -> dict:
    return {
        "forbid_probe_model_handle": True,
        "forbid_qpos_qvel_write": True,
        "forbid_runtime_model_mutation": True,
        "allow_benchmark_grasp_api": allow_grasp,
        "allow_static_field_adapter": allow_field,
    }


def base_actuator_limits(force_limit: float, yaw_limit: float) -> dict:
    return {
        "base_force_x_ctrl": [-force_limit, force_limit],
        "base_force_y_ctrl": [-force_limit, force_limit],
        "base_force_z_ctrl": [-5500.0, 5500.0],
        "base_roll_ctrl": [-5500.0, 5500.0],
        "base_pitch_ctrl": [-5500.0, 5500.0],
        "base_yaw_ctrl": [-yaw_limit, yaw_limit],
    }


def base_review(note: str) -> dict:
    return {
        "status": "development_frozen",
        "formal_annotator_count": 0,
        "adjudicated": False,
        "thresholds_frozen_before_reference_runs": True,
        "official_use": False,
        "note": note,
    }


def case_payloads(workspace: Path) -> list[dict]:
    service_path, service_yaw = service_executable_trajectory(workspace)
    airport_path, airport_yaw = source_trajectory(workspace, "indoor2_airport")
    hotel_path, hotel_yaw = source_trajectory(workspace, "indoor3_hotel")
    hills_path = nat02_path(workspace)
    return [
        {
            "id": "NAT-01-rock-kick", "world": "nat01-canyon", "domain": "natural_outdoor",
            "family": "dynamic_interaction", "subtype": "contact_dynamic",
            "prompt": "Use the Go2 feet to make the source-bound rock move physically in the uneven canyon, then let it settle.",
            "source": "scene_1.blend", "manifest": "benchmark/simcode4d_unified/assets/nat01_meshes/manifest.json", "lineage": "verified",
            "probe": "probes/nat01_rock_kick.py", "validator": "validators/nat01_rock_kick.py",
            "entities": [
                {"entity_id": "go2", "role": "robot", "mobility": "articulated", "model_binding": "base", "required": True},
                {"entity_id": "rock", "role": "target_object", "mobility": "free", "model_binding": "stone", "required": True,
                 "assembly_members": ["stone_col_*"]},
                {"entity_id": "canyon_terrain", "role": "support", "mobility": "fixed", "model_binding": "terrain_*", "required": True},
            ],
            "bodies": ["base", "stone"],
            "joints": [prefix + suffix + "_joint" for prefix in ("FL", "FR", "RL", "RR") for suffix in ("_hip", "_thigh", "_calf")],
            "limits": {name: ([-23.7, 23.7] if "calf" not in name else [-45.43, 45.43]) for name in
                       ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
                        "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]},
            "waypoints": [],
            "initialization": {"keyframe": "m2_init"},
            "regions": {"goal": {"type": "relative_displacement", "minimum_planar_displacement_m": 0.03}},
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_planar_displacement_m": 0.03, "maximum_terminal_speed_mps": 0.05},
            "dag": {"nodes": ["foot_rock_contact", "rock_displaced", "rock_settled"],
                    "edges": [["foot_rock_contact", "rock_displaced"], ["rock_displaced", "rock_settled"]]},
            "contacts": {"allowed": [["go2_feet", "rock"], ["rock", "canyon_terrain"], ["go2_feet", "canyon_terrain"]],
                         "forbidden": [["go2_base", "rock"]], "penetration_exemptions": []},
            "perturbations": perturbations("stone", "base", "stone"),
            "integrity": integrity_rules(False, False),
            "review_note": "The rock is explicitly bound to the 0.07-scale source boulder named in dynamics_program.json.",
        },
        {
            "id": "NAT-02-hills-turn-stop", "world": "nat02-hills", "domain": "natural_outdoor",
            "family": "obstacle_navigation", "subtype": None,
            "prompt": "Drive the Jackal across the authored hills, complete the authored left turn, enter the goal region, and remain stopped for 0.5 seconds.",
            "source": "scene_3.blend", "manifest": "benchmark/simcode4d_unified/source_reexports/NAT-02-hills-turn-stop-v3/manifest.json", "lineage": "verified",
            "probe": "probes/nat02_hills_turn_stop.py", "validator": "validators/nat02_hills_turn_stop.py",
            "entities": [
                {"entity_id": "jackal", "role": "robot", "mobility": "articulated", "model_binding": "jackal_base", "required": True},
                {"entity_id": "hill_terrain", "role": "support", "mobility": "fixed", "model_binding": "terrain_hfield", "required": True},
                {"entity_id": "route_obstacles", "role": "obstacle", "mobility": "fixed", "model_binding": "prop_*", "required": True},
            ],
            "bodies": ["jackal_base"],
            "joints": ["jackal_front_left_joint", "jackal_front_right_joint", "jackal_rear_left_joint", "jackal_rear_right_joint"],
            "limits": {
                **{name: [-1.0, 1.0] for name in
                   ["jackal_front_left_motor", "jackal_front_right_motor", "jackal_rear_left_motor", "jackal_rear_right_motor"]},
                **base_actuator_limits(5000.0, 650.0),
            },
            "waypoints": hills_path,
            "initialization": {"hfield_vehicle": {
                "body": "jackal_base", "xy": [hills_path[0][0], hills_path[0][1]], "yaw_rad": -3.141592653589793,
                "wheel_xy_m": [[0.53055, 0.76056975], [0.53055, -0.76056975], [-0.53055, 0.76056975], [-0.53055, -0.76056975]],
                "wheel_bodies": ["jackal_front_left_wheel", "jackal_front_right_wheel",
                                 "jackal_rear_left_wheel", "jackal_rear_right_wheel"],
                "wheel_radius_m": 0.3969, "wheel_center_z_m": 0.139725,
                "hfield_geom": "terrain_hfield", "clearance_m": 0.010
            }},
            "regions": {
                "hill_checkpoint": {"type": "aabb", "min_xyz_m": [-7.0, -4.2, -2.0], "max_xyz_m": [-5.2, -2.2, 8.0]},
                "goal": {"type": "aabb", "min_xyz_m": [hills_path[-1][0] - 0.9, hills_path[-1][1] - 0.9, -2.0],
                         "max_xyz_m": [hills_path[-1][0] + 0.9, hills_path[-1][1] + 0.9, 8.0]}
            },
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_heading_change_rad": 0.60, "maximum_terminal_speed_mps": 0.10,
                           "maximum_tilt_rad": 0.70},
            "dag": {"nodes": ["hill_checkpoint", "left_turn", "goal_entry", "stop"],
                    "edges": [["hill_checkpoint", "left_turn"], ["left_turn", "goal_entry"], ["goal_entry", "stop"]]},
            "contacts": {"allowed": [["jackal_wheels", "hill_terrain"]],
                         "forbidden": [["jackal_chassis", "route_obstacles"]], "penetration_exemptions": []},
            "perturbations": perturbations(None, "jackal_base", "jackal_base"),
            "integrity": integrity_rules(False, False),
            "review_note": "Left turn is mandatory because it is explicit in both the prompt and retained program.",
        },
        {
            "id": "NAT-03-water-land", "world": "nat03-amphibious", "domain": "natural_outdoor",
            "family": "dynamic_interaction", "subtype": "cross_medium",
            "prompt": "Move the amphibious rover from the declared static water field, contact the bank, reach dry land, and stop in the land goal.",
            "source": "deliverables/final_scene_revision_i04_2026-08-16/blends/scene10_amphibious_final_i04.blend",
            "manifest": "benchmark/simcode4d_unified/source_reexports/scene10_amphibious/physical_manifest.json",
            "lineage": "verified", "probe": "probes/nat03_water_land.py", "validator": "validators/nat03_water_land.py",
            "entities": [
                {"entity_id": "amphibious_rover", "role": "robot", "mobility": "articulated", "model_binding": "robot", "required": True},
                {"entity_id": "shore", "role": "support", "mobility": "fixed", "model_binding": "shore_hfield", "required": True},
                {"entity_id": "water_field", "role": "other", "mobility": "field", "model_binding": "static_fluid:water", "required": True},
                {"entity_id": "water_surface", "role": "other", "mobility": "visual_only", "model_binding": "water_surface_visual", "required": False},
            ],
            "bodies": ["robot"], "joints": ["shore_x", "shore_z", "shore_pitch"] +
                       [kind + "_" + wheel + "_joint" for wheel in ("fl", "fr", "rl", "rr") for kind in ("steer", "wheel")],
            "limits": {
                "propeller_x_ctrl": [-2800.0, 2800.0], "heave_ctrl": [-1300.0, 1300.0], "pitch_ctrl": [-900.0, 900.0],
                **{("steer_" + wheel + "_ctrl"): [-1.35, 1.35] for wheel in ("fl", "fr", "rl", "rr")},
                **{("wheel_" + wheel + "_ctrl"): [-12.0, 12.0] for wheel in ("fl", "fr", "rl", "rr")}
            },
            "waypoints": [], "initialization": {"joint_positions": {
                "shore_x": -5.4, "shore_z": -0.1848446578, "shore_pitch": -0.0088379382
            }},
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [3.25, -1.0, 0.55], "max_xyz_m": [4.05, 1.0, 1.55]}},
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_water_submergence": 0.50, "maximum_terminal_submergence": 0.05,
                           "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["water_state", "bank_contact", "dry_land_state", "goal_stop"],
                    "edges": [["water_state", "bank_contact"], ["bank_contact", "dry_land_state"], ["dry_land_state", "goal_stop"]]},
            "contacts": {"allowed": [["rover_wheels", "shore"]], "forbidden": [["rover_hull", "shore"]], "penetration_exemptions": []},
            "perturbations": perturbations(None, "robot", "robot"),
            "integrity": integrity_rules(False, True),
            "static_field": {"type": "static_fluid", "body": "robot", "surface_z_m": 0.0,
                             "immersion_span_m": 0.84, "density_kg_m3": 1000.0,
                             "displaced_volume_m3": 0.094, "drag_x": 88.0, "drag_z": 115.0, "pitch_drag": 145.0},
            "review_note": "Water surface is visual-only; buoyancy comes from the M1-owned static fluid descriptor.",
        },
        {
            "id": "SER-01-medicine", "world": "ser01-clinic", "domain": "service",
            "family": "pick_and_place", "subtype": None,
            "prompt": "Approach, legally grasp, and deliver the medicine pouch to the authored delivery region while retaining a stable grasp.",
            "source": "deliverables/final_scene_revision_i04_2026-08-16/blends/scene_indoor_service_final_i04.blend",
            "manifest": "benchmark/simcode4d_unified/source_reexports/scene_indoor_service/physical_manifest.json",
            "lineage": "verified", "probe": "probes/ser01_medicine.py", "validator": "validators/ser01_medicine.py",
            "entities": [
                {"entity_id": "service_robot", "role": "robot", "mobility": "articulated", "model_binding": "robot", "required": True},
                {"entity_id": "medicine_pouch", "role": "target_object", "mobility": "free", "model_binding": "task_item", "required": True,
                 "assembly_members": ["ServiceTaskItem_MedicinePouch", "ServiceTaskItem_MedicinePouch_Label"]},
                {"entity_id": "source_counter", "role": "support", "mobility": "fixed", "model_binding": "counter_top_left", "required": True},
            ],
            "bodies": ["robot", "task_item", "grasp_frame"], "joints": ["wheel_left_joint", "wheel_right_joint", "lift_joint", "arm_joint"],
            "limits": {"wheel_left_ctrl": [-18.0, 18.0], "wheel_right_ctrl": [-18.0, 18.0],
                       "lift_ctrl": [0.0, 0.42], "arm_ctrl": [0.0, 0.46],
                       **base_actuator_limits(900.0, 520.0),
            },
            "reference_yaw": service_yaw,
            "waypoints": service_path, "grasp_api": {"equality": "task_grasp", "body_a": "task_item",
                                                    "body_b": "grasp_frame", "maximum_distance_m": 0.16,
                                                    "contact_required": False},
            "initialization": {"free_body_poses": {
                "robot": {"quaternion_wxyz": [0.9847265389, 0.0, 0.0, -0.1741081342]}
            }},
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [-5.15, -1.32, 1.15], "max_xyz_m": [-4.45, -0.62, 1.75]}},
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_transport_m": 2.5, "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["approach", "grasp", "transport", "goal_hold"],
                    "edges": [["approach", "grasp"], ["grasp", "transport"], ["transport", "goal_hold"]]},
            "contacts": {"allowed": [["gripper", "medicine_pouch"], ["robot_wheels", "floor"]],
                         "forbidden": [["robot_body", "counter"], ["medicine_pouch", "floor"]], "penetration_exemptions": []},
            "perturbations": perturbations("task_item", "robot", "robot"),
            "integrity": integrity_rules(True, False),
            "review_note": "The source animation retains the pouch at the grasp anchor; release is therefore not required in this development case.",
        },
        {
            "id": "SER-02-airport-luggage", "world": "ser02-airport", "domain": "service",
            "family": "pick_and_place", "subtype": None,
            "prompt": "Approach the blue luggage assembly, legally grasp and lift it, then deliver it to the authored airport goal while retaining a stable grasp.",
            "source": "deliverables/final_scene_revision_i04_2026-08-16/blends/indoor2_airport_final_i04.blend",
            "manifest": "benchmark/simcode4d_unified/source_reexports/indoor2_airport/physical_manifest.json",
            "lineage": "verified", "probe": "probes/ser02_airport_luggage.py", "validator": "validators/ser02_airport_luggage.py",
            "entities": [
                {"entity_id": "airport_robot", "role": "robot", "mobility": "articulated", "model_binding": "robot", "required": True},
                {"entity_id": "blue_luggage", "role": "target_object", "mobility": "free", "model_binding": "task_item", "required": True,
                 "assembly_members": ["LuggageFactory_blue_root/**"]},
                {"entity_id": "airport_floor", "role": "support", "mobility": "fixed", "model_binding": "floor", "required": True},
            ],
            "bodies": ["robot", "task_item", "grasp_frame"], "joints": ["wheel_left_joint", "wheel_right_joint", "lift_joint"],
            "limits": {"wheel_left_ctrl": [-20.0, 20.0], "wheel_right_ctrl": [-20.0, 20.0],
                       "lift_ctrl": [0.0, 0.36],
                       **base_actuator_limits(900.0, 520.0),
            },
            "reference_yaw": airport_yaw,
            "waypoints": airport_path, "grasp_api": {"equality": "task_grasp", "body_a": "task_item",
                                                    "body_b": "grasp_frame", "maximum_distance_m": 0.06,
                                                    "contact_required": False},
            "initialization": {},
            "regions": {"goal": {"type": "aabb", "min_xyz_m": [4.35, 3.10, 0.40], "max_xyz_m": [5.25, 4.00, 1.10]}},
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_lift_m": 0.25, "minimum_transport_m": 5.0,
                           "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["approach", "grasp", "lift", "transport", "goal_hold"],
                    "edges": [["approach", "grasp"], ["grasp", "lift"], ["lift", "transport"], ["transport", "goal_hold"]]},
            "contacts": {"allowed": [["luggage_lift", "blue_luggage"], ["robot_wheels", "floor"]],
                         "forbidden": [["robot_body", "seats"], ["blue_luggage", "floor_after_lift"]], "penetration_exemptions": []},
            "perturbations": perturbations("task_item", "robot", "robot"),
            "integrity": integrity_rules(True, False),
            "review_note": "The entire blue Empty-root assembly is one dynamic target. Source animation retains it; release is not required.",
        },
        {
            "id": "SER-03-hotel-drawer", "world": "ser03-hotel", "domain": "service",
            "family": "dynamic_interaction", "subtype": "articulated",
            "prompt": "Navigate the hotel robot to the terminal region, stop, open its payload drawer, and keep the meal parcel supported for 0.5 seconds.",
            "source": "deliverables/final_scene_revision_i04_2026-08-16/blends/indoor3_hotel_final_i04.blend",
            "manifest": "benchmark/simcode4d_unified/source_reexports/indoor3_hotel/physical_manifest.json",
            "lineage": "verified", "probe": "probes/ser03_hotel_drawer.py", "validator": "validators/ser03_hotel_drawer.py",
            "entities": [
                {"entity_id": "hotel_robot", "role": "robot", "mobility": "articulated", "model_binding": "robot", "required": True},
                {"entity_id": "payload_drawer", "role": "articulated_object", "mobility": "articulated", "model_binding": "upper_drawer", "required": True},
                {"entity_id": "meal_parcel", "role": "target_object", "mobility": "free", "model_binding": "meal_parcel", "required": True},
                {"entity_id": "hotel_floor", "role": "support", "mobility": "fixed", "model_binding": "floor", "required": True},
            ],
            "bodies": ["robot", "upper_drawer", "meal_parcel"], "joints": ["wheel_left_joint", "wheel_right_joint", "drawer_joint"],
            "limits": {"wheel_left_ctrl": [-20.0, 20.0], "wheel_right_ctrl": [-20.0, 20.0],
                       "drawer_ctrl": [0.0, 0.30],
                       **base_actuator_limits(900.0, 520.0),
            },
            "reference_yaw": hotel_yaw,
            "waypoints": hotel_path,
            "initialization": {},
            "regions": {"robot_goal": {"type": "aabb", "min_xyz_m": [5.15, 4.10, -0.1], "max_xyz_m": [6.15, 5.10, 1.0]}},
            "thresholds": {"illegal_penetration_m": 0.001, "terminal_hold_s": 0.5,
                           "minimum_drawer_extension_m": 0.27, "minimum_parcel_height_m": 0.85,
                           "maximum_parcel_drawer_z_offset_m": 0.20, "maximum_terminal_speed_mps": 0.10},
            "dag": {"nodes": ["robot_goal_arrival", "drawer_open", "parcel_supported"],
                    "edges": [["robot_goal_arrival", "drawer_open"], ["drawer_open", "parcel_supported"]]},
            "contacts": {"allowed": [["meal_parcel", "payload_drawer"], ["robot_wheels", "floor"]],
                         "forbidden": [["robot_body", "room_door"], ["meal_parcel", "floor"]], "penetration_exemptions": []},
            "perturbations": perturbations("meal_parcel", "robot", "robot"),
            "integrity": integrity_rules(False, False),
            "review_note": "This is the robot's own payload drawer; opening uses its declared actuator, not a remote equality unlock.",
        },
    ]


def build(workspace: Path, suite: Path) -> None:
    cases = case_payloads(workspace)
    registry_cases = []
    semantic_entries = []
    for item in cases:
        case_id = item["id"]
        contact_policy = {**item["contacts"], "groups": CONTACT_GROUPS[case_id]}
        artifact_dir = suite / "artifacts" / case_id
        model_path = artifact_dir / "m1_world.xml"
        overlay_path = artifact_dir / "m2_overlay.json"
        closure_path = artifact_dir / "dependency_manifest.json"
        closure = {
            "schema_version": "simcode4d.rc3a-dev1.dependency-manifest",
            "case_id": case_id,
            "files": xml_closure(workspace, model_path),
        }
        write_json(closure_path, closure)
        probe_path = suite / item["probe"]
        validator_path = suite / item["validator"]
        spec = {
            "schema_version": "simcode4d.rc3a-dev1.case-spec",
            "case_id": case_id,
            "world_id": item["world"],
            "domain": item["domain"],
            "split": "development",
            "task_family": item["family"],
            "dynamic_subtype": item["subtype"],
            "prompt": item["prompt"],
            "source": file_ref(workspace, workspace / item["source"]),
            "source_manifest": file_ref(workspace, workspace / item["manifest"]),
            "source_lineage_status": item["lineage"],
            "artifacts": {
                "m1_base": file_ref(workspace, model_path),
                "m2_overlay": file_ref(workspace, overlay_path),
                "runtime_model": file_ref(workspace, model_path),
                "dependency_manifest": file_ref(workspace, closure_path),
            },
            "entities": item["entities"],
            "observation": {"bodies": item["bodies"], "joints": item["joints"]},
            "probe": {
                "owner": "benchmark", "path": item["probe"], "sha256": digest(probe_path),
                "support_modules": [
                    {"path": "probes/probe_api.py", "sha256": digest(suite / "probes/probe_api.py")},
                    {"path": "probes/base_control.py", "sha256": digest(suite / "probes/base_control.py")},
                ],
                "episode_duration_s": 8.0, "log_hz": 30,
                "actuator_limits": item["limits"], "waypoints_xy": item["waypoints"],
                **({"reference_yaw_rad": item["reference_yaw"]} if "reference_yaw" in item else {}),
                **({"grasp_api": item["grasp_api"]} if "grasp_api" in item else {}),
            },
            "validator": {
                "owner": "benchmark", "path": item["validator"], "sha256": digest(validator_path),
                "support_modules": [{"path": "validators/common.py", "sha256": digest(suite / "validators/common.py")}],
            },
            "simulator": {"name": "mujoco", "version": mujoco.__version__, "timestep_s": 0.001},
            **({"static_field": item["static_field"]} if "static_field" in item else {}),
            "ownership": {
                "m1": "immutable geometry, physical properties, canonical initial state, and static fields",
                "m2": "task-independent source dynamics overlay bound to the M1 hash; NoOp in Dev6",
                "benchmark_overlay": "robot actuator probe, perturbation, grasp API, validator, and records",
            },
            "semantic_review": {
                "status": "development_frozen", "formal_double_annotation": False,
                "frozen_before_runs": True, "basis": COMMON_BASIS, "official_admission": False,
            },
        }
        spec_path = suite / "cases" / case_id / "case_spec.json"
        write_json(spec_path, spec)
        gold = {
            "schema_version": "simcode4d.rc3a-dev1.case-gold",
            "case_id": case_id,
            "case_spec_sha256": digest(spec_path),
            "gold_visibility": "development_private_placeholder",
            "initialization": item["initialization"],
            "regions": item["regions"],
            "thresholds": item["thresholds"],
            "event_dag": item["dag"],
            "contact_policy": contact_policy,
            "perturbations": item["perturbations"],
            "integrity_rules": item["integrity"],
            "review": base_review(item["review_note"]),
        }
        gold_path = suite / "cases" / case_id / "case_gold.json"
        write_json(gold_path, gold)
        task_confirm_path, physics_profile_path = write_case_contracts(
            workspace, suite, spec_path, gold_path
        )
        registry_cases.append({
            "case_id": case_id,
            "case_spec": file_ref(workspace, spec_path),
            "case_gold": file_ref(workspace, gold_path),
            "task_confirm": file_ref(workspace, task_confirm_path),
            "physics_profile": file_ref(workspace, physics_profile_path),
            "world_id": item["world"],
            "task_family": item["family"],
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

    semantic_path = suite / "config" / "semantic_freeze_v1.json"
    semantic_payload = {
        "schema_version": "simcode4d.rc3a-dev1.semantic-freeze",
        "status": "development_frozen_before_reference_runs",
        "cases": semantic_entries,
    }
    write_json(semantic_path, semantic_payload)
    suite_payload = {
        "schema_version": "simcode4d.rc3a-dev1.suite",
        "suite_id": "SimCode4D-Dev6-RC3A",
        "status": "development_only_not_official_core15",
        "generation_seeds": GENERATION_SEEDS,
        "physics_seeds": PHYSICS_SEEDS,
        "nominal_physics_seed": 401,
        "semantic_freeze": file_ref(workspace, semantic_path),
        "contract_schemas": {
            "task_confirm": file_ref(workspace, suite / "schemas/task_confirm.schema.json"),
            "physics_profile": file_ref(workspace, suite / "schemas/physics_profile.schema.json"),
        },
        "cases": registry_cases,
        "replication_note": "Three rows are deterministic rebuild/replay slots unless a generating method consumes the generation seed; identical hashes are disclosed and are not independent generation replicates.",
    }
    suite_path = suite / "config" / "suite.json"
    write_json(suite_path, suite_payload)

    lock_files = [
        semantic_path,
        suite_path,
        suite / "schemas/task_confirm.schema.json",
        suite / "schemas/physics_profile.schema.json",
    ]
    for entry in registry_cases:
        lock_files.append(workspace / entry["case_spec"]["path"])
        lock_files.append(workspace / entry["case_gold"]["path"])
        lock_files.append(workspace / entry["task_confirm"]["path"])
        lock_files.append(workspace / entry["physics_profile"]["path"])
    lock = {
        "schema_version": "simcode4d.rc3a-dev1.freeze-lock",
        "suite_id": suite_payload["suite_id"],
        "files": [file_ref(workspace, path) for path in sorted(lock_files)],
    }
    write_json(suite / "config" / "freeze_lock.json", lock)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    build(args.workspace.resolve(), args.suite.resolve())


if __name__ == "__main__":
    main()
