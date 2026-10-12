#!/usr/bin/env python3
"""Build auditable task and physics contracts for the frozen Dev15 cases."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


ROBOT_NAMES = {
    "NAT-01-rock-kick": "Unitree Go2 quadruped",
    "NAT-02-hills-turn-stop": "Clearpath Jackal UGV",
    "NAT-03-water-land": "four-wheel amphibious rover",
    "IND-01-vacuum-living-room": "mobile vacuum",
    "IND-02-go2-living-room": "Unitree Go2 quadruped",
    "IND-03-vacuum-west": "mobile vacuum",
    "INS-01-agv-aisle": "four-wheel AGV",
    "INS-02-agv-cargo": "four-wheel AGV with welded cargo",
    "INS-03-weld": "UR5e welding arm",
    "SER-01-medicine": "differential-drive service robot with lift and arm",
    "SER-02-airport-luggage": "differential-drive airport luggage robot with lift",
    "SER-03-hotel-drawer": "differential-drive hotel service robot with payload drawer",
    "URB-01-agv-city": "four-wheel AGV",
    "URB-02-intersection": "Unitree Go2 quadruped",
    "URB-03-pole-blocking": "Unitree Go2 quadruped",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(workspace: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(workspace.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


def _number(value: Any) -> float:
    value = float(value)
    if abs(value) < 5e-15:
        return 0.0
    return float(f"{value:.12g}")


def _vector(values: Any) -> list[float]:
    return [_number(value) for value in np.asarray(values).reshape(-1)]


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int, prefix: str) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"__unnamed_{prefix}_{index}"


def _enum_name(enum_type: Any, value: Any) -> str:
    return enum_type(int(value)).name


def _joint_widths(joint_type: int) -> tuple[int, int]:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def build_task_confirm(spec: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    case_id = spec["case_id"]
    robot_entity = next(entity for entity in spec["entities"] if entity["role"] == "robot")
    return {
        "schema_version": "simcode4d.dev15.task-confirm.v1",
        "case_id": case_id,
        "world_id": spec["world_id"],
        "canonical_source": {
            **spec["source"],
            "lineage_status": spec["source_lineage_status"],
        },
        "robot": {
            "name": ROBOT_NAMES[case_id],
            "entity_id": robot_entity["entity_id"],
            "model_binding": robot_entity["model_binding"],
            "mobility": robot_entity["mobility"],
        },
        "task": spec["prompt"],
        "task_family": spec["task_family"],
        "dynamic_subtype": spec["dynamic_subtype"],
        "start_state": gold["initialization"],
        "goal": gold["regions"],
        "required_event_order": gold["event_dag"],
        "allowed_contacts": gold["contact_policy"]["allowed"],
        "forbidden_contacts": gold["contact_policy"]["forbidden"],
        "contact_group_bindings": gold["contact_policy"]["groups"],
        "timeout_s": spec["probe"]["episode_duration_s"],
        "success_tolerance": gold["thresholds"],
        "review": {
            "status": "benchmark_authored_development_frozen",
            "basis": spec["semantic_review"]["basis"],
            "frozen_before_reference_runs": gold["review"]["thresholds_frozen_before_reference_runs"],
            "formal_annotator_count": gold["review"]["formal_annotator_count"],
            "official_core15_use": False,
            "note": gold["review"].get("note", ""),
        },
    }


def _model_options(model: mujoco.MjModel) -> dict[str, Any]:
    option = model.opt
    optional_scalars = (
        "ls_iterations", "noslip_iterations", "ccd_iterations", "ls_tolerance",
        "noslip_tolerance", "ccd_tolerance", "apirate", "density", "viscosity",
        "o_margin",
    )
    result: dict[str, Any] = {
        "timestep_s": _number(option.timestep),
        "gravity_mps2": _vector(option.gravity),
        "wind_mps": _vector(option.wind),
        "magnetic_tesla": _vector(option.magnetic),
        "integrator": _enum_name(mujoco.mjtIntegrator, option.integrator),
        "solver": _enum_name(mujoco.mjtSolver, option.solver),
        "friction_cone": _enum_name(mujoco.mjtCone, option.cone),
        "jacobian": _enum_name(mujoco.mjtJacobian, option.jacobian),
        "iterations": int(option.iterations),
        "tolerance": _number(option.tolerance),
        "impratio": _number(option.impratio),
    }
    for name in optional_scalars:
        if hasattr(option, name):
            value = getattr(option, name)
            result[name] = int(value) if "iterations" in name else _number(value)
    return result


def _body_records(model: mujoco.MjModel) -> list[dict[str, Any]]:
    records = []
    for body_id in range(model.nbody):
        parent_id = int(model.body_parentid[body_id])
        mass = _number(model.body_mass[body_id])
        inertia = _vector(model.body_inertia[body_id])
        records.append({
            "id": body_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"),
            "parent": None if body_id == 0 else _name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id, "body"),
            "mass_kg": mass,
            "diagonal_inertia_kg_m2": inertia,
            "inertial_position_m": _vector(model.body_ipos[body_id]),
            "inertial_quaternion_wxyz": _vector(model.body_iquat[body_id]),
            "joint_count": int(model.body_jntnum[body_id]),
            "mocap_id": int(model.body_mocapid[body_id]),
            "dynamic": bool(mass > 0.0 or int(model.body_jntnum[body_id]) > 0),
        })
    return records


def _joint_records(model: mujoco.MjModel) -> list[dict[str, Any]]:
    records = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qpos_width, dof_width = _joint_widths(joint_type)
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        records.append({
            "id": joint_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint"),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id]), "body"),
            "type": _enum_name(mujoco.mjtJoint, joint_type),
            "qpos_address": qpos_address,
            "qpos_width": qpos_width,
            "dof_address": dof_address,
            "dof_width": dof_width,
            "axis": _vector(model.jnt_axis[joint_id]),
            "position_m": _vector(model.jnt_pos[joint_id]),
            "limited": bool(model.jnt_limited[joint_id]),
            "range": _vector(model.jnt_range[joint_id]),
            "margin": _number(model.jnt_margin[joint_id]),
            "stiffness": _number(model.jnt_stiffness[joint_id]),
            "damping": _vector(model.dof_damping[dof_address:dof_address + dof_width]),
            "armature": _vector(model.dof_armature[dof_address:dof_address + dof_width]),
            "frictionloss": _vector(model.dof_frictionloss[dof_address:dof_address + dof_width]),
        })
    return records


def _actuator_records(model: mujoco.MjModel) -> list[dict[str, Any]]:
    records = []
    for actuator_id in range(model.nu):
        transmission_type = int(model.actuator_trntype[actuator_id])
        target_id = int(model.actuator_trnid[actuator_id, 0])
        target_name = None
        if transmission_type in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT) and target_id >= 0:
            target_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id, "joint")
        records.append({
            "id": actuator_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id, "actuator"),
            "transmission_type": _enum_name(mujoco.mjtTrn, transmission_type),
            "transmission_ids": [int(value) for value in model.actuator_trnid[actuator_id]],
            "transmission_target": target_name,
            "dynamics_type": _enum_name(mujoco.mjtDyn, model.actuator_dyntype[actuator_id]),
            "gain_type": _enum_name(mujoco.mjtGain, model.actuator_gaintype[actuator_id]),
            "bias_type": _enum_name(mujoco.mjtBias, model.actuator_biastype[actuator_id]),
            "gear": _vector(model.actuator_gear[actuator_id]),
            "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
            "control_range": _vector(model.actuator_ctrlrange[actuator_id]),
            "force_limited": bool(model.actuator_forcelimited[actuator_id]),
            "force_range_N_or_Nm": _vector(model.actuator_forcerange[actuator_id]),
            "length_range": _vector(model.actuator_lengthrange[actuator_id]),
            "gain_parameters": _vector(model.actuator_gainprm[actuator_id]),
            "bias_parameters": _vector(model.actuator_biasprm[actuator_id]),
            "dynamics_parameters": _vector(model.actuator_dynprm[actuator_id]),
        })
    return records


def _collision_records(model: mujoco.MjModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_ids: dict[tuple[Any, ...], str] = {}
    classes: list[dict[str, Any]] = []
    geoms: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if contype == 0 and conaffinity == 0:
            continue
        values = (
            tuple(_vector(model.geom_friction[geom_id])),
            int(model.geom_condim[geom_id]),
            int(model.geom_priority[geom_id]),
            tuple(_vector(model.geom_solref[geom_id])),
            tuple(_vector(model.geom_solimp[geom_id])),
            _number(model.geom_margin[geom_id]),
            _number(model.geom_gap[geom_id]),
            contype,
            conaffinity,
        )
        if values not in class_ids:
            class_id = f"contact_material_{len(class_ids) + 1:03d}"
            class_ids[values] = class_id
            classes.append({
                "id": class_id,
                "friction": {
                    "sliding": values[0][0],
                    "torsional": values[0][1],
                    "rolling": values[0][2],
                },
                "contact_dimension": values[1],
                "priority": values[2],
                "solver_reference": list(values[3]),
                "solver_impedance": list(values[4]),
                "margin_m": values[5],
                "gap_m": values[6],
                "contact_type_mask": values[7],
                "contact_affinity_mask": values[8],
            })
        body_id = int(model.geom_bodyid[geom_id])
        record = {
            "id": geom_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom"),
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"),
            "type": _enum_name(mujoco.mjtGeom, model.geom_type[geom_id]),
            "size": _vector(model.geom_size[geom_id]),
            "position_m": _vector(model.geom_pos[geom_id]),
            "quaternion_wxyz": _vector(model.geom_quat[geom_id]),
            "contact_material": class_ids[values],
            "group": int(model.geom_group[geom_id]),
        }
        if hasattr(model, "geom_mass"):
            record["mass_kg"] = _number(model.geom_mass[geom_id])
        if hasattr(model, "geom_density"):
            record["density_kg_m3"] = _number(model.geom_density[geom_id])
        geoms.append(record)
    return classes, geoms


def _resolved_groups(model: mujoco.MjModel, contact_policy: dict[str, Any]) -> dict[str, Any]:
    geom_names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, index, "geom") for index in range(model.ngeom)]
    body_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, index, "body") for index in range(model.nbody)]
    resolved = {}
    for label, binding in contact_policy["groups"].items():
        resolved[label] = {
            "geoms": sorted({name for pattern in binding.get("geoms", []) for name in geom_names if fnmatch.fnmatchcase(name, pattern)}),
            "bodies": sorted({name for pattern in binding.get("bodies", []) for name in body_names if fnmatch.fnmatchcase(name, pattern)}),
            **({"active_when": binding["active_when"]} if "active_when" in binding else {}),
        }
    return resolved


def _equality_records(model: mujoco.MjModel) -> list[dict[str, Any]]:
    records = []
    for equality_id in range(model.neq):
        records.append({
            "id": equality_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id, "equality"),
            "type": _enum_name(mujoco.mjtEq, model.eq_type[equality_id]),
            "object_ids": [int(model.eq_obj1id[equality_id]), int(model.eq_obj2id[equality_id])],
            "active_at_reset": bool(model.eq_active0[equality_id]),
            "solver_reference": _vector(model.eq_solref[equality_id]),
            "solver_impedance": _vector(model.eq_solimp[equality_id]),
        })
    return records


def build_physics_profile(
    workspace: Path,
    spec_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    gold = load_json(gold_path)
    model_path = workspace / spec["artifacts"]["runtime_model"]["path"]
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    material_classes, collision_geoms = _collision_records(model)
    timestep = float(model.opt.timestep)
    body_records = _body_records(model)
    nonphysical_dynamic_bodies = [
        item["name"] for item in body_records
        if item["dynamic"] and (
            item["mass_kg"] <= 0.0
            or any(value <= 0.0 for value in item["diagonal_inertia_kg_m2"])
        )
    ]
    return {
        "schema_version": "simcode4d.dev15.physics-profile.v1",
        "case_id": spec["case_id"],
        "world_id": spec["world_id"],
        "runtime_model": file_ref(workspace, model_path),
        "case_spec_sha256": sha256_file(spec_path),
        "case_gold_sha256": sha256_file(gold_path),
        "provenance": {
            "parameter_status": "exact_from_frozen_runtime_model",
            "semantic_contact_status": "exact_from_frozen_case_gold",
            "estimated_parameter_count": 0,
            "note": "This profile indexes the frozen Dev15 runtime; unavailable pre-export authoring values are explicitly identified by provenance records.",
        },
        "simulator": {
            "name": "mujoco",
            "runtime_version": mujoco.__version__,
            "model_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0) or "unnamed_model",
            "dimensions": {
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "nbody": model.nbody,
                "njoint": model.njnt,
                "ngeom": model.ngeom,
                "nequality": model.neq,
            },
            "options": _model_options(model),
        },
        "control_loop": {
            "physics_hz": _number(1.0 / timestep),
            "benchmark_probe_command_hz": _number(1.0 / timestep),
            "record_hz": spec["probe"]["log_hz"],
            "command_interface": "declared MuJoCo actuators only; probe has no qpos/qvel/model write access",
            "actuator_limits": spec["probe"]["actuator_limits"],
        },
        "bodies": body_records,
        "joints": _joint_records(model),
        "actuators": _actuator_records(model),
        "equalities": _equality_records(model),
        "contact_material_classes": material_classes,
        "collision_geoms": collision_geoms,
        "semantic_contact_policy": gold["contact_policy"],
        "resolved_contact_groups": _resolved_groups(model, gold["contact_policy"]),
        "static_field": spec.get("static_field"),
        "physics_seed_perturbations": gold["perturbations"],
        "success_and_safety_thresholds": gold["thresholds"],
        "diagnostic_contract": {
            "mechanics": {
                "channels": [
                    "finite generalized state", "positive body mass and principal inertia",
                    "normal and tangential contact force", "penetration depth",
                    "generalized force-balance residual", "constraint-force solver residual (diagnostic)",
                    "kinetic and potential energy",
                ],
                "nonphysical_dynamic_bodies": nonphysical_dynamic_bodies,
                "force_balance_relative_tolerance": 1e-5,
            },
            "kinematics": {
                "channels": [
                    "body pose and twist", "quaternion norm", "joint position and velocity",
                    "finite-difference acceleration and jerk", "joint-limit dwell",
                ],
                "quaternion_norm_tolerance": 1e-6,
                "joint_limit_tolerance": 1e-4,
                "violation_minimum_dwell_s": 0.01,
            },
            "friction": {
                "channels": [
                    "sliding/torsional/rolling coefficients", "tangential contact force",
                    "Coulomb-cone utilization", "contact-point tangential slip speed",
                ],
                "cone_utilization_warning": 1.05,
                "cone_utilization_failure": 1.25,
                "failure_minimum_dwell_s": 0.01,
                "slip_speed_reporting_threshold_mps": 0.02,
                "slip_is_diagnostic_not_global_failure": True,
            },
        },
    }


def write_case_contracts(
    workspace: Path,
    suite: Path,
    spec_path: Path,
    gold_path: Path,
) -> tuple[Path, Path]:
    spec = load_json(spec_path)
    gold = load_json(gold_path)
    case_dir = suite / "cases" / spec["case_id"]
    task_path = case_dir / "task_confirm.yaml"
    physics_path = case_dir / "physics_profile.json"
    write_yaml(task_path, build_task_confirm(spec, gold))
    write_json(physics_path, build_physics_profile(workspace, spec_path, gold_path))
    return task_path, physics_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="*")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    suite = args.suite.resolve()
    registry = load_json(suite / "config/suite.json")
    selected = set(args.case_ids or [item["case_id"] for item in registry["cases"]])
    for item in registry["cases"]:
        if item["case_id"] not in selected:
            continue
        spec_path = workspace / item["case_spec"]["path"]
        gold_path = workspace / item["case_gold"]["path"]
        write_case_contracts(workspace, suite, spec_path, gold_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
