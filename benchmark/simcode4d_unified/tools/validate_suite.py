#!/usr/bin/env python3
"""Strict preflight for the unified SimCode4D Dev15 machine contract."""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import sys

import jsonschema
import mujoco
import yaml

from dev6_contracts import build_physics_profile, build_task_confirm


BANNED_PROBE_ATTRIBUTES = {
    "qpos", "qvel", "xfrc_applied", "eq_active", "geom_friction",
    "geom_contype", "geom_conaffinity", "body_mass", "body_inertia",
}
BANNED_PROBE_IMPORTS = {"mujoco", "numpy", "os", "pathlib", "subprocess", "sys"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_ref(workspace: Path, ref: dict, label: str, errors: list[str]) -> Path | None:
    path = (workspace / ref["path"]).resolve()
    try:
        path.relative_to(workspace.resolve())
    except ValueError:
        errors.append(label + ": path escapes workspace")
        return None
    if not path.is_file():
        errors.append(label + ": missing " + ref["path"])
        return None
    if path.stat().st_size != ref["bytes"]:
        errors.append(label + ": byte count mismatch")
    if sha256_file(path) != ref["sha256"]:
        errors.append(label + ": sha256 mismatch")
    return path


def schema_errors(instance: dict, schema: dict, label: str) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [
        label + ":" + "/".join(str(part) for part in error.absolute_path) + ": " + error.message
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    ]


def audit_probe(path: Path) -> list[str]:
    errors = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BANNED_PROBE_IMPORTS:
                    errors.append(path.name + ": banned import " + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in BANNED_PROBE_IMPORTS:
                errors.append(path.name + ": banned import " + str(node.module))
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_PROBE_ATTRIBUTES:
            errors.append(path.name + ": banned state/model attribute " + node.attr)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--allow-pending-lineage", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    suite = args.suite.resolve()
    errors: list[str] = []
    spec_schema = load_json(suite / "schemas/case_spec.schema.json")
    gold_schema = load_json(suite / "schemas/case_gold.schema.json")
    task_schema = load_json(suite / "schemas/task_confirm.schema.json")
    physics_schema = load_json(suite / "schemas/physics_profile.schema.json")
    registry = load_json(suite / "config/suite.json")
    lock = load_json(suite / "config/freeze_lock.json")

    if registry["generation_seeds"] != [1103, 2207, 3301]:
        errors.append("suite: unexpected generation seed grid")
    if registry["physics_seeds"] != [401, 409, 419, 421, 431]:
        errors.append("suite: unexpected physics seed grid")
    if len(registry["cases"]) != 15:
        errors.append("suite: expected exactly fifteen development cases")

    expected_toolchain = {
        "runtime", "physics_diagnostics", "matrix_runner",
        "result_aggregator", "strict_preflight", "suite_bootstrap",
        "dev6_bootstrap", "contract_generator", "episode_worker",
    }
    if set(registry.get("toolchain", {})) != expected_toolchain:
        errors.append("suite: incomplete Dev15 toolchain registry")
    else:
        for label, ref in registry["toolchain"].items():
            verify_ref(workspace, ref, "suite.toolchain." + label, errors)

    verify_ref(workspace, registry["semantic_freeze"], "suite.semantic_freeze", errors)
    for schema_name in ("case_spec", "case_gold", "task_confirm", "physics_profile"):
        if schema_name not in registry["contract_schemas"]:
            errors.append("suite: missing contract schema " + schema_name)
        else:
            verify_ref(workspace, registry["contract_schemas"][schema_name], "suite.schema." + schema_name, errors)
    seen = set()
    for item in registry["cases"]:
        case_id = item["case_id"]
        if case_id in seen:
            errors.append("suite: duplicate case " + case_id)
        seen.add(case_id)
        spec_path = verify_ref(workspace, item["case_spec"], case_id + ".spec_ref", errors)
        gold_path = verify_ref(workspace, item["case_gold"], case_id + ".gold_ref", errors)
        task_path = verify_ref(workspace, item["task_confirm"], case_id + ".task_confirm_ref", errors)
        physics_path = verify_ref(workspace, item["physics_profile"], case_id + ".physics_profile_ref", errors)
        if spec_path is None or gold_path is None:
            continue
        spec = load_json(spec_path)
        gold = load_json(gold_path)
        errors.extend(schema_errors(spec, spec_schema, case_id + ".spec"))
        errors.extend(schema_errors(gold, gold_schema, case_id + ".gold"))
        if spec["case_id"] != case_id or gold["case_id"] != case_id:
            errors.append(case_id + ": identifier mismatch")
        if gold["case_spec_sha256"] != sha256_file(spec_path):
            errors.append(case_id + ": gold does not bind current spec")
        if spec["source_lineage_status"] == "pending_reexport" and not args.allow_pending_lineage:
            errors.append(case_id + ": source lineage remains pending_reexport")

        if task_path is not None:
            with task_path.open("r", encoding="utf-8") as handle:
                task_confirm = yaml.safe_load(handle)
            errors.extend(schema_errors(task_confirm, task_schema, case_id + ".task_confirm"))
            if task_confirm != build_task_confirm(spec, gold):
                errors.append(case_id + ": task confirmation does not exactly bind current spec/gold")
        if physics_path is not None:
            physics_profile = load_json(physics_path)
            errors.extend(schema_errors(physics_profile, physics_schema, case_id + ".physics_profile"))
            try:
                expected_profile = build_physics_profile(workspace, spec_path, gold_path)
            except Exception as exc:
                errors.append(case_id + ": physics profile regeneration failed: " + str(exc))
            else:
                if physics_profile != expected_profile:
                    errors.append(case_id + ": physics profile does not exactly match frozen runtime/spec/gold")
            nonphysical = (
                physics_profile.get("diagnostic_contract", {})
                .get("mechanics", {})
                .get("nonphysical_dynamic_bodies")
            )
            if nonphysical:
                errors.append(case_id + ": physics profile contains nonphysical dynamic bodies")
            diagnostic = physics_profile.get("diagnostic_contract", {})
            mechanics = diagnostic.get("mechanics", {})
            kinematics = diagnostic.get("kinematics", {})
            friction = diagnostic.get("friction", {})
            required_channels = (
                (mechanics, "generalized force-balance residual"),
                (mechanics, "kinetic and potential energy"),
                (kinematics, "finite-difference acceleration and jerk"),
                (friction, "sliding/torsional/rolling coefficients"),
                (friction, "Coulomb-cone utilization"),
                (friction, "contact-point tangential slip speed"),
            )
            if any(name not in section.get("channels", []) for section, name in required_channels):
                errors.append(case_id + ": incomplete mechanics/kinematics/friction diagnostic channels")
            if mechanics.get("force_balance_relative_tolerance") != 1e-5 or kinematics.get("quaternion_norm_tolerance") != 1e-6 or friction.get("cone_utilization_failure") != 1.25:
                errors.append(case_id + ": unexpected frozen physical diagnostic tolerance")
            for label, binding in physics_profile.get("resolved_contact_groups", {}).items():
                if not binding.get("geoms") and not binding.get("bodies"):
                    errors.append(case_id + ": resolved contact group is empty: " + label)

        for label, ref in spec["artifacts"].items():
            verify_ref(workspace, ref, case_id + ".artifacts." + label, errors)
        source_path = verify_ref(workspace, spec["source"], case_id + ".source", errors)
        manifest_path = verify_ref(workspace, spec["source_manifest"], case_id + ".source_manifest", errors)
        dependency_path = workspace / spec["artifacts"]["dependency_manifest"]["path"]
        dependencies = load_json(dependency_path)
        for index, ref in enumerate(dependencies["files"]):
            verify_ref(workspace, ref, case_id + ".dependency." + str(index), errors)

        m1_path = workspace / spec["artifacts"]["m1_base"]["path"]
        overlay_path = workspace / spec["artifacts"]["m2_overlay"]["path"]
        overlay = load_json(overlay_path)
        if overlay.get("m1_base_sha256") != sha256_file(m1_path):
            errors.append(case_id + ": M2 overlay does not bind M1")
        if not overlay.get("task_independent"):
            errors.append(case_id + ": M2 overlay is not task-independent")

        probe_path = suite / spec["probe"]["path"]
        validator_path = suite / spec["validator"]["path"]
        if sha256_file(probe_path) != spec["probe"]["sha256"]:
            errors.append(case_id + ": probe hash mismatch")
        if sha256_file(validator_path) != spec["validator"]["sha256"]:
            errors.append(case_id + ": validator hash mismatch")
        for support in spec["validator"]["support_modules"]:
            support_path = suite / support["path"]
            if not support_path.is_file():
                errors.append(case_id + ": missing validator support module " + support["path"])
                continue
            if sha256_file(support_path) != support["sha256"]:
                errors.append(case_id + ": validator support module hash mismatch " + support["path"])
        errors.extend(audit_probe(probe_path))
        for support in spec["probe"].get("support_modules", []):
            support_path = suite / support["path"]
            if not support_path.is_file():
                errors.append(case_id + ": missing probe support module " + support["path"])
                continue
            if sha256_file(support_path) != support["sha256"]:
                errors.append(case_id + ": support module hash mismatch " + support["path"])
            errors.extend(audit_probe(support_path))

        # M1 must remain independently loadable, while robot/control bindings
        # are checked against the executable M1+M2 runtime.
        try:
            mujoco.MjModel.from_xml_path(str(m1_path))
        except Exception as exc:
            errors.append(case_id + ": immutable M1 load failed: " + str(exc))
            continue
        runtime_path = workspace / spec["artifacts"]["runtime_model"]["path"]
        try:
            model = mujoco.MjModel.from_xml_path(str(runtime_path))
        except Exception as exc:
            errors.append(case_id + ": executable runtime load failed: " + str(exc))
            continue
        body_names = {model.body(index).name for index in range(model.nbody)}
        joint_names = {model.joint(index).name for index in range(model.njnt)}
        actuator_names = {model.actuator(index).name for index in range(model.nu)}
        geom_names = {model.geom(index).name for index in range(model.ngeom)}
        collision_friction = []
        for geom_id in range(model.ngeom):
            if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
                coefficients = [float(value) for value in model.geom_friction[geom_id]]
                collision_friction.append(coefficients)
                if not all(math.isfinite(value) and value >= 0.0 for value in coefficients):
                    errors.append(case_id + ": invalid collision friction coefficients")
        if not collision_friction or not any(values[0] > 0.0 for values in collision_friction):
            errors.append(case_id + ": no positive sliding friction on collision geoms")
        if not any(values[1] > 0.0 and values[2] > 0.0 for values in collision_friction):
            errors.append(case_id + ": torsional/rolling friction channels are not configured")
        terminal_body = item.get("terminal_body")
        if not terminal_body or terminal_body not in body_names:
            errors.append(case_id + ": missing registry terminal_body in runtime")
        contact_policy = gold["contact_policy"]
        groups = contact_policy["groups"]
        pair_labels = {label for pair in contact_policy["allowed"] + contact_policy["forbidden"] for label in pair}
        for label in sorted(pair_labels):
            if label not in groups:
                errors.append(case_id + ": contact group has no binding: " + label)
                continue
            binding = groups[label]
            for selector, candidates in (("geoms", geom_names), ("bodies", body_names)):
                for pattern in binding.get(selector, []):
                    if not any(fnmatch.fnmatchcase(name, pattern) for name in candidates):
                        errors.append(case_id + ": contact group " + label + " has unmatched " + selector + " pattern " + pattern)
            condition = binding.get("active_when")
            if condition and condition["joint"] not in joint_names:
                errors.append(case_id + ": contact group " + label + " uses missing activation joint " + condition["joint"])

        for name in spec["observation"]["bodies"]:
            if name not in body_names:
                errors.append(case_id + ": missing observed body " + name)
        for name in spec["observation"]["joints"]:
            if name not in joint_names:
                errors.append(case_id + ": missing observed joint " + name)
        for name in spec["probe"]["actuator_limits"]:
            if name not in actuator_names:
                errors.append(case_id + ": missing controlled actuator " + name)

    for index, ref in enumerate(lock["files"]):
        verify_ref(workspace, ref, "freeze_lock." + str(index), errors)

    result = {
        "schema_version": "simcode4d.rc3a-dev1.preflight",
        "passed": not errors,
        "case_count": len(registry["cases"]),
        "errors": errors,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
