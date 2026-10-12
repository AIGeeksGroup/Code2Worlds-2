#!/usr/bin/env python3
"""Evaluate a first-output method package with frozen DEV15 probes/validators.

Input layout: SUBMISSION_ROOT/CASE_ID/{runtime.xml,task_overlay.json,bindings.json,assets...}.
No authoring scenes or internal experiment scripts are needed to test a package.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import gc
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from method_metrics import micro_f1_counts, robust_task_success, summarize_method  # noqa: E402
from runtime import sha256_file  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def local_xml_assets(xml_path: Path, package: Path) -> list[str]:
    """Check that every referenced MJCF asset/include is inside the package."""
    errors: list[str] = []
    visited: set[Path] = set()
    root = package.resolve()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        try:
            path.relative_to(root)
        except ValueError:
            errors.append("XML/include escapes package: " + str(path))
            return
        try:
            document = ET.parse(path)
        except (OSError, ET.ParseError) as exc:
            errors.append("XML parse failure: " + str(exc))
            return
        compiler = document.getroot().find("compiler")
        directories = {kind: Path(compiler.get(kind, "")) if compiler is not None else Path("")
                       for kind in ("assetdir", "meshdir", "texturedir")}
        for kind, directory in directories.items():
            if directory.is_absolute() or ".." in directory.parts:
                errors.append("compiler " + kind + " is not package-relative")
        for element in document.iter():
            name = element.get("file")
            if name is None:
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append("asset path is not package-relative: " + name)
                continue
            prefix = Path("") if element.tag == "include" else (
                directories["meshdir"] if element.tag == "mesh" and str(directories["meshdir"]) != "." else
                directories["texturedir"] if element.tag == "texture" and str(directories["texturedir"]) != "." else
                directories["assetdir"])
            asset = (path.parent / prefix / relative).resolve()
            try:
                asset.relative_to(root)
            except ValueError:
                errors.append("asset escapes package: " + name)
                continue
            if not asset.is_file():
                errors.append("missing local asset: " + name)
            elif element.tag == "include":
                visit(asset)
    visit(xml_path)
    return errors


def model_names(model: mujoco.MjModel) -> dict[str, set[str]]:
    return {
        "bodies": {model.body(i).name for i in range(model.nbody)},
        "joints": {model.joint(i).name for i in range(model.njnt)},
        "actuators": {model.actuator(i).name for i in range(model.nu)},
        "geoms": {model.geom(i).name for i in range(model.ngeom)},
        "equalities": {model.equality(i).name for i in range(model.neq)},
    }


def interface_errors(model: mujoco.MjModel, spec: dict, gold: dict) -> list[str]:
    names = model_names(model)
    errors = []
    required = (
        ("observed body", spec["observation"]["bodies"], names["bodies"]),
        ("observed joint", spec["observation"]["joints"], names["joints"]),
        ("probe actuator", spec["probe"]["actuator_limits"], names["actuators"]),
    )
    for label, expected, present in required:
        errors += ["missing " + label + ": " + name for name in expected if name not in present]
    grasp = spec["probe"].get("grasp_api")
    if grasp and grasp["equality"] not in names["equalities"]:
        errors.append("missing grasp equality: " + grasp["equality"])
    for entity in spec["entities"]:
        if not entity.get("required", True):
            continue
        pattern = entity["model_binding"]
        if pattern.startswith("static_fluid:"):
            if not spec.get("static_field"):
                errors.append("missing static field: " + pattern)
        elif not any(fnmatch.fnmatchcase(name, pattern) for name in names["bodies"] | names["geoms"]):
            errors.append("missing entity " + entity["entity_id"] + ": " + pattern)
    for label, binding in gold["contact_policy"]["groups"].items():
        for kind, candidates in (("bodies", names["bodies"]), ("geoms", names["geoms"])):
            for pattern in binding.get(kind, []):
                if not any(fnmatch.fnmatchcase(name, pattern) for name in candidates):
                    errors.append("unmatched contact group " + label + ": " + pattern)
        condition = binding.get("active_when")
        if condition and condition["joint"] not in names["joints"]:
            errors.append("missing contact activation joint: " + condition["joint"])
    if not np.all(np.isfinite(model.opt.gravity)) or not np.allclose(model.opt.gravity, [0, 0, -9.81]):
        errors.append("gravity differs from frozen benchmark")
    if not math.isclose(float(model.opt.timestep), float(spec["simulator"]["timestep_s"]), abs_tol=1e-12):
        errors.append("timestep differs from frozen benchmark")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        errors.append("integrator differs from frozen implicitfast setup")
    if not any(int(model.geom_contype[i]) or int(model.geom_conaffinity[i]) for i in range(model.ngeom)):
        errors.append("no collision geometry")
    for i in range(1, model.nbody):
        if model.body_mass[i] > 0 and (not np.isfinite(model.body_mass[i]) or not np.all(np.isfinite(model.body_inertia[i])) or np.any(model.body_inertia[i] <= 0)):
            errors.append("invalid mass/inertia on body " + model.body(i).name)
    for i in range(model.ngeom):
        if int(model.geom_contype[i]) or int(model.geom_conaffinity[i]):
            if not np.all(np.isfinite(model.geom_friction[i])) or model.geom_friction[i, 0] <= 0:
                errors.append("invalid friction on geom " + model.geom(i).name)
    for actuator in spec["probe"]["actuator_limits"]:
        if actuator in names["actuators"]:
            index = model.actuator(actuator).id
            if model.actuator_ctrllimited[index] and (not np.isfinite(model.actuator_ctrlrange[index]).all() or model.actuator_ctrlrange[index, 0] >= model.actuator_ctrlrange[index, 1]):
                errors.append("invalid control range: " + actuator)
    return errors


def binding_score(payload: dict, spec: dict, model: mujoco.MjModel) -> dict:
    names = model_names(model)
    available = names["bodies"] | names["geoms"]
    entries = payload.get("bindings", [])
    predictions = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("entity_id"), str):
            predictions.setdefault(entry["entity_id"], []).extend(entry.get("model_names", []))
    required = {e["entity_id"]: e for e in spec["entities"] if e.get("required", True)}
    tp = fp = fn = 0
    details = []
    for entity_id, entity in required.items():
        pattern = entity["model_binding"]
        given = predictions.get(entity_id, [])
        expected = {name for name in available if fnmatch.fnmatchcase(name, pattern)}
        if pattern.startswith("static_fluid:"):
            expected = {pattern}
        correct = bool(given and set(given) == expected)
        tp += int(correct)
        fp += int(bool(given) and not correct)
        fn += int(not correct)
        details.append({"entity_id": entity_id, "expected": sorted(expected),
                        "predicted": sorted(given), "correct": correct})
    fp += len(set(predictions) - set(required))
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / denominator if denominator else None, "details": details,
            "scope": "registered runtime entity bindings; source-name F1 needs authoring-source catalogs"}


def support_score(payload: dict, case_id: str, labels: dict) -> dict | None:
    gold = labels.get(case_id)
    if gold is None:
        return None
    expected = {tuple(pair) for pair in gold}
    children = {child for child, _ in expected}
    predicted = {
        (row["child"], row["parent"])
        for row in payload.get("support_relations", [])
        if isinstance(row, dict) and row.get("child") in children and isinstance(row.get("parent"), str)
    }
    result = micro_f1_counts(predicted, expected)
    result["scope"] = "annotated initial task-object support relations only"
    return result


def overlay_errors(payload: dict, case_id: str, runtime_hash: str, spec: dict, model: mujoco.MjModel) -> list[str]:
    errors = []
    if payload.get("schema_version") != "code2worlds2.task-dynamics.v1":
        errors.append("overlay schema_version")
    if payload.get("case_id") != case_id:
        errors.append("overlay case_id")
    if payload.get("runtime_sha256") != runtime_hash:
        errors.append("overlay runtime_sha256")
    expected = {e["model_binding"] for e in spec["entities"] if e.get("required", True) and e["role"] in {"target_object", "articulated_object"} and e["mobility"] in {"free", "articulated"}}
    declared = payload.get("dynamic_bodies")
    if not isinstance(declared, list) or set(declared) != expected or len(declared) != len(expected):
        errors.append("overlay dynamic_bodies do not cover required dynamic objects")
    if not isinstance(payload.get("events", []), list):
        errors.append("overlay events must be a list")
    else:
        for event in payload.get("events", []):
            if not isinstance(event, dict) or event.get("type") not in {"rigid_body", "joint", "visual_effect"}:
                errors.append("overlay event has invalid type")
                break
    return errors


def score_case(suite: Path, workspace: Path, registry_item: dict, package: Path,
               output: Path, generation_seed: int, physics_seeds: list[int], labels: dict) -> tuple[dict, list[dict]]:
    case_id = registry_item["case_id"]
    spec_path = workspace / registry_item["case_spec"]["path"]
    gold_path = workspace / registry_item["case_gold"]["path"]
    spec, gold = read_json(spec_path), read_json(gold_path)
    xml_path = package / "runtime.xml"
    overlay_path = package / "task_overlay.json"
    binding_path = package / "bindings.json"
    row: dict[str, Any] = {"case_id": case_id, "build_pass": False, "overlay_pass": False,
                           "build_errors": [], "overlay_errors": [], "entity_binding": None,
                           "support_relation": None}
    episodes = []
    try:
        if not xml_path.is_file():
            raise FileNotFoundError("runtime.xml missing")
        row["build_errors"] += local_xml_assets(xml_path, package)
        if not binding_path.is_file():
            raise FileNotFoundError("bindings.json missing")
        bindings = read_json(binding_path)
        if bindings.get("schema_version") != "code2worlds2.bindings.v1" or bindings.get("case_id") != case_id:
            row["build_errors"].append("bindings schema/case ID mismatch")
        if not isinstance(bindings.get("bindings"), list) or not isinstance(bindings.get("support_relations", []), list):
            row["build_errors"].append("invalid bindings list or support relations")
            bindings = {"bindings": [], "support_relations": []}
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        row["build_errors"] += interface_errors(model, spec, gold)
        row["entity_binding"] = binding_score(bindings, spec, model)
        row["support_relation"] = support_score(bindings, case_id, labels)
        if row["entity_binding"]["fn"] or row["entity_binding"]["fp"]:
            row["build_errors"].append("required entity binding mismatch")
        data = mujoco.MjData(model)
        for _ in range(round(0.2 / model.opt.timestep)):
            old_time = float(data.time)
            mujoco.mj_step(model, data)
            if float(data.time) <= old_time:
                row["build_errors"].append("numerical reset during smoke execution")
                break
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            row["build_errors"].append("non-finite 0.2 s smoke execution")
        row["build_pass"] = not row["build_errors"]
        if overlay_path.is_file():
            overlay = read_json(overlay_path)
            row["overlay_errors"] = overlay_errors(overlay, case_id, sha256_file(xml_path), spec, model)
        else:
            row["overlay_errors"].append("task_overlay.json missing")
        row["overlay_pass"] = not row["overlay_errors"]
    except Exception as exc:
        row["build_errors"].append(type(exc).__name__ + ": " + str(exc))

    # Admission and execution use separate model lifetimes. The large terrain
    # cases otherwise retain two compiled MuJoCo worlds at the same time.
    if "data" in locals():
        del data
    if "model" in locals():
        del model
    gc.collect()

    if row["build_pass"] and row["overlay_pass"]:
        shadow = copy.deepcopy(spec)
        shadow["artifacts"]["runtime_model"] = {"path": str(xml_path.resolve()),
            "sha256": sha256_file(xml_path), "bytes": xml_path.stat().st_size}
        shadow_path = output / "shadow_specs" / (case_id + ".json")
        save_json(shadow_path, shadow)
    for seed in physics_seeds:
        episode = {"case_id": case_id, "generation_seed": generation_seed,
                   "physics_seed": seed, "validation": None, "robust_task_success": False}
        if row["build_pass"] and row["overlay_pass"]:
            try:
                episode_dir = output / "episodes" / case_id / ("p" + str(seed))
                command = [sys.executable, str(TOOLS / "episode_worker.py"),
                           "--suite", str(suite), "--spec", str(shadow_path),
                           "--gold", str(gold_path), "--generation-seed", str(generation_seed),
                           "--physics-seed", str(seed), "--output", str(episode_dir)]
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
                validation = read_json(episode_dir / "validation.json")
                episode["validation"] = validation
                episode["robust_task_success"] = robust_task_success(True, True, validation)
            except Exception as exc:
                episode["execution_error"] = type(exc).__name__ + ": " + str(exc)
        episodes.append(episode)
    return row, episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=TOOLS.parent)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--generation-seed", type=int, default=1103)
    parser.add_argument("--physics-seeds", type=int, nargs="+")
    parser.add_argument("--skip-suite-check", action="store_true",
                        help="skip the full public hash/model check after a separate successful run")
    parser.add_argument("--_single-case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    suite = args.suite.resolve()
    workspace = suite.parents[1]
    registry = read_json(suite / "config/suite.json")
    selected = args.case_ids or [item["case_id"] for item in registry["cases"]]
    if len(selected) != len(set(selected)) or set(selected) - {item["case_id"] for item in registry["cases"]}:
        parser.error("case IDs must be distinct registered cases")
    seeds = args.physics_seeds or registry["physics_seeds"]
    if len(seeds) != len(set(seeds)) or set(seeds) - set(registry["physics_seeds"]):
        parser.error("physics seeds must be distinct registered seeds")
    if args.output.exists() and not args._single_case:
        parser.error("output already exists; choose a fresh path")
    if not args.submission_root.is_dir():
        parser.error("submission root is missing")
    if not args.skip_suite_check and not args._single_case:
        completed = subprocess.run([sys.executable, str(suite / "check_public_release.py")],
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError("frozen evaluator integrity check failed: " + (completed.stdout.strip() or completed.stderr.strip()))
    labels = read_json(suite / "config/initial_support_labels.json")["relations"]
    items = {item["case_id"]: item for item in registry["cases"]}
    if args._single_case:
        if len(selected) != 1:
            parser.error("internal worker requires one case")
        case_id = selected[0]
        row, trials = score_case(suite, workspace, items[case_id],
                                 args.submission_root.resolve() / case_id, args.output.resolve(),
                                 args.generation_seed, seeds, labels)
        save_json(args.output.resolve() / "case_results" / (case_id + ".json"),
                  {"package": row, "episodes": trials})
        return 0
    packages, episodes = [], []
    args.output.mkdir(parents=True)
    for case_id in selected:
        command = [sys.executable, str(Path(__file__)), "--suite", str(suite),
                   "--submission-root", str(args.submission_root.resolve()),
                   "--output", str(args.output.resolve()), "--case-ids", case_id,
                   "--generation-seed", str(args.generation_seed),
                   "--physics-seeds", *(str(seed) for seed in seeds),
                   "--skip-suite-check", "--_single-case"]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            result = read_json(args.output.resolve() / "case_results" / (case_id + ".json"))
            row, trials = result["package"], result["episodes"]
        else:
            row = {"case_id": case_id, "build_pass": False, "overlay_pass": False,
                   "build_errors": ["case worker failed: " + (completed.stderr.strip() or completed.stdout.strip())],
                   "overlay_errors": [], "entity_binding": None, "support_relation": None}
            trials = [{"case_id": case_id, "generation_seed": args.generation_seed,
                       "physics_seed": seed, "validation": None, "robust_task_success": False,
                       "execution_error": "case worker failed"} for seed in seeds]
        packages.append(row)
        episodes.extend(trials)
        print(case_id + ": build=" + str(row["build_pass"]) + " overlay=" + str(row["overlay_pass"])
              + " RTS=" + str(sum(t["robust_task_success"] for t in trials)) + "/" + str(len(trials)), flush=True)
    report = {"schema_version": "simcode4d.dev15.method-score.v1",
              "suite_registry_sha256": sha256_file(suite / "config/suite.json"),
              "scorer_sha256": sha256_file(Path(__file__)),
              "metric_functions_sha256": sha256_file(TOOLS / "method_metrics.py"),
              "support_labels_sha256": sha256_file(suite / "config/initial_support_labels.json"),
              "submission_root": str(args.submission_root.resolve()),
              "support_label_scope": "five annotated task-object cases; unannotated cases excluded from support F1",
              "entity_binding_scope": "runtime entity names; paper source-name F1 needs original authoring catalogs",
              "metrics": summarize_method(packages, episodes), "packages": packages, "episodes": episodes}
    save_json(args.output.resolve() / "method_score.json", report)
    return 0 if all(row["build_pass"] and row["overlay_pass"] for row in packages) and all("execution_error" not in row for row in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
