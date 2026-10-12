#!/usr/bin/env python3
"""Run a registry-locked SimCode4D case/seed matrix without overwriting episodes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import mujoco

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from runtime import sha256_file  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def select_registered(requested, registered, label: str):
    values = list(registered if requested is None else requested)
    if len(values) != len(set(values)):
        raise ValueError(label + " contains duplicates")
    unknown = [value for value in values if value not in registered]
    if unknown:
        raise ValueError(label + " contains unregistered values: " + repr(unknown))
    return values


def verify_ref(workspace: Path, ref: dict[str, Any], label: str) -> Path:
    path = workspace / ref["path"]
    if not path.is_file():
        raise FileNotFoundError(label + " missing: " + str(path))
    if path.stat().st_size != int(ref["bytes"]):
        raise RuntimeError(label + " byte count mismatch: " + str(path))
    if sha256_file(path) != ref["sha256"]:
        raise RuntimeError(label + " sha256 mismatch: " + str(path))
    return path


def run_preflight(workspace: Path, suite_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(suite_root / "tools/validate_suite.py"),
        "--workspace", str(workspace),
        "--suite", str(suite_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise RuntimeError("strict preflight failed:\n" + detail)
    return json.loads(completed.stdout)


def load_existing_episode(
    output_dir: Path,
    case_id: str,
    generation_seed: int,
    physics_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_path = output_dir / "episode_log.json"
    validation_path = output_dir / "validation.json"
    if not raw_path.is_file() or not validation_path.is_file():
        raise RuntimeError("incomplete existing episode: " + str(output_dir))
    raw = load_json(raw_path)
    validation = load_json(validation_path)
    expected = (case_id, generation_seed, physics_seed)
    raw_identity = (raw["case_id"], raw["generation_seed"], raw["physics_seed"])
    validation_identity = (
        validation["case_id"],
        validation["generation_seed"],
        validation["physics_seed"],
    )
    if raw_identity != expected or validation_identity != expected:
        raise RuntimeError("existing episode identity mismatch: " + str(output_dir))
    if validation["raw_event_log_sha256"] != sha256_file(raw_path):
        raise RuntimeError("existing episode raw-log hash mismatch: " + str(output_dir))
    return raw, validation


def manifest_row(
    output_root: Path,
    output_dir: Path,
    raw: dict[str, Any],
    validation: dict[str, Any],
    reused: bool,
) -> dict[str, Any]:
    return {
        "case_id": validation["case_id"],
        "generation_seed": validation["generation_seed"],
        "physics_seed": validation["physics_seed"],
        "status": validation["status"],
        "dev_reference_success": validation["dev_reference_success"],
        "task_predicates_pass": validation["task_predicates_pass"],
        "safety_pass": validation["safety_pass"],
        "integrity_pass": validation["integrity_pass"],
        "output_dir": str(output_dir.relative_to(output_root)),
        "raw_event_log_sha256": validation["raw_event_log_sha256"],
        "artifact_hashes": raw["artifact_hashes"],
        "wall_time_s": raw["summary"]["wall_time_s"],
        "reused_on_resume": reused,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+")
    parser.add_argument("--physics-seeds", type=int, nargs="+")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    suite_root = args.suite.resolve()
    workspace = suite_root.parents[1]
    output_root = args.output.resolve()
    registry_path = suite_root / "config/suite.json"
    registry = load_json(registry_path)
    for label, ref in registry.get("toolchain", {}).items():
        verify_ref(workspace, ref, "toolchain " + label)
    generation_seeds = select_registered(
        args.generation_seeds, registry["generation_seeds"], "generation seeds"
    )
    physics_seeds = select_registered(
        args.physics_seeds, registry["physics_seeds"], "physics seeds"
    )
    registered_cases = [item["case_id"] for item in registry["cases"]]
    case_ids = select_registered(args.case_ids, registered_cases, "case IDs")
    case_by_id = {item["case_id"]: item for item in registry["cases"]}

    if output_root.exists() and not args.resume:
        raise FileExistsError("output exists; use a fresh path or --resume: " + str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = {"passed": True, "skipped": True}
    if not args.skip_preflight:
        preflight = run_preflight(workspace, suite_root)

    verify_ref(workspace, registry["semantic_freeze"], "semantic freeze")
    for case_id in case_ids:
        item = case_by_id[case_id]
        verify_ref(workspace, item["case_spec"], case_id + " case spec")
        verify_ref(workspace, item["case_gold"], case_id + " case gold")

    expected_count = len(case_ids) * len(generation_seeds) * len(physics_seeds)
    manifest: dict[str, Any] = {
        "schema_version": "simcode4d.dev15.run-manifest.v1",
        "suite_id": registry["suite_id"],
        "suite_registry_sha256": sha256_file(registry_path),
        "semantic_freeze_sha256": registry["semantic_freeze"]["sha256"],
        "runtime_sha256": sha256_file(suite_root / "tools/runtime.py"),
        "physics_diagnostics_sha256": sha256_file(suite_root / "tools/physics_diagnostics.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "toolchain_hashes": {label: ref["sha256"] for label, ref in registry.get("toolchain", {}).items()},
        "simulator": {"name": "mujoco", "version": mujoco.__version__},
        "selection": {
            "case_ids": case_ids,
            "generation_seeds": generation_seeds,
            "physics_seeds": physics_seeds,
        },
        "replication_note": registry["replication_note"],
        "preflight": preflight,
        "expected_episode_count": expected_count,
        "completed_episode_count": 0,
        "execution_complete": False,
        "execution_errors": [],
        "episodes": [],
    }
    manifest_path = output_root / "run_manifest.json"
    write_json(manifest_path, manifest)

    index = 0
    for case_id in case_ids:
        item = case_by_id[case_id]
        spec_path = workspace / item["case_spec"]["path"]
        gold_path = workspace / item["case_gold"]["path"]
        for generation_seed in generation_seeds:
            for physics_seed in physics_seeds:
                index += 1
                output_dir = (
                    output_root / "episodes" / case_id
                    / ("g" + str(generation_seed))
                    / ("p" + str(physics_seed))
                )
                reused = output_dir.exists()
                try:
                    if reused:
                        if not args.resume:
                            raise FileExistsError(str(output_dir))
                        raw, validation = load_existing_episode(
                            output_dir, case_id, generation_seed, physics_seed
                        )
                    else:
                        command = [
                            sys.executable, str(TOOLS_DIR / "episode_worker.py"),
                            "--suite", str(suite_root),
                            "--spec", str(spec_path),
                            "--gold", str(gold_path),
                            "--generation-seed", str(generation_seed),
                            "--physics-seed", str(physics_seed),
                            "--output", str(output_dir),
                        ]
                        completed = subprocess.run(command, capture_output=True, text=True)
                        if completed.returncode != 0:
                            raise RuntimeError("episode worker failed: " + (completed.stderr.strip() or completed.stdout.strip()))
                        validation = load_json(output_dir / "validation.json")
                        raw = load_json(output_dir / "episode_log.json")
                    row = manifest_row(
                        output_root, output_dir, raw, validation, reused
                    )
                    manifest["episodes"].append(row)
                    manifest["completed_episode_count"] = len(manifest["episodes"])
                    print(
                        f"[{index:03d}/{expected_count:03d}] {case_id} "
                        f"g={generation_seed} p={physics_seed} -> {validation['status']}",
                        flush=True,
                    )
                except Exception as exc:
                    error = {
                        "case_id": case_id,
                        "generation_seed": generation_seed,
                        "physics_seed": physics_seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    manifest["execution_errors"].append(error)
                    print(
                        f"[{index:03d}/{expected_count:03d}] {case_id} "
                        f"g={generation_seed} p={physics_seed} -> EXECUTION_ERROR: {exc}",
                        flush=True,
                    )
                write_json(manifest_path, manifest)

    manifest["execution_complete"] = bool(
        manifest["completed_episode_count"] == expected_count
        and not manifest["execution_errors"]
    )
    write_json(manifest_path, manifest)
    passed = sum(row["dev_reference_success"] for row in manifest["episodes"])
    print(
        f"completed={manifest['completed_episode_count']}/{expected_count} "
        f"reference_pass={passed}/{expected_count} "
        f"execution_errors={len(manifest['execution_errors'])}",
        flush=True,
    )
    return 0 if manifest["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
