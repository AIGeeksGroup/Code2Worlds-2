#!/usr/bin/env python3
"""Create a self-contained SER-03 reference package to exercise the public API."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from pathlib import Path

import mujoco


CASE = "SER-03-hotel-drawer"
SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve() / CASE
    if args.output.exists():
        parser.error("output already exists; use a fresh path")
    spec = json.loads((SUITE / "cases" / CASE / "case_spec.json").read_text(encoding="utf-8"))
    source = ROOT / spec["artifacts"]["runtime_model"]["path"]
    destination.mkdir(parents=True)
    runtime = destination / "runtime.xml"
    shutil.copy2(source, runtime)
    model = mujoco.MjModel.from_xml_path(str(runtime))
    available = {model.body(i).name for i in range(model.nbody)} | {model.geom(i).name for i in range(model.ngeom)}
    bindings = [
        {"entity_id": entity["entity_id"],
         "model_names": sorted(name for name in available if fnmatch.fnmatchcase(name, entity["model_binding"]))}
        for entity in spec["entities"] if entity.get("required", True)
    ]
    (destination / "bindings.json").write_text(json.dumps({
        "schema_version": "code2worlds2.bindings.v1", "case_id": CASE,
        "bindings": bindings,
        "support_relations": [{"child": "meal_parcel", "parent": "payload_drawer"}],
    }, indent=2) + "\n", encoding="utf-8")
    (destination / "task_overlay.json").write_text(json.dumps({
        "schema_version": "code2worlds2.task-dynamics.v1", "case_id": CASE,
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "dynamic_bodies": ["upper_drawer", "meal_parcel"], "events": [],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference_package": str(destination), "case_id": CASE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
