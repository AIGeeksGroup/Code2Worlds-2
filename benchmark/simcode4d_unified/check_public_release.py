#!/usr/bin/env python3
"""Verify the publishable DEV15 runtime closure without Blender source archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco


SUITE = Path(__file__).resolve().parent
ROOT = SUITE.parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def iter_refs(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        for child in value.values():
            yield from iter_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_refs(child)


def verify_ref(ref: dict, errors: list[str]) -> None:
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"invalid path: {relative}")
        return
    path = ROOT / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
    elif path.stat().st_size != ref["bytes"] or digest(path) != ref["sha256"]:
        errors.append(f"hash/size mismatch: {relative}")


def verify_suite_path(ref: dict, errors: list[str]) -> None:
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"invalid suite path: {relative}")
        return
    path = SUITE / relative
    if not path.is_file():
        errors.append(f"missing: {path.relative_to(ROOT)}")
    elif digest(path) != ref["sha256"]:
        errors.append(f"hash mismatch: {path.relative_to(ROOT)}")


def main() -> int:
    registry = load_json(SUITE / "config/suite.json")
    lock = load_json(SUITE / "config/freeze_lock.json")
    errors: list[str] = []
    refs = {ref["path"]: ref for ref in iter_refs(registry)}
    if lock["suite_id"] != registry["suite_id"]:
        errors.append("freeze lock suite ID differs from registry")
    for ref in lock["files"]:
        refs[ref["path"]] = ref
    omitted_sources: set[str] = set()
    cases = registry["cases"]
    if len(cases) != 15:
        errors.append(f"expected 15 cases, found {len(cases)}")
    for case in cases:
        spec_path = ROOT / case["case_spec"]["path"]
        if not spec_path.is_file():
            continue
        spec = load_json(spec_path)
        for ref in iter_refs(spec):
            if ref["path"].endswith(".blend") and ref == spec.get("source"):
                omitted_sources.add(ref["path"])
            else:
                refs[ref["path"]] = ref
        dependency_path = ROOT / spec["artifacts"]["dependency_manifest"]["path"]
        if dependency_path.is_file():
            for ref in iter_refs(load_json(dependency_path)):
                refs[ref["path"]] = ref
        verify_suite_path(spec["probe"], errors)
        verify_suite_path(spec["validator"], errors)
        for support in spec["probe"].get("support_modules", []):
            verify_suite_path(support, errors)
        for support in spec["validator"].get("support_modules", []):
            verify_suite_path(support, errors)

    for ref in refs.values():
        verify_ref(ref, errors)

    if not errors:
        for case in cases:
            spec = load_json(ROOT / case["case_spec"]["path"])
            for label in ("m1_base", "runtime_model"):
                path = ROOT / spec["artifacts"][label]["path"]
                try:
                    mujoco.MjModel.from_xml_path(str(path))
                except Exception as exc:
                    errors.append(f"{case['case_id']} {label} MuJoCo load: {exc}")

    print(json.dumps({
        "passed": not errors,
        "case_count": len(cases),
        "verified_files": len(refs),
        "omitted_authoring_blends": len(omitted_sources),
        "errors": errors,
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
