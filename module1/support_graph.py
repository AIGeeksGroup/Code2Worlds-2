#!/usr/bin/env python3
"""Infer an auditable initial support graph from an exported scene manifest.

This is a geometric proposal. MuJoCo contact data must confirm the relation
after settling; rotated thin geometry and terrain height variation need care.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORT_GROUPS = ("background_terrain", "background_floor", "prop_concave_large",
                  "prop_convex_small", "prop_dynamic")
OBJECT_GROUPS = ("prop_concave_large", "prop_convex_small", "prop_dynamic")


def aabb(entry: dict[str, Any]) -> tuple[float, ...] | None:
    if "world_aabb" in entry:
        return tuple(float(value) for value in entry["world_aabb"])
    if "world_pos" in entry and "world_size" in entry:
        center = [float(value) for value in entry["world_pos"]]
        half = [float(value) / 2 for value in entry["world_size"]]
        return tuple(center[i] - half[i] for i in range(3)) + tuple(center[i] + half[i] for i in range(3))
    return None


def xy_overlap(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(0.0, min(left[3], right[3]) - max(left[0], right[0])) * max(0.0, min(left[4], right[4]) - max(left[1], right[1]))


def infer_support_graph(manifest: dict[str, Any], tolerance_m: float = 0.05) -> dict[str, Any]:
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive")
    groups = manifest.get("groups", {})
    records = []
    for group in (*SUPPORT_GROUPS, "background_wall"):
        for entry in groups.get(group, []):
            box = aabb(entry)
            if box is not None and len(box) == 6:
                records.append((group, entry["name"], box))
    objects = [(group, name, box) for group, name, box in records if group in OBJECT_GROUPS]
    supports = [(group, name, box) for group, name, box in records if group in SUPPORT_GROUPS]
    relations = []
    for _, name, box in objects:
        candidates = []
        for group, parent, surface in supports:
            if parent == name or xy_overlap(box, surface) <= 0:
                continue
            if surface[5] > box[5] + tolerance_m:
                continue
            gap = box[2] - surface[5]
            candidates.append((abs(gap), gap, group, parent))
        if candidates:
            _, gap, group, parent = min(candidates)
        else:
            # Decorative floor tiles may omit the true room floor. Use the
            # lowest credible prop bottom as a conservative global fallback.
            floor_levels = [surface[5] for group, _, surface in supports if group in {"background_floor", "background_terrain"}]
            level = min(floor_levels) if floor_levels else min((item[2][2] for item in objects), default=box[2])
            parent, group, gap = "@global_ground", "fallback", box[2] - level
        relation = "supported_by" if abs(gap) <= tolerance_m else "floating" if gap > 0 else "embedded"
        relations.append({"child": name, "parent": parent, "relation": relation,
                          "vertical_gap_m": round(gap, 6), "support_kind": group})
        for other_group, other_name, other_box in records:
            if other_name == name:
                continue
            inside = all(other_box[i] - tolerance_m <= box[i] for i in range(3)) and all(box[i + 3] <= other_box[i + 3] + tolerance_m for i in range(3))
            if inside and other_group in OBJECT_GROUPS:
                relations.append({"child": name, "parent": other_name, "relation": "inside"})
            if other_group == "background_wall" and min(abs(box[0] - other_box[3]), abs(box[3] - other_box[0]), abs(box[1] - other_box[4]), abs(box[4] - other_box[1])) <= tolerance_m and min(box[5], other_box[5]) > max(box[2], other_box[2]):
                relations.append({"child": name, "parent": other_name, "relation": "attached_to"})
        if any(word in name.lower() for word in ("door", "panel")):
            relations.append({"child": name, "parent": parent, "relation": "hinged_to",
                              "note": "name-based articulation proposal; confirm the actual joint"})
    return {"schema_version": "code2worlds2.support-graph.v1", "tolerance_m": tolerance_m,
            "method": "world-space AABB proposal; confirm using simulator contact after settling",
            "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--tolerance-m", type=float, default=0.05)
    args = parser.parse_args()
    export = args.export_dir.resolve()
    graph = infer_support_graph(json.loads((export / "manifest.json").read_text(encoding="utf-8")), args.tolerance_m)
    target = export / "module1/support_graph.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "relation_count": len(graph["relations"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
