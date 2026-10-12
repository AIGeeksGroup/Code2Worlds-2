"""Module 1, sub-step 3: CoACD convex decomposition for KEY boulders.

Why this exists:
  - MuJoCo can use a non-convex mesh as a collision geometry, but it's slow
    and unreliable (penetration / phantom contacts) on irregular shapes.
  - Production pipelines (SceneSmith, InternScenes, Drake) all use V-HACD or
    its modern successor CoACD to break a visual mesh into a small set of
    convex pieces. MuJoCo can then use a `<geom type="mesh"/>` per convex
    piece and compute exact contact.
  - This is one of the concrete deliverables of Module 1.

For each KEY boulder we:
  1. Load its visual OBJ with trimesh
  2. Run CoACD with reasonable defaults (threshold ~0.05 means stop when
     remaining concavity below 5% of bbox -- visually faithful while keeping
     piece count low)
  3. Save each convex piece as boulder_XXX_col_YY.obj into module1/collision/
  4. Update keys.json with the per-boulder list of collision pieces

Run:
    python sim/module1_coacd.py EXPORT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import time

import coacd
import numpy as np
import trimesh


if len(sys.argv) < 2:
    print("usage: module1_coacd.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
M1_DIR = os.path.join(EXPORT_DIR, "module1")
keys_path = os.path.join(M1_DIR, "keys.json")
with open(keys_path) as f:
    keys = json.load(f)

COL_DIR = os.path.join(M1_DIR, "collision")
os.makedirs(COL_DIR, exist_ok=True)

# CoACD parameters. Tuning:
#   threshold:       smaller -> more faithful, more pieces. 0.05 is a good
#                    middle ground for natural rocks.
#   max_convex_hull: hard cap on piece count. 24 is overkill for a boulder
#                    but cheap.
#   preprocess_mode: "auto" handles non-watertight inputs gracefully.
#   resolution:      voxel grid for the inner pruning step. 2000 is plenty.
COACD_THRESHOLD = 0.05
COACD_MAX_PIECES = 24
COACD_RESOLUTION = 2000

print(f"[m1-coacd] CoACD threshold={COACD_THRESHOLD} max_pieces={COACD_MAX_PIECES}")
print(f"[m1-coacd] decomposing {len(keys['key_boulders'])} key boulders...")

results = []
for i, b in enumerate(keys["key_boulders"]):
    name = b["name"]
    src_path = os.path.join(EXPORT_DIR, b["mesh_rel"])
    if not os.path.exists(src_path):
        print(f"[m1-coacd] SKIP {name}: missing {src_path}")
        continue

    t0 = time.time()
    mesh = trimesh.load(src_path, force="mesh")
    if mesh.vertices.shape[0] == 0:
        print(f"[m1-coacd] SKIP {name}: empty mesh")
        continue

    # CoACD wants vertices + faces (np arrays); trimesh has both.
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_PIECES,
        resolution=COACD_RESOLUTION,
        preprocess_mode="auto",
    )
    dt = time.time() - t0

    pieces_rel = []
    total_verts = 0
    for j, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        piece_name = f"{name}_col_{j:02d}.obj"
        piece_path = os.path.join(COL_DIR, piece_name)
        piece.export(piece_path)
        pieces_rel.append(f"module1/collision/{piece_name}")
        total_verts += len(verts)

    print(f"[m1-coacd]  #{i:02d} {name}: {b['n_tris']} tris -> "
          f"{len(parts)} convex pieces ({total_verts} verts total, {dt:.1f}s)")
    results.append({
        "name": name,
        "n_pieces": len(parts),
        "total_verts": total_verts,
        "pieces_rel": pieces_rel,
        "decomp_seconds": round(dt, 2),
    })

# Update keys.json with the new collision info
keys["coacd"] = {
    "threshold": COACD_THRESHOLD,
    "max_pieces_per_boulder": COACD_MAX_PIECES,
    "resolution": COACD_RESOLUTION,
    "results": results,
}
with open(keys_path, "w") as f:
    json.dump(keys, f, indent=2)
print(f"[m1-coacd] DONE: {len(results)} boulders decomposed -> updated {keys_path}")
