"""Module 1 (extension), terrain grid prism: convex height-field cells.

Problem fixed:
  Module 1 emit treats terrain as a single <geom type="mesh">; MuJoCo
  silently convex-hulls it, so the entire 75x75m gobi terrain becomes a
  single boulder-shaped solid (z_max ~= +1.23 m even at flat regions).
  Any dynamic object spawned at low z gets ejected.

Approach:
  1. Slice the terrain in xy plane into N x N cells.
  2. In each cell, ray-cast downward on a SUBGRID (M+1 x M+1) to recover
     the true upper surface as a height field.
  3. Build a closed prism per cell: top height-field surface + 4 side
     walls + flat base at floor_z.
  4. Save each cell as a SINGLE convex mesh -- no CoACD. The prism is
     non-convex in general, but MuJoCo will convex-hull it, and that
     hull is a tight envelope of just THAT cell, not the whole terrain.
     With cell_size << terrain_size the per-cell convex hull error is
     bounded by the cell's z variation.

We rely on `<geom type="mesh">` semantics in MuJoCo: each individual cell
is one geom, so per-cell convex-hulling stays local. This is the same
trick CoACD uses, just with an a-priori grid instead of greedy splitting.

Outputs:
  EXPORT_DIR/module1/terrain_grid/cell_IJ.obj    (one prism per cell)
  keys.json["terrain_grid"] = {grid_n, sub_n, cell_size, cells: [...]}

Run:
    python sim/module1_terrain_grid.py EXPORT_DIR [GRID_N] [SUB_N]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import trimesh


if len(sys.argv) < 2:
    print("usage: module1_terrain_grid.py EXPORT_DIR [GRID_N] [SUB_N]")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
GRID_N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SUB_N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

M1_DIR = os.path.join(EXPORT_DIR, "module1")
OUT_DIR = os.path.join(M1_DIR, "terrain_grid")
os.makedirs(OUT_DIR, exist_ok=True)
# Wipe stale CoACD pieces if any
for f in os.listdir(OUT_DIR):
    if f.endswith(".obj"):
        os.remove(os.path.join(OUT_DIR, f))

with open(os.path.join(EXPORT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
with open(os.path.join(M1_DIR, "keys.json")) as f:
    keys = json.load(f)

terrain_rel = manifest["terrain"].get("collision_obj") or manifest["terrain"]["visual_obj"]
terrain_abs = os.path.join(EXPORT_DIR, terrain_rel)

print(f"[m1-terrain] loading {terrain_abs}")
mesh = trimesh.load(terrain_abs, force="mesh")
print(f"[m1-terrain]  bounds={mesh.bounds.tolist()}  "
      f"verts={len(mesh.vertices)}  faces={len(mesh.faces)}")

x_min, y_min, z_min = mesh.bounds[0]
x_max, y_max, z_max = mesh.bounds[1]
cell_w = (x_max - x_min) / GRID_N
cell_h = (y_max - y_min) / GRID_N
print(f"[m1-terrain] grid {GRID_N}x{GRID_N}  cell = {cell_w:.2f} x {cell_h:.2f} m")
print(f"[m1-terrain] subgrid {SUB_N}x{SUB_N} samples per cell -> "
      f"step = {cell_w/SUB_N:.2f} x {cell_h/SUB_N:.2f} m")

intr = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
RAY_Z = float(z_max + 100.0)
floor_z = float(z_min - 0.5)


def sample_heightfield(cx_min, cx_max, cy_min, cy_max, n):
    xs = np.linspace(cx_min, cx_max, n + 1)
    ys = np.linspace(cy_min, cy_max, n + 1)
    origins = np.stack([
        np.tile(xs[:, None], (1, n + 1)).ravel(),
        np.tile(ys[None, :], (n + 1, 1)).ravel(),
        np.full((n + 1) * (n + 1), RAY_Z),
    ], axis=1)
    direction = np.tile(np.array([[0.0, 0.0, -1.0]]), (origins.shape[0], 1))
    locs, idx_ray, _ = intr.intersects_location(
        origins, direction, multiple_hits=True
    )
    zs = np.full(origins.shape[0], np.nan)
    for ri, hit in zip(idx_ray, locs):
        cur = zs[ri]
        if np.isnan(cur) or hit[2] > cur:
            zs[ri] = hit[2]
    zs[np.isnan(zs)] = float(z_min)
    grid = np.zeros((n + 1, n + 1, 3))
    grid[:, :, 0] = xs[:, None]
    grid[:, :, 1] = ys[None, :]
    grid[:, :, 2] = zs.reshape(n + 1, n + 1)
    return grid


def build_prism(grid_xyz):
    n = grid_xyz.shape[0] - 1
    verts_top = grid_xyz.reshape(-1, 3)
    n_top = verts_top.shape[0]

    faces_top = []
    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v01 = i * (n + 1) + (j + 1)
            v10 = (i + 1) * (n + 1) + j
            v11 = (i + 1) * (n + 1) + (j + 1)
            faces_top.append([v00, v10, v11])
            faces_top.append([v00, v11, v01])
    faces_top = np.array(faces_top, dtype=np.int64)

    bb_min = grid_xyz[:, :, :2].min(axis=(0, 1))
    bb_max = grid_xyz[:, :, :2].max(axis=(0, 1))
    base = np.array([
        [bb_min[0], bb_min[1], floor_z],
        [bb_max[0], bb_min[1], floor_z],
        [bb_max[0], bb_max[1], floor_z],
        [bb_min[0], bb_max[1], floor_z],
    ])
    base_idx = n_top + np.arange(4)
    base_tris = np.array([
        [base_idx[0], base_idx[2], base_idx[1]],
        [base_idx[0], base_idx[3], base_idx[2]],
    ])

    side_tris = []
    for j in range(n):
        a = 0 * (n + 1) + j
        b = 0 * (n + 1) + (j + 1)
        side_tris.append([a, b, base_idx[0]])
        side_tris.append([b, base_idx[3], base_idx[0]])
    for j in range(n):
        a = n * (n + 1) + j
        b = n * (n + 1) + (j + 1)
        side_tris.append([a, base_idx[1], b])
        side_tris.append([b, base_idx[1], base_idx[2]])
    for i in range(n):
        a = i * (n + 1) + 0
        b = (i + 1) * (n + 1) + 0
        side_tris.append([a, base_idx[0], b])
        side_tris.append([b, base_idx[0], base_idx[1]])
    for i in range(n):
        a = i * (n + 1) + n
        b = (i + 1) * (n + 1) + n
        side_tris.append([a, b, base_idx[3]])
        side_tris.append([b, base_idx[2], base_idx[3]])
    side_tris = np.array(side_tris, dtype=np.int64)

    verts = np.vstack([verts_top, base])
    faces = np.vstack([faces_top, side_tris, base_tris])
    return verts, faces


cells = []
t0 = time.time()
for ix in range(GRID_N):
    for iy in range(GRID_N):
        cx_min = x_min + ix * cell_w
        cx_max = cx_min + cell_w
        cy_min = y_min + iy * cell_h
        cy_max = cy_min + cell_h

        grid_xyz = sample_heightfield(cx_min, cx_max, cy_min, cy_max, SUB_N)
        verts, faces = build_prism(grid_xyz)

        prism = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        cell_name = f"cell_{ix:02d}_{iy:02d}.obj"
        prism.export(os.path.join(OUT_DIR, cell_name))

        cells.append({
            "ix": ix, "iy": iy,
            "xy_bounds": [float(cx_min), float(cy_min),
                          float(cx_max), float(cy_max)],
            "z_top_max": float(grid_xyz[:, :, 2].max()),
            "z_top_min": float(grid_xyz[:, :, 2].min()),
            "piece_rel": f"module1/terrain_grid/{cell_name}",
        })

dt_total = time.time() - t0
print(f"[m1-terrain] DONE in {dt_total:.1f}s: {len(cells)} cells "
      f"(avg cell z-range = "
      f"{np.mean([c['z_top_max']-c['z_top_min'] for c in cells]):.2f} m)")

keys["terrain_grid"] = {
    "grid_n": GRID_N,
    "sub_n": SUB_N,
    "cell_w": cell_w,
    "cell_h": cell_h,
    "x_min": float(x_min), "x_max": float(x_max),
    "y_min": float(y_min), "y_max": float(y_max),
    "floor_z": floor_z,
    "n_cells": len(cells),
    "cells": cells,
}
with open(os.path.join(M1_DIR, "keys.json"), "w") as f:
    json.dump(keys, f, indent=2)
print(f"[m1-terrain] keys.json updated")
