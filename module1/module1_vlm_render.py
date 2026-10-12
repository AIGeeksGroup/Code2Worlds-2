"""Module 1, sub-step 4a: render 4 views per KEY boulder for VLM input.

We use matplotlib trisurf because:
  - boulders are low-poly (100-200 tris), trisurf handles them in <0.5s each
  - works headless without EGL/OpenGL context (matters on cloud VMs where
    pyrender/Blender headless can be flaky)
  - the VLM doesn't need PBR-accurate shading; it needs shape cues + size

For each KEY boulder we render 4 views:
  iso, front (+X), side (+Y), top (+Z)
into module1/vlm_views/<boulder_name>/view_NN.png at 480x480 px.

Run:
    python sim/module1_vlm_render.py EXPORT_DIR
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


if len(sys.argv) < 2:
    print("usage: module1_vlm_render.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
M1_DIR = os.path.join(EXPORT_DIR, "module1")
keys_path = os.path.join(M1_DIR, "keys.json")
with open(keys_path) as f:
    keys = json.load(f)

VIEW_DIR = os.path.join(M1_DIR, "vlm_views")
os.makedirs(VIEW_DIR, exist_ok=True)

# (elev_deg, azim_deg, label)
VIEWS = [
    (25,  45, "iso"),
    (10,   0, "front"),
    (10,  90, "side"),
    (89,   0, "top"),
]


def render_boulder(mesh: trimesh.Trimesh, out_dir: str, name: str) -> list[str]:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    cen = (mins + maxs) / 2
    extent = (maxs - mins).max() * 0.55  # slight padding

    out_paths = []
    for elev, azim, label in VIEWS:
        fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces,
            color=(0.55, 0.50, 0.45),
            shade=True,
            edgecolor=(0.15, 0.15, 0.15),
            linewidth=0.15,
            antialiased=True,
        )
        ax.set_xlim(cen[0] - extent, cen[0] + extent)
        ax.set_ylim(cen[1] - extent, cen[1] + extent)
        ax.set_zlim(cen[2] - extent, cen[2] + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        # tight to mesh
        out_path = os.path.join(out_dir, f"view_{label}.png")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02,
                    facecolor=(0.95, 0.95, 0.95))
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


print(f"[m1-vlm-render] rendering 4 views per boulder for "
      f"{len(keys['key_boulders'])} key boulders")

for i, b in enumerate(keys["key_boulders"]):
    name = b["name"]
    src = os.path.join(EXPORT_DIR, b["mesh_rel"])
    if not os.path.exists(src):
        print(f"[m1-vlm-render] SKIP {name}: missing {src}")
        continue
    mesh = trimesh.load(src, force="mesh")
    if mesh.vertices.shape[0] == 0:
        print(f"[m1-vlm-render] SKIP {name}: empty mesh")
        continue
    # Subsample faces for huge indoor meshes (matplotlib trisurf scales O(n)).
    if len(mesh.faces) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(mesh.faces), size=5000, replace=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx],
                               process=False)
        print(f"[m1-vlm-render]  subsampled {name} to {len(mesh.faces)} faces")
    out_dir = os.path.join(VIEW_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    paths = render_boulder(mesh, out_dir, name)
    print(f"[m1-vlm-render]  #{i:02d} {name}: 4 views -> {out_dir}")

print(f"[m1-vlm-render] DONE -> {VIEW_DIR}")
