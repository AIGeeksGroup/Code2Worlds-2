"""Module 1 emit MJCF (generic / scene-agnostic).

Driven by manifest["groups"] from export_scene_to_obj_v2.py:
  - background_terrain : per-cell convex prism (terrain_grid) if available,
                         else single mesh
  - background_floor   : single mesh per floor (already near-convex)
  - background_wall    : single mesh per wall
  - prop_concave_large : KEY -> CoACD pieces + VLM friction;
                         NON-KEY -> visual mesh as collider

Outputs:
  <export_dir>/module1/scene_default.xml
  <export_dir>/module1/scene_enriched.xml

The "default" baseline is what MuJoCo would do with the visual meshes
unmodified (every concave prop = its own convex hull). The "enriched"
variant applies the registry-driven collision plan from Module 1.

Run:
    python sim/module1_emit_mjcf_v2.py EXPORT_DIR
"""
from __future__ import annotations

import json
import os
import sys


if len(sys.argv) < 2:
    print("usage: module1_emit_mjcf_v2.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
M1_DIR = os.path.join(EXPORT_DIR, "module1")
os.makedirs(M1_DIR, exist_ok=True)

with open(os.path.join(EXPORT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
with open(os.path.join(M1_DIR, "keys.json")) as f:
    keys = json.load(f)

# Backward compatibility: scene_1 was exported with v1 (no `groups`).
# Translate v1's flat schema (terrain + boulders) into the v2 groups view.
if "groups" not in manifest:
    print("[m1-emit-v2] manifest is v1-flat; building groups shim")
    g = {"background_terrain": [], "background_floor": [],
         "background_wall": [], "prop_concave_large": [],
         "prop_convex_small": [], "prop_dynamic": [],
         "vegetation": [], "decoration": []}
    if "terrain" in manifest:
        t = dict(manifest["terrain"])
        t.setdefault("name", "background_terrain_00")
        g["background_terrain"].append(t)
    g["prop_concave_large"] = list(manifest.get("boulders", []))
    g["vegetation"] = list(manifest.get("vegetation", []))
    manifest["groups"] = g

groups = manifest["groups"]
key_names = {p["name"] for p in keys["key_boulders"]}
coacd_by_name = {r["name"]: r for r in keys.get("coacd", {}).get("results", [])}
phys_by_name = keys.get("physical", {})

print(f"[m1-emit-v2] groups: " + ", ".join(
    f"{g}={len(groups.get(g, []))}" for g in
    ("background_terrain", "background_floor", "background_wall",
     "prop_concave_large", "prop_dynamic")))
print(f"[m1-emit-v2] {len(key_names)} KEY props with CoACD={len(coacd_by_name)}")


def pos_str(xyz): return " ".join(f"{x:.6f}" for x in xyz)
def quat_str(q):  return " ".join(f"{x:.6f}" for x in q)


def safe_mesh_name(s: str) -> str:
    """MJCF mesh names cannot contain '/' or special chars; use the obj
    file basename."""
    return os.path.splitext(os.path.basename(s))[0].replace(".", "_")


def emit_thin_static_aabb(asset_lines, geom_lines, group_name, prefix,
                          friction):
    """Indoor floor / wall meshes are 2D planes (coplanar verts). MuJoCo
    refuses to convex-hull those, so we substitute a thin axis-aligned box
    sized to the world AABB. We pad the smallest extent up to MIN_THICK so
    contact has measurable depth; physically equivalent to a wall/floor.
    """
    MIN_THICK = 0.01
    for i, e in enumerate(groups.get(group_name, [])):
        bb = e.get("world_aabb")
        if not bb:
            continue
        full_extents = (bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2])
        # Some procedural indoor/industrial rooms export all walls as one shell
        # mesh. Its AABB covers the room volume; emitting that as a box collider
        # would make dynamic objects collide with "air" inside the room. Keep
        # only genuinely thin structural surfaces here.
        if group_name == "background_wall" and min(full_extents) > 0.25:
            print(f"[m1-emit-v2] skip non-thin wall shell {e.get('name', i)} "
                  f"extents={[round(v, 3) for v in full_extents]}")
            continue
        sx, sy, sz = full_extents[0]/2, full_extents[1]/2, full_extents[2]/2
        cx, cy, cz = (bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2
        sx = max(sx, MIN_THICK); sy = max(sy, MIN_THICK); sz = max(sz, MIN_THICK)
        name = f"{prefix}_{i:02d}"
        geom_lines.append(
            f'<geom name="{name}" type="box" '
            f'pos="{pos_str([cx, cy, cz])}" '
            f'size="{pos_str([sx, sy, sz])}" '
            'contype="2" conaffinity="1" '
            f'friction="{friction:.3f} 0.005 0.0001" condim="3" '
            'rgba="0.7 0.65 0.55 1"/>')


def emit_background(asset_lines, geom_lines, default=False):
    """Terrain (mesh) + floor/wall (axis-aligned box from AABB).

    For default: terrain uses one mesh, no terrain_grid.
    For enriched: terrain uses per-cell prisms if available (grid).
    Indoor floor/wall always use box (2D mesh would fail convexification).
    """
    # 1. terrain (outdoor)
    hf = keys.get("terrain_hfield")
    tg = keys.get("terrain_grid")
    if hf and not default:
        hx = (float(hf["x_max"]) - float(hf["x_min"])) / 2
        hy = (float(hf["y_max"]) - float(hf["y_min"])) / 2
        hz = float(hf["z_max"]) - float(hf["z_min"])
        cx = (float(hf["x_max"]) + float(hf["x_min"])) / 2
        cy = (float(hf["y_max"]) + float(hf["y_min"])) / 2
        with open(os.path.join(EXPORT_DIR, hf["elevation_rel"])) as f:
            elevation = f.read().strip()
        asset_lines.append(f'<hfield name="terrain_hfield" nrow="{int(hf["resolution"])}" ncol="{int(hf["resolution"])}" size="{hx:.6f} {hy:.6f} {hz:.6f} {float(hf.get("base_depth", 10.0)):.6f}" elevation="{elevation}"/>')
        geom_lines.append(f'<geom name="terrain_hfield" type="hfield" hfield="terrain_hfield" pos="{cx:.6f} {cy:.6f} {float(hf["z_min"]):.6f}" contype="2" conaffinity="1" friction="0.9 0.005 0.0001" condim="4"/>')
    elif tg and not default:
        for c in tg["cells"]:
            name = f"terrain_{c['ix']:02d}_{c['iy']:02d}"
            abs_path = os.path.join(EXPORT_DIR, c["piece_rel"])
            asset_lines.append(f'<mesh name="{name}" file="{abs_path}"/>')
            geom_lines.append(
                f'<geom name="{name}" type="mesh" mesh="{name}" '
                'contype="2" conaffinity="1" '
                'friction="0.9 0.005 0.0001" condim="3"/>')
    else:
        for i, t in enumerate(groups.get("background_terrain", [])):
            mesh_rel = (t.get("collision_obj") or t.get("visual_obj")
                        or t.get("mesh_rel"))
            if not mesh_rel:
                continue
            mesh_path = os.path.join(EXPORT_DIR, mesh_rel)
            name = f"terrain_{i:02d}"
            asset_lines.append(f'<mesh name="{name}" file="{mesh_path}"/>')
            geom_lines.append(
                f'<geom name="{name}" type="mesh" mesh="{name}" '
                'contype="2" conaffinity="1" '
                'friction="0.9 0.005 0.0001" condim="3"/>')

    # 2. floor + wall (indoor): thin AABB box -- avoids coplanar-mesh error
    emit_thin_static_aabb(asset_lines, geom_lines,
                          "background_floor", "floor", friction=0.7)
    emit_thin_static_aabb(asset_lines, geom_lines,
                          "background_wall", "wall", friction=0.5)


def emit_props(asset_lines, geom_lines, default=False):
    """Static props: KEY -> CoACD pieces; fallback -> AABB box collider."""
    for p in groups.get("prop_concave_large", []) + groups.get("prop_convex_small", []):
        name = safe_mesh_name(p["name"])
        px, py, pz = p["world_pos"]
        qw, qx, qy, qz = p["world_quat"]

        if (not default) and p["name"] in key_names and p["name"] in coacd_by_name:
            pieces_rel = coacd_by_name[p["name"]]["pieces_rel"]
            friction = phys_by_name.get(p["name"], {}).get(
                "vlm", {}).get("friction", 0.6)
            for j, rel in enumerate(pieces_rel):
                abs_path = os.path.join(EXPORT_DIR, rel)
                mesh_name = f"{name}_col_{j:02d}"
                asset_lines.append(
                    f'<mesh name="{mesh_name}" file="{abs_path}"/>')
                geom_lines.append(
                    f'<geom name="{mesh_name}" type="mesh" mesh="{mesh_name}" '
                    f'pos="{pos_str([px, py, pz])}" '
                    f'quat="{quat_str([qw, qx, qy, qz])}" '
                    'contype="2" conaffinity="1" '
                    f'friction="{friction:.3f} 0.005 0.0001" '
                    f'solref="0.02 1" solimp="0.95 0.99 0.001" condim="4"/>')
        else:
            sx, sy, sz = [max(float(v) / 2.0, 0.005) for v in p.get("world_size", [0.05, 0.05, 0.05])]
            geom_lines.append(
                f'<geom name="{name}" type="box" '
                f'pos="{pos_str([px, py, pz])}" '
                f'size="{pos_str([sx, sy, sz])}" '
                'contype="2" conaffinity="1" '
                'friction="0.6 0.005 0.0001" condim="3"/>')


def emit_xml(out_path: str, model_name: str, default: bool):
    asset_lines: list[str] = []
    geom_lines: list[str] = []
    emit_background(asset_lines, geom_lines, default=default)
    emit_props(asset_lines, geom_lines, default=default)

    asset_block = "\n    ".join(asset_lines)
    geom_block = "\n    ".join(geom_lines)
    xml = f"""<mujoco model="{model_name}">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.7 0.65 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    {asset_block}
  </asset>
  <custom>
    <text name="mobile_robot_model" data="clearpath_jackal_ugv"/>
  </custom>
  <worldbody>
    <light name="sun" pos="0 0 30" dir="-0.3 -0.2 -1" directional="true"
           diffuse="0.9 0.85 0.7" specular="0.2 0.2 0.2"/>
    {geom_block}
  </worldbody>
</mujoco>
"""
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"[m1-emit-v2] wrote {model_name} -> {out_path} "
          f"({len(asset_lines)} meshes, {len(geom_lines)} geoms)")


emit_xml(os.path.join(M1_DIR, "scene_default.xml"), "scene_default", default=True)
emit_xml(os.path.join(M1_DIR, "scene_enriched.xml"), "scene_enriched", default=False)
print("[m1-emit-v2] DONE")
