"""Generic Code2Worlds scene.blend -> MuJoCo-ready OBJ + manifest.

Replaces the scene_1-specific export_scene_to_obj.py. Every Blender
collection / object is classified via sim/asset_registry.py into one
of the asset classes:

  background_terrain  background_floor  background_wall
  prop_concave_large  prop_convex_small  prop_dynamic
  vegetation          decoration         skip

Per class:
  - background_terrain / floor : visual + decimated collision OBJ
  - background_wall            : single static OBJ
  - prop_concave_large /
    prop_convex_small /
    prop_dynamic               : per-placeholder OBJ + (pos, quat, size)
  - vegetation, decoration     : recorded in manifest but no OBJ
                                 (visual-only; rendered by Blender,
                                 not by MuJoCo)
  - skip                       : entirely ignored (liquid, atmosphere,
                                 emitter helpers, scatter pointers)

Output:
  out_dir/
    {bg_class}_visual.obj      e.g. background_terrain_visual.obj
    {bg_class}_collision.obj
    props/{class}/{name}.obj   per-prop placeholder mesh
    manifest.json              {groups: {class: [{name, mesh_rel, pos, quat, size}, ...]}}

Run inside the c2w conda env:
    python sim/export_scene_to_obj_v2.py BLEND OUT_DIR

(Or via Blender if you don't have bpy in your env:
    blender BLEND --background --python sim/export_scene_to_obj_v2.py -- OUT_DIR)
"""
from __future__ import annotations

import bpy
import json
import os
import sys
from typing import Optional
from mathutils import Matrix, Vector

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from asset_registry import (  # noqa: E402
    GROUP_ORDER, classify_collection, classify_object,
)


# --- args ----------------------------------------------------------------
# When invoked as `blender BLEND --background --python script.py -- OUT_DIR`,
# Blender opens the .blend itself; we only need OUT_DIR after `--`.
# When invoked as `python script.py BLEND OUT_DIR` (with bpy in-env),
# we accept both args.
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
else:
    args = argv[1:]

if len(args) == 1:
    OUT_DIR = args[0]
    BLEND = bpy.data.filepath or "<from-cli>"
elif len(args) == 2:
    BLEND, OUT_DIR = args[0], args[1]
    print(f"[export] open {BLEND}")
    bpy.ops.wm.open_mainfile(filepath=BLEND)
else:
    print("usage:")
    print("  blender BLEND --background --python export_scene_to_obj_v2.py -- OUT_DIR")
    print("  python export_scene_to_obj_v2.py BLEND OUT_DIR")
    sys.exit(2)

BG_VIS_TRIS = int(os.environ.get("C2W_BG_VIS_TRIS", "20000"))
BG_COL_TRIS = int(os.environ.get("C2W_BG_COL_TRIS", "5000"))

os.makedirs(OUT_DIR, exist_ok=True)
print(f"[export] BLEND={BLEND} -> OUT_DIR={OUT_DIR}")


# --- helpers -------------------------------------------------------------
def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def world_aabb(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def write_obj_world(filepath, obj, in_world: bool = True):
    """Write evaluated mesh as OBJ. If in_world=False, write in object-local
    coordinates (which is what we want for placeholder props that get a
    pos+quat in the manifest)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    mw = obj.matrix_world if in_world else Matrix.Identity(4)
    if all(abs(mw[i][j]) < 1e-9 for i in range(4) for j in range(4)):
        mw = Matrix.Identity(4)
    n_verts = len(mesh.vertices); n_polys = len(mesh.polygons)
    with open(filepath, "w") as f:
        f.write(f"# C2W export: {obj.name} ({n_verts}v {n_polys}f)\n")
        for v in mesh.vertices:
            wv = mw @ v.co
            f.write(f"v {wv.x:.6f} {wv.y:.6f} {wv.z:.6f}\n")
        for poly in mesh.polygons:
            verts = [poly.vertices[i] + 1 for i in range(len(poly.vertices))]
            f.write("f " + " ".join(str(i) for i in verts) + "\n")
    ev.to_mesh_clear()
    return n_verts, n_polys


def decimate_copy(src_obj, target_tris):
    bpy.ops.object.select_all(action="DESELECT")
    src_obj.select_set(True)
    bpy.context.view_layer.objects.active = src_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    cur_tris = len(dup.data.polygons)
    if cur_tris > target_tris:
        mod = dup.modifiers.new(name="Decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = max(target_tris / cur_tris, 0.01)
        bpy.context.view_layer.objects.active = dup
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return dup


# --- collect classified objects -----------------------------------------
# For each Blender mesh object, decide its asset class. Order:
#   1. classify_collection on each user_collection -> use first hit
#   2. fall back to classify_object on the object name (handles
#      Infinigen indoor scenes where placeholders sit in generic
#      `Collection` / `placeholders` / `unique_assets` containers).
def asset_class_of(obj) -> Optional[str]:
    """Decide an object's asset class.

    Priority:
      1. classify each user_collection; first non-skip class wins.
         (`RigidBodyWorld` / `Collection` etc. are pure containers
         and yield None or 'skip' -- we ignore them and try the next
         collection.)
      2. fall back to classify_object on the object name (catches
         Infinigen indoor scenes where all props sit in generic
         `Collection` / `placeholders` / `unique_assets`).
    """
    container_classes = {"skip", None}
    fallback_skip = False
    for coll in obj.users_collection:
        cls = classify_collection(coll.name)
        if cls is None:
            continue
        if cls == "skip":
            fallback_skip = True
            continue
        return cls
    name_cls = classify_object(obj.name)
    if name_cls is not None:
        return name_cls
    return "skip" if fallback_skip else None


groups: dict[str, list[dict]] = {cls: [] for cls in GROUP_ORDER}
groups["skip"] = []  # kept for reporting; not exported
SEEN_NAMES = set()


# --- emit handlers per class --------------------------------------------
def emit_background_terrain(obj, idx) -> dict:
    """Outdoor terrain: visual + decimated collision. One per scene."""
    safe = f"background_terrain_{idx:02d}"
    bb = world_aabb(obj)
    vis = decimate_copy(obj, BG_VIS_TRIS)
    select_only(vis); vis_path = os.path.join(OUT_DIR, f"{safe}_visual.obj")
    write_obj_world(vis_path, vis, in_world=True)
    bpy.ops.object.delete()
    col = decimate_copy(obj, BG_COL_TRIS)
    select_only(col); col_path = os.path.join(OUT_DIR, f"{safe}_collision.obj")
    write_obj_world(col_path, col, in_world=True)
    bpy.ops.object.delete()
    return {
        "name": safe, "source_name": obj.name,
        "visual_obj": os.path.basename(vis_path),
        "collision_obj": os.path.basename(col_path),
        "world_aabb": list(bb),
    }


def emit_background_floor_or_wall(obj, group_subdir, idx) -> dict:
    """Indoor floor/wall: single static collider OBJ, in world coords."""
    bb = world_aabb(obj)
    safe_name = obj.name.replace("/", "_").replace(" ", "_").replace(".", "_")
    name = f"{group_subdir}_{idx:02d}_{safe_name}"[:80]
    sub = os.path.join(OUT_DIR, group_subdir)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{name}.obj")
    write_obj_world(path, obj, in_world=True)
    return {
        "name": name, "source_name": obj.name,
        "mesh_rel": f"{group_subdir}/{name}.obj",
        "world_aabb": list(bb),
    }


def emit_prop(obj, group_subdir, idx) -> Optional[dict]:
    """Prop / dynamic candidate: write OBJ in object-local coords + record
    world (pos, quat, size) so MJCF can place it."""
    bb = world_aabb(obj)
    size = (bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2])
    if min(size) < 0.01:
        return None
    loc = obj.matrix_world.translation
    quat = obj.matrix_world.to_quaternion()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.matrix_world.identity()
    select_only(dup)
    safe_name = obj.name.replace("/", "_").replace(" ", "_").replace(".", "_")
    name = f"{group_subdir}_{idx:03d}_{safe_name}"[:80]
    sub = os.path.join(OUT_DIR, group_subdir)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{name}.obj")
    write_obj_world(path, dup, in_world=False)
    n_tris = len(dup.data.polygons)
    bpy.ops.object.delete()
    return {
        "name": name, "source_name": obj.name,
        "mesh_rel": f"{group_subdir}/{name}.obj",
        "world_pos": [loc.x, loc.y, loc.z],
        "world_quat": [quat.w, quat.x, quat.y, quat.z],
        "world_size": list(size), "n_tris": n_tris,
    }


# --- main pass ----------------------------------------------------------
counters = {cls: 0 for cls in GROUP_ORDER}
for obj in bpy.data.objects:
    if obj.type != "MESH" or obj.data is None:
        continue
    if obj.name in SEEN_NAMES:
        continue
    SEEN_NAMES.add(obj.name)
    cls = asset_class_of(obj)
    if cls is None or cls == "skip":
        groups.setdefault("skip", []).append(
            {"name": obj.name, "reason": cls or "unmatched"})
        continue

    try:
        if cls == "background_terrain":
            entry = emit_background_terrain(obj, counters[cls])
        elif cls == "background_floor":
            entry = emit_background_floor_or_wall(obj, "floor", counters[cls])
        elif cls == "background_wall":
            entry = emit_background_floor_or_wall(obj, "wall", counters[cls])
        elif cls in ("prop_concave_large", "prop_convex_small", "prop_dynamic"):
            entry = emit_prop(obj, cls, counters[cls])
        elif cls in ("vegetation", "decoration"):
            # Visual-only: record metadata, no OBJ.
            bb = world_aabb(obj)
            loc = obj.matrix_world.translation
            quat = obj.matrix_world.to_quaternion()
            entry = {
                "name": obj.name, "source_name": obj.name,
                "world_pos": [loc.x, loc.y, loc.z],
                "world_quat": [quat.w, quat.x, quat.y, quat.z],
                "world_size": [bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2]],
            }
        else:
            entry = None
    except Exception as e:
        print(f"[export] WARN emit failed for {obj.name} ({cls}): {e}")
        entry = None

    if entry is not None:
        groups[cls].append(entry)
        counters[cls] += 1


# --- cameras -----------------------------------------------------------
cameras = []
for o in bpy.data.objects:
    if o.type == "CAMERA":
        loc = o.matrix_world.translation
        quat = o.matrix_world.to_quaternion()
        cameras.append({
            "name": o.name,
            "world_pos": [loc.x, loc.y, loc.z],
            "world_quat": [quat.w, quat.x, quat.y, quat.z],
        })


manifest = {
    "blend": os.path.abspath(BLEND),
    "groups": groups,
    "cameras": cameras,
}

# Backward-compat shims so existing module1_* scripts keep working.
# scene_1 hardwired manifest["terrain"] / manifest["boulders"] / manifest["vegetation"].
if groups["background_terrain"]:
    t = groups["background_terrain"][0]
    manifest["terrain"] = {
        "visual_obj": t["visual_obj"],
        "collision_obj": t["collision_obj"],
        "world_aabb": t["world_aabb"],
    }
manifest["boulders"] = []
for entry in groups["prop_concave_large"]:
    e = dict(entry); e.setdefault("mesh_rel", entry.get("mesh_rel"))
    manifest["boulders"].append(e)
manifest["vegetation"] = list(groups["vegetation"])
manifest["dynamic_candidates"] = list(groups["prop_dynamic"])

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print()
for cls in GROUP_ORDER:
    print(f"[export] {cls:22s}: {len(groups[cls])}")
print(f"[export] skip                  : {len(groups['skip'])}")
print(f"[export] cameras               : {len(cameras)}")
print(f"[export] -> {OUT_DIR}/manifest.json")
print("[export] DONE")

