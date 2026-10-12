"""Generic postproc for any articulated robot + dynamic props.

Rewrites rollout_log.json to absorb MuJoCo's mesh_pos/mesh_quat transforms,
producing rollout_log_blender.json that Blender can use directly.

Works with:
- Any articulated robot (Go2, Franka, UR5, Spot, etc.)
- Multiple robots in the same scene
- Any dynamic props (Bottle, stone, etc.)

Usage:
    python module2/postproc_rollout_blender.py EXPORT_DIR

Expects:
    EXPORT_DIR/module2/rollout_log.json
    EXPORT_DIR/module2/scene_m2.xml

Outputs:
    EXPORT_DIR/module2/rollout_log_blender.json
"""
import sys
import json
import os
import numpy as np
import mujoco

if len(sys.argv) < 2:
    print("usage: postproc_rollout_blender.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = sys.argv[1]
M2_DIR = os.path.join(EXPORT_DIR, "module2")
LOG_JSON = os.path.join(M2_DIR, "rollout_log.json")
SCENE_XML = os.environ.get("C2W_M2_SCENE_XML", os.path.join(M2_DIR, "scene_m2.xml"))
OUT_JSON = os.path.join(M2_DIR, "rollout_log_blender.json")

if not os.path.exists(LOG_JSON):
    print(f"ERROR: {LOG_JSON} not found")
    sys.exit(1)
if not os.path.exists(SCENE_XML):
    print(f"ERROR: {SCENE_XML} not found")
    sys.exit(1)

with open(LOG_JSON) as f:
    log = json.load(f)
model = mujoco.MjModel.from_xml_path(SCENE_XML)

# Build mesh_name -> (pos, quat) from the COMPILED model
mesh_xform = {}
for mid in range(model.nmesh):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
    mesh_xform[name] = (model.mesh_pos[mid].copy(), model.mesh_quat[mid].copy())

print(f"[postproc-generic] loaded {len(mesh_xform)} meshes from {SCENE_XML}")

# Auto-detect all robot visual geom fields in log
# Pattern: <robot_name>_visual_geoms / <robot_name>_geom_xforms
robot_fields = {}
for key in log.keys():
    if key.endswith("_visual_geoms"):
        robot_name = key[:-len("_visual_geoms")]
        xform_key = f"{robot_name}_geom_xforms"
        if xform_key in log["frames"][0]:
            robot_fields[robot_name] = {
                "visual_geoms": log[key],
                "xform_key": xform_key
            }
            print(f"[postproc-generic] detected robot: {robot_name} ({len(log[key])} visual geoms)")

if not robot_fields:
    print("[postproc-generic] WARNING: no robot visual geoms found in log")
    print("  Expected keys like 'go2_visual_geoms' / 'franka_visual_geoms'")

# Prepare mesh transforms for each robot
for robot_name, info in robot_fields.items():
    vis = info["visual_geoms"]
    need_pos = []
    need_quat = []
    for g in vis:
        mesh_name = g["mesh_name"]
        if mesh_name in mesh_xform:
            mp, mq = mesh_xform[mesh_name]
            need_pos.append(mp)
            need_quat.append(mq)
        else:
            print(f"[postproc-generic] WARNING: mesh '{mesh_name}' not found in model")
            need_pos.append(np.zeros(3))
            need_quat.append(np.array([1, 0, 0, 0]))
    info["need_pos"] = need_pos
    info["need_quat"] = need_quat

# Auto-detect dynamic prop fields
# Pattern: stone_pos / stone_quat (legacy) or <prop_name>_pos / <prop_name>_quat
prop_fields = []
robot_prefixes = tuple(f"{name}_" for name in robot_fields)
for key in log["frames"][0].keys():
    if (key.endswith("_pos") and not key.startswith(robot_prefixes)
            and key != "go2_base_pos"):
        prop_name = key[:-len("_pos")]
        quat_key = f"{prop_name}_quat"
        if quat_key in log["frames"][0]:
            # Find mesh name (try common patterns)
            mesh_candidates = [
                f"kick_{prop_name}_visual",
                f"{prop_name}_visual",
                prop_name
            ]
            mesh_name = None
            for cand in mesh_candidates:
                if cand in mesh_xform:
                    mesh_name = cand
                    break

            if mesh_name is None:
                try:
                    body_id = model.body(prop_name).id
                except KeyError:
                    body_id = -1
                for geom_id in range(model.ngeom):
                    if int(model.geom_bodyid[geom_id]) != body_id:
                        continue
                    mesh_id = int(model.geom_dataid[geom_id])
                    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH and mesh_id >= 0:
                        mesh_name = model.mesh(mesh_id).name
                        break

            if mesh_name:
                mp, mq = mesh_xform[mesh_name]
                prop_fields.append({
                    "name": prop_name,
                    "pos_key": key,
                    "quat_key": quat_key,
                    "mesh_pos": mp,
                    "mesh_quat": mq
                })
                print(f"[postproc-generic] detected prop: {prop_name} (mesh: {mesh_name})")
            else:
                prop_fields.append({
                    "name": prop_name, "pos_key": key, "quat_key": quat_key,
                    "mesh_pos": np.zeros(3),
                    "mesh_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                })
                print(f"[postproc-generic] prop '{prop_name}' has no mesh transform; using body pose")

# Per-frame rewrite
qbuf = np.zeros(4)
rotated = np.zeros(3)
n_frames = len(log["frames"])

for fi, fr in enumerate(log["frames"]):
    # Process each robot
    for robot_name, info in robot_fields.items():
        xform_key = info["xform_key"]
        need_pos = info["need_pos"]
        need_quat = info["need_quat"]

        new_xf = []
        for i, x in enumerate(fr[xform_key]):
            gp = np.array(x["pos"])
            gq = np.array(x["quat"])
            mp = need_pos[i]
            mq = need_quat[i]

            # The Jackal STL files keep their source vertex coordinates.
            # Recreate MuJoCo's compiled mesh placement as G * M: translate
            # mesh_pos by the geom frame (not by the already-combined frame),
            # then compose rotations.  The old generic inverse was valid only
            # for the legacy exported OBJ convention and tore this assembly apart.
            mujoco.mju_mulQuat(qbuf, gq, mq)
            if robot_name == "jackal":
                mujoco.mju_rotVecQuat(rotated, mp, gq)
                bl_pos = gp + rotated
            else:
                mujoco.mju_rotVecQuat(rotated, mp, qbuf)
                bl_pos = gp - rotated
            new_xf.append({"pos": bl_pos.tolist(), "quat": qbuf.tolist()})

        fr[xform_key] = new_xf

    # Process each prop
    for prop in prop_fields:
        sp = np.array(fr[prop["pos_key"]])
        sq = np.array(fr[prop["quat_key"]])
        mp = prop["mesh_pos"]
        mq = prop["mesh_quat"]

        mujoco.mju_mulQuat(qbuf, sq, mq)
        mujoco.mju_rotVecQuat(rotated, mp, qbuf)
        fr[prop["pos_key"]] = (sp - rotated).tolist()
        fr[prop["quat_key"]] = qbuf.tolist()

    if (fi + 1) % 25 == 0 or fi == n_frames - 1:
        print(f"[postproc-generic] processed {fi + 1}/{n_frames} frames")

with open(OUT_JSON, "w") as f:
    json.dump(log, f)

print(f"[postproc-generic] DONE -> {OUT_JSON}")
print(f"  Processed {len(robot_fields)} robot(s): {list(robot_fields.keys())}")
print(f"  Processed {len(prop_fields)} prop(s): {[p['name'] for p in prop_fields]}")
