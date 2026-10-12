"""Module 2 emit MJCF (generic / scene-agnostic).

Driven by manifest["groups"] from export_scene_to_obj_v2.py and
keys.json (with optional terrain_grid for outdoor scenes). Two
dynamics_program schemas are supported:

  schema A (outdoor "kick_stone" pattern):
    {"object": "kick_stone_0", "source_boulder": "<KEY name>",
     "size_scale": 0.12, "mass": 2.0, "friction": 0.7,
     "initial_pose": {"pos": [...], "quat": [...]},
     "initial_velocity": [...], "initial_angular_velocity": [...]}

  schema B (indoor "reuse existing dynamic prop"):
    {"object": "bottle_0", "reuse_existing": "<prop_dynamic name>",
     "mass": 0.35, "friction": 0.45,
     "initial_pose": {...}, "initial_velocity": [...], ...}
  -> uses the prop's own visual mesh as both visual and collision
     (Bottle is already convex enough; if it is concave we fall back
     to a CoACD pass at a future date).

The static scene is emitted from manifest["groups"] (terrain_grid if
available, else background_terrain mesh; always background_floor +
background_wall + prop_concave_large + prop_convex_small). Decoration
and vegetation are not emitted (visual-only).

Output:
  EXPORT_DIR/module2/scene_m2.xml
  EXPORT_DIR/module2/go2_local/   (localized Go2 with absolute mesh paths)

Run:
    python sim/module2_emit_dynamic_v2.py EXPORT_DIR GO2_DIR
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import coacd
import numpy as np
import trimesh
from xml.sax.saxutils import escape

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module1"))
from asset_registry import material_from_name  # noqa: E402


if len(sys.argv) < 3:
    print("usage: module2_emit_dynamic_v2.py EXPORT_DIR GO2_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
GO2_DIR = os.path.abspath(sys.argv[2])
M1_DIR = os.path.join(EXPORT_DIR, "module1")
M2_DIR = os.path.join(EXPORT_DIR, "module2")
os.makedirs(M2_DIR, exist_ok=True)

with open(os.path.join(EXPORT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
with open(os.path.join(M1_DIR, "keys.json")) as f:
    keys = json.load(f)
with open(os.path.join(M2_DIR, "dynamics_program.json")) as f:
    program = json.load(f)


# Backward compat: scene_1's manifest is v1-flat (no `groups`). Synthesize.
if "groups" not in manifest:
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
dyn_props_by_name = {p["name"]: p for p in groups.get("prop_dynamic", [])}

# Pick a static-scene reference for ground snapping.
# Outdoor: terrain mesh ray-cast.
# Indoor: collect floor AABBs and use max-z under (x,y) lookup.
TERRAIN_RAY = None
FLOOR_BOXES = []
if groups.get("background_terrain"):
    t = groups["background_terrain"][0]
    terrain_rel = (t.get("collision_obj") or t.get("visual_obj")
                   or t.get("mesh_rel"))
    if terrain_rel:
        terrain_abs = os.path.join(EXPORT_DIR, terrain_rel)
        try:
            tm = trimesh.load(terrain_abs, force="mesh")
            TERRAIN_RAY = trimesh.ray.ray_triangle.RayMeshIntersector(tm)
            print(f"[m2-emit-v2] terrain ray-cast on {terrain_rel}")
        except Exception as e:
            print(f"[m2-emit-v2] WARN terrain ray-cast unavailable: {e}")
for fl in groups.get("background_floor", []):
    bb = fl.get("world_aabb")
    if bb:
        FLOOR_BOXES.append(bb)
print(f"[m2-emit-v2] {len(FLOOR_BOXES)} floor box(es), "
      f"terrain_ray={'yes' if TERRAIN_RAY else 'no'}")
def get_or_create_coacd_terrain_cell(name: str, abs_path: str) -> list[str]:
    """对单个地形网格进行快速在线减面与凸分解，返回生成的凸包相对路径列表"""
    terrain_col_dir = os.path.join(M2_DIR, "terrain_collision")
    os.makedirs(terrain_col_dir, exist_ok=True)
    
    # 1. 缓存机制：如果已经存在分解好的文件，直接返回
    existing_pieces = []
    for j in range(4):  # max_convex_hull = 4
        piece_name = f"{name}_col_{j:02d}.obj"
        piece_path = os.path.join(terrain_col_dir, piece_name)
        if os.path.exists(piece_path):
            existing_pieces.append(f"terrain_collision/{piece_name}")
            
    if len(existing_pieces) > 0:
        return existing_pieces
        
    # 尝试加载网格
    mesh = trimesh.load(abs_path, force="mesh")
    if len(mesh.vertices) == 0:
        return []
        
    # 2. 在线自动减面 (需要环境中有 scipy 或 fast-simplification)
    target_faces = 800
    if len(mesh.faces) > target_faces:
        try:
            if hasattr(mesh, "simplify_quadric_decimation"):
                mesh = mesh.simplify_quadric_decimation(target_faces)
            elif hasattr(mesh, "simplify_quadratic_decimation"):
                mesh = mesh.simplify_quadratic_decimation(target_faces)
            print(f"  -> [{name}] 成功将三角面数减至 {len(mesh.faces)}")
        except Exception as e:
            # 如果缺少依赖会走到这里，我们通过优化 CoACD 参数来兜底速度
            pass

    # 3. 运行极速配置的 CoACD (针对减面可能失败的情况进行参数兜底)
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=0.15,                 # 适当放宽阈值
        max_convex_hull=4,
        preprocess_mode="auto",
        preprocess_resolution=30,       # 关键优化：降低预处理体素分辨率（默认50），可缩短数倍时间
        mcts_iterations=50,             # 关键优化：减少蒙特卡洛树搜索迭代次数（默认150），大幅提速
        mcts_max_depth=2,               # 关键优化：限制搜索深度（默认3）
    )
    
    pieces_rel = []
    for j, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        piece_name = f"{name}_col_{j:02d}.obj"
        piece_path = os.path.join(terrain_col_dir, piece_name)
        piece.export(piece_path)
        pieces_rel.append(f"terrain_collision/{piece_name}")
        
    return pieces_rel

def ground_z_at(x: float, y: float) -> float | None:
    """Topmost ground z under world (x, y). Tries terrain first, then
    indoor floor AABBs (any floor whose XY footprint contains the point)."""
    if TERRAIN_RAY is not None:
        origin = np.array([[x, y, 1000.0]])
        direction = np.array([[0.0, 0.0, -1.0]])
        locs, _, _ = TERRAIN_RAY.intersects_location(
            origin, direction, multiple_hits=True)
        if len(locs):
            return float(np.max(locs[:, 2]))
    candidates = []
    for bb in FLOOR_BOXES:
        if bb[0] - 0.01 <= x <= bb[3] + 0.01 and bb[1] - 0.01 <= y <= bb[4] + 0.01:
            candidates.append(bb[5])  # max-z of floor box
    if candidates:
        return max(candidates)
    return None


def pos_str(p): return " ".join(f"{x:.6f}" for x in p)
def quat_str(q): return " ".join(f"{x:.6f}" for x in q)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _is_industrial_arm_program(program: dict) -> bool:
    robot = program.get("robot", {})
    return (robot.get("kind") == "industrial_arm"
            or str(program.get("robot_model", "")).startswith("universal_robots"))


def _robot_control(program: dict, robot_dir: str) -> dict:
    robot = program.get("robot", {})
    ctrl = robot.get("control", {})
    waypoints = (ctrl.get("waypoints") or program.get("joint_waypoints")
                 or [[0.0, -1.45, 1.75, -1.85, -1.57, 0.0]])
    joint_names = ctrl.get("joint_names") or program.get("joint_names") or [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    return {
        "kind": "industrial_arm",
        "name": robot.get("name") or program.get("robot_name") or "ur5e",
        "model": robot.get("model") or program.get("robot_model") or "universal_robots_ur5e",
        "asset_dir": robot.get("asset_dir") or program.get("robot_asset_dir") or robot_dir,
        "base_pose": robot.get("base_pose") or program.get("base_pose") or {
            "pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
        "control": {
            "type": ctrl.get("type") or "joint_waypoints",
            "joint_names": joint_names,
            "waypoints": waypoints,
            "duration": float(ctrl.get("duration") or program.get("duration") or 4.0),
        }
    }


def _copy_robot_assets(robot_dir: str, out_dir: str) -> str:
    local = os.path.join(out_dir, "robot_local")
    if os.path.exists(local):
        shutil.rmtree(local)
    shutil.copytree(robot_dir, local)
    candidates = sorted(p for p in os.listdir(local) if p.endswith(".xml"))
    xml_name = "ur5e.xml" if "ur5e.xml" in candidates else candidates[0]
    robot_xml = os.path.join(local, xml_name)
    with open(robot_xml) as f:
        text = f.read()
    text = re.sub(r'\s+meshdir="assets"', "", text)
    assets_abs = os.path.join(local, "assets")

    def repl(m):
        head, fname = m.group(1), m.group(2)
        if "/" in fname:
            return m.group(0)
        return f'{head}file="{os.path.join(assets_abs, fname)}"'

    text = re.sub(r'(<mesh[^/>]*?)file="([^"]+)"', repl, text)
    with open(robot_xml, "w") as f:
        f.write(text)
    return robot_xml


def _static_proxy_geoms(groups: dict) -> list[str]:
    lines = []
    idx = 0
    for group_name in ("background_floor", "background_wall",
                       "prop_concave_large", "prop_convex_small"):
        for item in groups.get(group_name, []):
            bb = item.get("world_aabb")
            if not bb and item.get("world_pos") and item.get("world_size"):
                px, py, pz = item["world_pos"]
                sx, sy, sz = [float(v) / 2.0 for v in item["world_size"]]
                bb = [px - sx, py - sy, pz - sz, px + sx, py + sy, pz + sz]
            if not bb:
                continue
            ext = [float(bb[3] - bb[0]), float(bb[4] - bb[1]), float(bb[5] - bb[2])]
            if min(ext) < 0.008:
                continue
            if group_name == "background_wall" and min(ext) > 0.25:
                print(f"[m2-emit-v2] skip non-thin wall proxy {item.get('name', idx)}")
                continue
            cx, cy, cz = [(float(bb[i]) + float(bb[i + 3])) / 2.0 for i in range(3)]
            sx, sy, sz = [max(v / 2.0, 0.005) for v in ext]
            mat = "mat_floor" if group_name == "background_floor" else "mat_static"
            name = safe_name(item.get("name", f"{group_name}_{idx}"))[:80]
            lines.append(
                f'<geom name="{escape(name)}" type="box" pos="{pos_str([cx, cy, cz])}" '
                f'size="{pos_str([sx, sy, sz])}" material="{mat}" '
                'contype="2" conaffinity="1" friction="0.7 0.005 0.0001" condim="3"/>')
            idx += 1
    return lines


def _emit_industrial_arm_scene(robot_dir: str) -> None:
    robot_cfg = _robot_control(program, robot_dir)
    asset_dir = robot_cfg["asset_dir"]
    if (not os.path.isabs(asset_dir)) or (not os.path.exists(asset_dir)):
        print(f"[m2-emit-v2] robot asset_dir '{asset_dir}' unavailable; using CLI path {robot_dir}")
        asset_dir = robot_dir
    robot_xml = _copy_robot_assets(os.path.abspath(asset_dir), M2_DIR)
    robot_cfg["asset_dir"] = os.path.dirname(robot_xml)
    robot_cfg["xml"] = robot_xml
    program["robot"] = robot_cfg
    program.setdefault("robot_action", "weld_seam_trace")
    with open(os.path.join(M2_DIR, "dynamics_program.json"), "w") as f:
        json.dump(program, f, indent=2)

    static = "\n    ".join(_static_proxy_geoms(groups))
    q0 = robot_cfg["control"]["waypoints"][0]
    task_objects = program.get("task_objects") or []
    if not task_objects:
        task_objects = [
            {"name": "module2_weld_coupon_collision", "type": "box",
             "pos": [0.55, -1.13, 1.09], "size": [0.34, 0.12, 0.025]},
            {"name": "module2_weld_seam_marker", "type": "capsule_marker",
             "fromto": [0.24, -1.17, 1.125, 0.76, -0.90, 1.125], "size": 0.012},
        ]
    task_lines = []
    for obj in task_objects:
        if obj.get("type") == "box":
            task_lines.append(
                f'<geom name="{escape(obj["name"])}" type="box" '
                f'pos="{pos_str(obj["pos"])}" size="{pos_str(obj["size"])}" '
                'material="mat_weld_target" contype="2" conaffinity="1" friction="0.8 0.005 0.0001"/>')
        elif obj.get("type") == "capsule_marker":
            task_lines.append(
                f'<geom name="{escape(obj["name"])}" type="capsule" '
                f'fromto="{pos_str(obj["fromto"])}" size="{float(obj.get("size", 0.012)):.6f}" '
                'material="mat_seam" contype="0" conaffinity="0"/>')
    task_block = "\n    ".join(task_lines)
    xml = f"""<mujoco model="scene_m2_industrial_arm">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.62" ambient="0.28 0.28 0.28" specular="0.1 0.1 0.1"/>
    <global azimuth="135" elevation="-22" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <material name="mat_floor" rgba="0.35 0.36 0.35 1"/>
    <material name="mat_static" rgba="0.45 0.46 0.45 1"/>
    <material name="mat_weld_target" rgba="0.62 0.64 0.62 1"/>
    <material name="mat_seam" rgba="0.95 0.55 0.08 1"/>
  </asset>
  <worldbody>
    <light name="ceiling_light" pos="0 -1 4" dir="0 0 -1" directional="true"/>
    {static}
    {task_block}
  </worldbody>
  <include file="{robot_xml}"/>
  <keyframe>
    <key name="m2_init" qpos="{pos_str(q0)}" ctrl="{pos_str(q0)}"/>
  </keyframe>
</mujoco>
"""
    out_path = os.path.join(M2_DIR, "scene_m2.xml")
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"[m2-emit-v2] industrial arm -> {out_path}")
    print(f"[m2-emit-v2] robot_xml={robot_xml}")


if _is_industrial_arm_program(program):
    _emit_industrial_arm_scene(GO2_DIR)
    sys.exit(0)


asset_lines: list[str] = []
material_lines = [
    '<material name="mat_terrain" rgba="0.62 0.51 0.38 1"/>',
    '<material name="mat_floor"   rgba="0.78 0.66 0.50 1"/>',
    '<material name="mat_wall"    rgba="0.85 0.83 0.78 1"/>',
    '<material name="mat_prop_static"  rgba="0.55 0.42 0.30 1"/>',
    '<material name="mat_prop_dynamic" rgba="0.88 0.78 0.55 1"/>',
]
geom_lines: list[str] = []
body_lines: list[str] = []
keyframe_qpos: list[float] = []
keyframe_qvel: list[float] = []

# --- 1. static background -------------------------------------------------
hf = keys.get("terrain_hfield")
tg = keys.get("terrain_grid")
if hf:
    hx = (float(hf["x_max"]) - float(hf["x_min"])) / 2
    hy = (float(hf["y_max"]) - float(hf["y_min"])) / 2
    hz = float(hf["z_max"]) - float(hf["z_min"])
    cx = (float(hf["x_max"]) + float(hf["x_min"])) / 2
    cy = (float(hf["y_max"]) + float(hf["y_min"])) / 2
    with open(os.path.join(EXPORT_DIR, hf["elevation_rel"])) as f:
        elevation = f.read().strip()
    asset_lines.append(f'<hfield name="terrain_hfield" nrow="{int(hf["resolution"])}" ncol="{int(hf["resolution"])}" size="{hx:.6f} {hy:.6f} {hz:.6f} {float(hf.get("base_depth", 10.0)):.6f}" elevation="{elevation}"/>')
    geom_lines.append(f'<geom name="terrain_hfield" type="hfield" hfield="terrain_hfield" pos="{cx:.6f} {cy:.6f} {float(hf["z_min"]):.6f}" material="mat_terrain" friction="1.25 0.02 0.002" condim="4"/>')
    print(f"[m2-emit-v2] continuous hfield {hf['resolution']}x{hf['resolution']} (no terrain mesh convex hull).")
elif tg:
    # terrain_grid cells are already local closed prisms.  MuJoCo's local
    # convex hull is the intended collision approximation; decomposing them
    # again with coarse CoACD creates lifted facets and wheel-air gaps.
    print(f"[m2-emit-v2] terrain grid {tg['grid_n']}x{tg['grid_n']}, "
          f"{tg['n_cells']} direct local-prism collision geoms (no CoACD).")
    for c in tg["cells"]:
        name = f"terrain_{c['ix']:02d}_{c['iy']:02d}"
        abs_path = os.path.join(EXPORT_DIR, c["piece_rel"])
        asset_lines.append(f'<mesh name="{name}" file="{abs_path}"/>')
        geom_lines.append(
            f'<geom name="{name}" type="mesh" mesh="{name}" '
            'material="mat_terrain" friction="1.25 0.02 0.002" condim="4"/>')
else:
    for i, t in enumerate(groups.get("background_terrain", [])):
        mesh_rel = (t.get("collision_obj") or t.get("visual_obj")
                    or t.get("mesh_rel"))
        if not mesh_rel:
                    continue
        name = f"terrain_{i:02d}"
        abs_path = os.path.join(EXPORT_DIR, mesh_rel)
        
        # 在线分解
        pieces = get_or_create_coacd_terrain_cell(name, abs_path)
        
        if not pieces:  # 降级备用方案
            asset_lines.append(f'<mesh name="{name}" file="{abs_path}"/>')
            geom_lines.append(
                f'<geom name="{name}" type="mesh" mesh="{name}" '
                'material="mat_terrain" friction="3.0 0.05 0.02" condim="3"/>')
        else:
            for j, p_rel in enumerate(pieces):
                m_name = f"{name}_col_{j:02d}"
                p_abs_path = os.path.join(M2_DIR, p_rel)
                asset_lines.append(f'<mesh name="{m_name}" file="{p_abs_path}"/>')
                geom_lines.append(
                    f'<geom name="{m_name}" type="mesh" mesh="{m_name}" '
                    'material="mat_terrain" friction="3.0 0.05 0.02" condim="3"/>')

# 1b. indoor floor + wall as AABB-derived boxes (mesh would be coplanar).
# Use contype=2 / conaffinity=1 so background-vs-background pairs are
# disabled (Infinigen wall/floor AABBs overlap at boundaries; default
# all-vs-all collision spawns hundreds of self-intersections that blow
# up the solver). Dynamic bodies + Go2 keep MuJoCo defaults
# (contype=1, conaffinity=1) and thus collide with backgrounds normally.
def emit_thin_box(group_name, prefix, mat_name, friction):
    MIN_THICK = 0.01
    for i, e in enumerate(groups.get(group_name, [])):
        bb = e.get("world_aabb")
        if not bb:
            continue
        sx, sy, sz = (bb[3]-bb[0])/2, (bb[4]-bb[1])/2, (bb[5]-bb[2])/2
        cx, cy, cz = (bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2
        sx = max(sx, MIN_THICK); sy = max(sy, MIN_THICK); sz = max(sz, MIN_THICK)
        name = f"{prefix}_{i:02d}"
        geom_lines.append(
            f'<geom name="{name}" type="box" '
            f'pos="{pos_str([cx, cy, cz])}" '
            f'size="{pos_str([sx, sy, sz])}" '
            f'material="{mat_name}" contype="2" conaffinity="1" '
            f'friction="{friction:.3f} 0.01 0.002" condim="3"/>')


emit_thin_box("background_floor", "floor", "mat_floor", 0.7)
# NOTE: We deliberately skip background_wall geom emit. Infinigen's wall
# mesh per room is "all 4 walls of a room as one mesh"; its AABB is the
# entire room volume, which makes box-collider walls eat anything inside
# the room. Properly handling walls requires CoACD or per-face splitting
# (future work, tracked in check_module1/README.md). For demo physics
# we only need the floor; walls are visual-only (rendered by Blender).
print(f"[m2-emit-v2] {len(groups.get('background_wall', []))} walls intentionally "
      "skipped from collision (AABB-as-box would eat the room volume)")
print(f"[m2-emit-v2] static bg: {len(geom_lines)} geoms so far")


# 1c. static props (prop_concave_large): KEY -> CoACD pieces; NON-KEY -> visual.
n_static_props = 0
for p in groups.get("prop_concave_large", []):
    name = safe_name(p["name"])
    px, py, pz = p["world_pos"]
    qw, qx, qy, qz = p["world_quat"]
    if p["name"] in key_names and p["name"] in coacd_by_name:
        pieces_rel = coacd_by_name[p["name"]]["pieces_rel"]
        friction = phys_by_name.get(p["name"], {}).get(
            "vlm", {}).get("friction", 0.6)
        for j, rel in enumerate(pieces_rel):
            mname = f"{name}_col_{j:02d}"
            asset_lines.append(
                f'<mesh name="{mname}" file="{os.path.join(EXPORT_DIR, rel)}"/>')
            geom_lines.append(
                f'<geom name="{mname}" type="mesh" mesh="{mname}" '
                f'pos="{pos_str([px,py,pz])}" '
                f'quat="{quat_str([qw,qx,qy,qz])}" '
                f'material="mat_prop_static" contype="2" conaffinity="1" '
                f'friction="{friction:.3f} 0.005 0.0001" '
                f'solref="0.02 1" solimp="0.95 0.99 0.001" condim="4"/>')
        n_static_props += 1
    elif "mesh_rel" in p:
        asset_lines.append(
            f'<mesh name="{name}" file="{os.path.join(EXPORT_DIR, p["mesh_rel"])}"/>')
        geom_lines.append(
            f'<geom name="{name}" type="mesh" mesh="{name}" '
            f'pos="{pos_str([px,py,pz])}" '
            f'quat="{quat_str([qw,qx,qy,qz])}" '
            f'material="mat_prop_static" contype="2" conaffinity="1" '
            f'friction="0.6 0.005 0.0001" condim="3"/>')
        n_static_props += 1
print(f"[m2-emit-v2] {n_static_props} static props emitted")


# --- 2. dynamic objects --------------------------------------------------
# Two modes:
#   A) "source_boulder" + "size_scale": spawn a smaller scaled copy of a
#      KEY prop's CoACD pieces (outdoor "kick a small stone" pattern).
#   B) "reuse_existing": use a prop_dynamic entry's own mesh as both
#      visual and collision (indoor "make the bottle roll" pattern).
SAFETY = 0.05

# Patched robust loader by user diagnosis
raw_dyn = program.get("simulator_dynamics", [])
dyn_list = raw_dyn.get("dynamic_objects", []) if isinstance(raw_dyn, dict) else raw_dyn
for d_idx, item in enumerate(dyn_list):
    obj_key = item.get("object", item.get("name", item.get("reuse_existing", f"stone_{d_idx}")))
    obj_name = safe_name(obj_key)
    mass = float(item.get("mass", 1.0))
    friction = float(item.get("friction", 0.6))

    # Resolve initial pose / velocity.
    # Patched robust pose resolver by TPAMI-v3
    init_pose = item.get("initial_pose", {}) if isinstance(item.get("initial_pose"), dict) else item
    px, py, pz = init_pose.get("pos", item.get("pos", [0.0, 0.0, 0.5]))
    qw, qx, qy, qz = init_pose.get("quat", item.get("quat", [1.0, 0.0, 0.0, 0.0]))
    vx, vy, vz = item.get("initial_velocity", [0, 0, 0])
    wx, wy, wz = item.get("initial_angular_velocity", [0, 0, 0])

    # Schema A: source_boulder + size_scale (CoACD pieces).
    if "source_boulder" in item:
        src = item["source_boulder"]
        if src not in coacd_by_name:
            print(f"[m2-emit-v2] SKIP {obj_name}: source {src} has no CoACD")
            continue
        pieces_rel = coacd_by_name[src]["pieces_rel"]
        size_scale = float(item.get("size_scale", 1.0))
        src_phys = phys_by_name.get(src, {})
        diag = src_phys.get("geometry", {}).get("diag_inertia", [0.01]*3)
        diag = [max(d, 1e-3) for d in diag]
        src_mass = float(src_phys.get("geometry", {}).get("mass_kg", mass))
        inertia_factor = (mass / max(src_mass, 1e-6)) * (size_scale ** 2)
        diag = [d * inertia_factor for d in diag]

        piece_zs = []
        for rel in pieces_rel:
            pm = trimesh.load(os.path.join(EXPORT_DIR, rel), force="mesh")
            piece_zs.append((float(pm.bounds[0, 2]), float(pm.bounds[1, 2])))
        local_min_z = min(z[0] for z in piece_zs) * size_scale
        gz = ground_z_at(px, py)
        if gz is not None:
            snapped = gz - local_min_z + SAFETY
            if pz < snapped:
                pz = snapped

        body_inner = [
            f'  <freejoint name="{obj_name}_freejoint"/>',
            f'  <inertial pos="0 0 0" mass="{mass:.4f}" '
            f'diaginertia="{diag[0]:.4e} {diag[1]:.4e} {diag[2]:.4e}"/>',
        ]
        for j, rel in enumerate(pieces_rel):
            mname = f"{obj_name}_col_{j:02d}"
            scale_attr = (f' scale="{size_scale} {size_scale} {size_scale}"'
                          if abs(size_scale - 1.0) > 1e-6 else "")
            asset_lines.append(
                f'<mesh name="{mname}" file="{os.path.join(EXPORT_DIR, rel)}"{scale_attr}/>')
            body_inner.append(
                f'  <geom name="{mname}" type="mesh" mesh="{mname}" '
                f'material="mat_prop_dynamic" '
                f'friction="{friction:.3f} 0.005 0.0001" '
                f'solref="0.01 1" solimp="0.9 0.95 0.001" condim="3"/>')
        print(f"[m2-emit-v2] dynamic-A: {obj_name} <- {src} scale={size_scale} "
              f"mass={mass}kg  pos=({px:.2f},{py:.2f},{pz:.2f})")

    # Schema B: reuse_existing (indoor pattern).
    elif "reuse_existing" in item:
        src_name = item["reuse_existing"]
        if src_name not in dyn_props_by_name:
            print(f"[m2-emit-v2] SKIP {obj_name}: prop_dynamic {src_name} not found")
            continue
        src = dyn_props_by_name[src_name]
        mesh_rel = src["mesh_rel"]
        pm = trimesh.load(os.path.join(EXPORT_DIR, mesh_rel), force="mesh")

        # SceneSmith-style: lookup material by source_name (cheap heuristic),
        # use trimesh.volume + trimesh.moment_inertia for true mass + inertia.
        # No VLM call needed for cleanly-named objects (Bottle / Mug / Cup).
        material, density, mat_friction, mat_e = material_from_name(
            src.get("source_name", src_name))
        if not pm.is_watertight:
            try:
                pm.fill_holes()
            except Exception:
                pass
        vol = float(pm.volume) if pm.is_volume else max(
            float(pm.bounding_box.volume) * 0.5, 1e-6)
        # User-given mass overrides density-derived mass; otherwise compute.
        if "mass" in item:
            mass = float(item["mass"])
            density = mass / vol if vol > 1e-9 else density
        else:
            mass = density * vol
        # Friction also takes user override before falling back to material.
        friction = float(item.get("friction", mat_friction))

        # Real moment-of-inertia from mesh (about COM, principal-axis frame
        # is mesh-local but we feed the full 3x3 to MuJoCo via fullinertia).
        try:
            pm_density_scaled = pm.copy()
            pm_density_scaled.density = density
            I_3x3 = np.asarray(pm_density_scaled.moment_inertia)
            com = np.asarray(pm.center_mass).tolist()
        except Exception as e:
            print(f"[m2-emit-v2]  WARN inertia calc failed for {obj_name}: {e}; "
                  "falling back to AABB box approx")
            sx, sy, sz = src["world_size"]
            I_3x3 = np.diag([
                mass*(sy*sy+sz*sz)/12,
                mass*(sx*sx+sz*sz)/12,
                mass*(sx*sx+sy*sy)/12,
            ])
            com = [0.0, 0.0, 0.0]

        # MuJoCo "fullinertia" expects (Ixx Iyy Izz Ixy Ixz Iyz)
        Ixx, Iyy, Izz = float(I_3x3[0,0]), float(I_3x3[1,1]), float(I_3x3[2,2])
        Ixy, Ixz, Iyz = float(I_3x3[0,1]), float(I_3x3[0,2]), float(I_3x3[1,2])

        # Ground snap: same as before.
        local_min_z = float(pm.bounds[0, 2])
        local_max_z = float(pm.bounds[1, 2])
        gz = ground_z_at(px, py)
        if gz is not None:
            snapped = gz - local_min_z + SAFETY
            if pz < snapped:
                print(f"[m2-emit-v2]  snap-B {obj_name}: pz {pz:.3f} -> {snapped:.3f} "
                      f"(ground={gz:.3f}, local_z=[{local_min_z:.3f},{local_max_z:.3f}])")
                pz = snapped
        else:
            print(f"[m2-emit-v2]  WARN: no ground at ({px:.2f},{py:.2f})")

        mname = f"{obj_name}_visual"
        asset_lines.append(
            f'<mesh name="{mname}" file="{os.path.join(EXPORT_DIR, mesh_rel)}"/>')
        body_inner = [
            f'  <freejoint name="{obj_name}_freejoint"/>',
            f'  <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" '
            f'mass="{mass:.4f}" '
            f'fullinertia="{Ixx:.4e} {Iyy:.4e} {Izz:.4e} '
            f'{Ixy:.4e} {Ixz:.4e} {Iyz:.4e}"/>',
            f'  <geom name="{mname}" type="mesh" mesh="{mname}" '
            f'material="mat_prop_dynamic" '
            f'friction="{friction:.3f} 0.005 0.0001" '
            f'solref="0.01 1" solimp="0.9 0.95 0.001" condim="3"/>',
        ]
        print(f"[m2-emit-v2] dynamic-B: {obj_name} <- reuse {src_name}  "
              f"material={material} rho={density:.0f} mass={mass:.3f}kg "
              f"vol={vol*1e3:.2f}L  pos=({px:.2f},{py:.2f},{pz:.2f})  "
              f"com_local=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})")
    else:
        print(f"[m2-emit-v2] SKIP {obj_name}: needs either "
              "'source_boulder' or 'reuse_existing'")
        continue

    body_lines.append(
        f'<body name="{obj_name}" pos="{pos_str([px, py, pz])}" '
        f'quat="{quat_str([qw, qx, qy, qz])}">')
    body_lines.extend(body_inner)
    body_lines.append('</body>')
    keyframe_qpos.extend([px, py, pz, qw, qx, qy, qz])
    keyframe_qvel.extend([vx, vy, vz, wx, wy, wz])


# --- 3. Go2 spawn position ----------------------------------------------
# Outdoor: place at (0,0) and let terrain ray-cast lift base.
# Indoor: place near the dynamic target so the kick is meaningful.
GO2_HOME_LEG = [0.0, 0.9, -1.8]
GO2_HIP_HEIGHT = 0.27
GO2_X = 0.0
GO2_Y = 0.0

# Go2 anchor strategy depends on robot_action.
#   kick : place Go2 so that its FL foot's reach lands ON the dynamic
#          target. Empirically (KICK_LEG=[0,0.3,-1.8] in module2_rollout_v2),
#          FL collision foot at strike-end sits at base + (~0.15, ~0.04, ~-0.20).
#          So anchor = target - (0.15, 0.04, 0).
#   walk : place 1.3m -X of target so it has runway before passing by.
#   watch / default : 0.6m -X of target so it can observe.
# Patched robust robot_action and target pose resolver by TPAMI-v3
raw_dyn = program.get("simulator_dynamics", [])
dyn_list = raw_dyn.get("dynamic_objects", []) if isinstance(raw_dyn, dict) else raw_dyn
robot_action = program.get("robot_action", raw_dyn.get("robot_action", "watch") if isinstance(raw_dyn, dict) else "watch")
is_jackal = "jackal" in str(program.get("robot_model", "")).lower() or "jackal" in str(program.get("robot", {}).get("model", "")).lower()

go2_yaw = 0.0
if dyn_list:
    target = dyn_list[0]
    init_pose = target.get("initial_pose", {}) if isinstance(target.get("initial_pose"), dict) else target
    t_pos = init_pose.get("pos", target.get("pos", [0.0, 0.0, 0.5]))
    tx = float(t_pos[0])
    ty = float(t_pos[1])
    if robot_action == "kick":
        # Iterative tuning: too-close (tx-0.20) hammered the bottle with FL
        # at home pose; too-far (tx-0.40) FL strike couldn't reach.
        # Picking the midpoint, with bottle Y nudged further to Go2's left
        # (base+0.20 instead of base+0.16) for clearer side-of-body contact.
        GO2_X = tx - 0.30
        GO2_Y = ty - 0.20
        print(f"[m2-emit-v2] Go2 anchored for FL kick reach: "
              f"({GO2_X:.2f}, {GO2_Y:.2f}) -> target ({tx:.2f}, {ty:.2f})")
    elif robot_action == "walk":
        GO2_X = tx - 1.30
        GO2_Y = ty - 0.10
        print(f"[m2-emit-v2] Go2 anchored for walk pass-by: "
              f"({GO2_X:.2f}, {GO2_Y:.2f}) -> target ({tx:.2f}, {ty:.2f})")
    else:
        GO2_X = tx - 0.6
        GO2_Y = ty
        print(f"[m2-emit-v2] Go2 anchored 0.6m -X of dynamic target: "
              f"({GO2_X:.2f}, {GO2_Y:.2f})")

go2_gz = ground_z_at(GO2_X, GO2_Y)
GO2_BASE_Z = (go2_gz if go2_gz is not None else 0.0) + GO2_HIP_HEIGHT
print(f"[m2-emit-v2] Go2 base_z = ground({GO2_X:.2f},{GO2_Y:.2f})="
      f"{go2_gz} + {GO2_HIP_HEIGHT} = {GO2_BASE_Z:.3f}")
CAR_BASE_Z = 3.6 if is_jackal else (go2_gz if go2_gz is not None else 0.0) + 0.04  # Jackal falls safely onto procedurally generated terrain
car_qpos_init = [GO2_X, GO2_Y, CAR_BASE_Z, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.0]  # 7位基座 + 4位轮子
car_qvel_init = [0.0] * 10  # 6位基座速度 + 4位轮子
keyframe_qpos.extend(car_qpos_init)
keyframe_qvel.extend(car_qvel_init)

# --- 4. Localize Robot (copy assets + rewrite mesh paths to absolute) ----
# 1. 动态获取源目录下的第一个 XML 包含文件名（例如 "car.xml" 或 "go2.xml"）
xml_files = [f for f in os.listdir(GO2_DIR) if f.endswith(".xml")]
if not xml_files:
    raise FileNotFoundError(f"在机器人源目录中未找到任何 XML 文件: {GO2_DIR}")
robot_xml_name = xml_files[0]  # 例如: "car.xml"
robot_label = os.path.splitext(robot_xml_name)[0]  # 例如: "car"
robot_local_folder_name = f"{robot_label}_local"  # 例如: "car_local"

# 2. 动态创建目标文件夹路径
go2_local = os.path.join(M2_DIR, robot_local_folder_name)

# 核心安全保护：只有当源路径和目标路径不一致时才执行清理，防止删掉自己的源文件
if os.path.abspath(GO2_DIR) == os.path.abspath(go2_local):
    print(f"[m2-emit-v2] 警告: 源路径与目标本地化路径相同，跳过清理和复制，直接进行就地处理")
else:
    if os.path.exists(go2_local):
        shutil.rmtree(go2_local)
    shutil.copytree(GO2_DIR, go2_local)

# 3. 动态加载对应的机器人 XML
go2_xml_local = os.path.join(go2_local, robot_xml_name)
with open(go2_xml_local) as f:
    go2_src = f.read()

# 4. Resolve mesh paths against the original robot XML before moving it.
# Bundled benchmark robot XMLs use relative paths with directory components;
# leaving those unchanged after copytree points them at the wrong directory.
go2_src = re.sub(r'\s+meshdir="assets"', "", go2_src)


def _abs_mesh(m):
    head, fname = m.group(1), m.group(2)
    if os.path.isabs(fname):
        resolved = fname
    elif "/" in fname or "\\" in fname:
        resolved = os.path.abspath(os.path.join(GO2_DIR, fname))
    else:
        resolved = os.path.join(go2_local, "assets", fname)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"robot mesh missing: {resolved}")
    return f'{head}file="{resolved}"'


go2_src = re.sub(r'(<mesh[^/>]*?)file="([^"]+)"', _abs_mesh, go2_src)
with open(go2_xml_local, "w") as f:
    f.write(go2_src)
print(f"[m2-emit-v2] localized Robot -> {go2_xml_local}")

# --- 5. Assemble XML ---------------------------------------------------
asset_block = "\n    ".join(asset_lines)
material_block = "\n    ".join(material_lines)
geom_block = "\n    ".join(geom_lines)
body_block = "\n    ".join(body_lines) if body_lines else ""
qpos_str = " ".join(f"{x:.6f}" for x in keyframe_qpos)
qvel_str = " ".join(f"{x:.6f}" for x in keyframe_qvel)

# 动态生成包含相对路径的 XML
xml = f"""<mujoco model="scene_m2">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.7 0.65 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.85 0.78 0.62 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.78 0.62" rgb2="0.55 0.65 0.85" width="512" height="3072"/>
    {material_block}
    {asset_block}
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 30" dir="-0.3 -0.2 -1" directional="true"
           diffuse="0.9 0.85 0.7" specular="0.2 0.2 0.2"/>
    {geom_block}
    {body_block}
  </worldbody>
  <include file="{robot_local_folder_name}/{robot_xml_name}"/>
  <keyframe>
    <key name="m2_init" qpos="{qpos_str}" qvel="{qvel_str}"/>
  </keyframe>
</mujoco>
"""

out_path = os.path.join(M2_DIR, "scene_m2.xml")
with open(out_path, "w") as f:
    f.write(xml)
print(f"[m2-emit-v2] wrote {out_path}")
print(f"[m2-emit-v2]   {len(asset_lines)} mesh assets, "
      f"{len(geom_lines)} static geoms, {len(body_lines)} body lines")
