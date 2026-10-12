"""Module 2 Blender re-render (generic / scene-agnostic).

Reads rollout_log.json + manifest.json + scene.blend, emits Cycles
frames matching the MuJoCo rollout. Supports both:

  schema A (outdoor "kick_stone"): a NEW dynamic body uses a KEY
    boulder's mesh as visual template, scaled by size_scale. Go2 +
    stone are anchored in front of the scene camera (offset onto
    OpaqueTerrain via ray-cast). Use this for scene_1 outdoor.

  schema B (indoor "reuse_existing"): the dynamic body IS a real
    object already in the scene.blend (Bottle / Mug / etc.) at the
    same world coordinates as the MuJoCo simulation. No anchor
    offset needed -- rollout_log world coords map 1:1 to scene.blend
    world coords. The original Bottle object is hidden, replaced by
    an animated copy driven by stone_pos / stone_quat.

Usage:
  blender SCENE_BLEND --background --python module2_render_blender_v2.py -- \\
      ROLLOUT_LOG_JSON GO2_DIR EXPORT_DIR OUTPUT_DIR [N_FRAMES]
"""
import bpy
import numpy as np

# Disable displacement and drain opaque muddy water planes (liquid_fine)
for obj in bpy.data.objects:
    # A. 强行抽干不透明的泥红色湖水面 (liquid_fine / water)
    if "liquid" in obj.name.lower() or "water" in obj.name.lower():
        obj.hide_render = True
        obj.hide_viewport = True
        print("Successfully drained water plane: " + obj.name)
    # Keep terrain modifiers: Module 1 exports the evaluated scene mesh, so
    # removing Blender's terrain detail can create a visual physics/ground gap.





import json
import os
import sys
from mathutils import Vector, Quaternion, Matrix


argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
if len(args) < 4:
    print("usage: render_v2 ROLLOUT_LOG GO2_DIR EXPORT_DIR OUT_DIR [N_FRAMES]")
    sys.exit(2)

LOG_JSON, GO2_DIR, EXPORT_DIR, OUTPUT_DIR = args[:4]
N_FRAMES = int(args[4]) if len(args) > 4 else -1
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(LOG_JSON) as f:
    log = json.load(f)
with open(os.path.join(EXPORT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
# Validate that LOG_JSON is the postproc'd version
if not LOG_JSON.endswith("_blender.json"):
    print(f"ERROR: LOG_JSON must be 'rollout_log_blender.json' (postproc'd version)")
    print(f"  Got: {LOG_JSON}")
    print(f"  Expected: {LOG_JSON.replace('.json', '_blender.json')}")
    print(f"\nThe rollout_log.json from MuJoCo contains mesh_pos/mesh_quat transforms")
    print(f"that will cause Go2 to render incorrectly in Blender.")
    print(f"\nIf you ran module2_rollout_v2.py, it should have auto-generated")
    print(f"rollout_log_blender.json. If not, run postproc_rollout_blender.py manually.")
    sys.exit(1)

if not os.path.exists(LOG_JSON):
    print(f"ERROR: {LOG_JSON} not found")
    print(f"\nDid you run module2_rollout_v2.py? It should auto-generate this file.")
    print(f"If not, run: python postproc_rollout_blender.py <export_dir>")
    sys.exit(1)

print(f"[render-v2] Using postproc'd log: {LOG_JSON}")
if not LOG_JSON.endswith("_blender.json"):
    print(f"ERROR: LOG_JSON must be 'rollout_log_blender.json' (postproc'd version)")
    print(f"  Got: {LOG_JSON}")
    print(f"  Expected: {LOG_JSON.replace('.json', '_blender.json')}")
    print(f"\nThe rollout_log.json from MuJoCo contains mesh_pos/mesh_quat transforms")
    print(f"that will cause Go2 to render incorrectly in Blender.")
    print(f"\nIf you ran module2_rollout_v2.py, it should have auto-generated")
    print(f"rollout_log_blender.json. If not, run postproc_rollout_blender.py manually.")
    sys.exit(1)

if not os.path.exists(LOG_JSON):
    print(f"ERROR: {LOG_JSON} not found")
    print(f"\nDid you run module2_rollout_v2.py? It should auto-generate this file.")
    print(f"If not, run: python postproc_rollout_blender.py <export_dir>")
    sys.exit(1)

print(f"[render-v2] Using postproc'd log: {LOG_JSON}")



# Detect schema. Schema-A/B are legacy Go2 + dynamic prop paths. Industrial
# robot logs expose <robot>_visual_geoms / <robot>_geom_xforms and may not have
# any stone_source_boulder.
robot_vis_keys = [k for k in log.keys() if k.endswith("_visual_geoms")]
robot_name = None
robot_xform_key = None
for key in robot_vis_keys:
    cand = key[:-len("_visual_geoms")]
    if log.get("frames") and f"{cand}_geom_xforms" in log["frames"][0]:
        robot_name = cand
        robot_xform_key = f"{cand}_geom_xforms"
        break
industrial_robot = robot_name is not None and robot_name != "go2" and "stone_source_boulder" not in log
src_name = log.get("stone_source_boulder")
groups = manifest.get("groups", {})
dyn_props = {p["name"]: p for p in groups.get("prop_dynamic", [])}
boulders = manifest.get("boulders", []) or groups.get("prop_concave_large", [])
boulder_meta = next((b for b in boulders if b["name"] == src_name), None) if src_name else None
schema_b = (src_name in dyn_props) if src_name else False
schema_a = boulder_meta is not None and not schema_b
print(f"[render-v2] source='{src_name}' schema_a={schema_a} schema_b={schema_b} industrial_robot={industrial_robot}")

mj_frames = log["frames"]
if 0 < N_FRAMES < len(mj_frames):
    n_total = len(mj_frames)
    step = max(1, (n_total - 1) // max(1, N_FRAMES - 1))
    mj_frames = [log["frames"][min(i * step, n_total - 1)]
                 for i in range(N_FRAMES)]
print(f"[render-v2] {len(mj_frames)} frames @ {log['fps']} fps")
VISUAL_GEOMS = log[f"{robot_name}_visual_geoms"] if industrial_robot else log["go2_visual_geoms"]
print(f"[render-v2] {len(VISUAL_GEOMS)} robot visual meshes per frame")


# Disable sequencer (c2w blends ship with a SequenceEditor whose post
# strips wash the render with a strong red tint).
bpy.context.scene.render.use_sequencer = False
print("[render-v2] disabled render.use_sequencer")


# ---------------------------------------------------------------------------
# Scene-wide: unhide leaf collections (Infinigen ships them hidden by default)
# and pick camera.
# ---------------------------------------------------------------------------
n_unhid = 0
for c in bpy.data.collections:
    if c.name.startswith(("assets", "particleassets", "unique_assets")):
        if c.hide_render or c.hide_viewport:
            c.hide_render = False; c.hide_viewport = False; n_unhid += 1
        for o in c.objects:
            if o.hide_render or o.hide_viewport:
                o.hide_render = False; o.hide_viewport = False; n_unhid += 1
print(f"[render-v2] unhid {n_unhid} collections/objects")


# ---------------------------------------------------------------------------
# Indoor scenes ship with a Sky Texture (Nishita) connected to the World
# Output -- if the sun angle is low, the sky tints the whole interior red/
# orange like a sunset. We replace it with a neutral white background for
# clean indoor renders. Outdoor scenes already use OpaqueTerrain + skybox
# atlases so they're unaffected.
world = bpy.context.scene.world
if world and world.use_nodes:
    nt = world.node_tree
    out_node = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None)
    if out_node and out_node.inputs["Surface"].is_linked:
        src = out_node.inputs["Surface"].links[0].from_node
        # Walk the tree to see if any Sky Texture or strongly-colored node
        # is feeding the surface.
        def has_sky(node, depth=0):
            if depth > 5: return False
            if node.bl_idname == "ShaderNodeTexSky": return True
            for inp in node.inputs:
                if inp.is_linked and has_sky(inp.links[0].from_node, depth+1):
                    return True
            return False
        if has_sky(src):
            # Replace surface with a neutral white Background (str=1.0).
            for ln in list(out_node.inputs["Surface"].links):
                nt.links.remove(ln)
            bg = nt.nodes.new("ShaderNodeBackground")
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 1.0
            nt.links.new(bg.outputs["Background"], out_node.inputs["Surface"])
            print("[render-v2] replaced sky-fed world bg with neutral white")


# ---------------------------------------------------------------------------
# Force all lights to neutral white. Blender 3.6 loading a 3.6+-authored
# blend can lose newer light attributes and default to red. Also replace
# any Blackbody emission node with white -- those tend to be sub-3000K
# tungsten which over-tints indoor renders.
n_white = 0
for o in bpy.data.objects:
    if o.type != "LIGHT":
        continue
    o.data.color = (1.0, 1.0, 1.0)
    n_white += 1
    if o.data.use_nodes:
        for nd in list(o.data.node_tree.nodes):
            if nd.bl_idname == "ShaderNodeBlackbody":
                rgb = o.data.node_tree.nodes.new("ShaderNodeRGB")
                rgb.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
                targets = []
                for ln in list(o.data.node_tree.links):
                    if ln.from_node == nd:
                        targets.append(ln.to_socket)
                        o.data.node_tree.links.remove(ln)
                for ts in targets:
                    o.data.node_tree.links.new(rgb.outputs[0], ts)
                o.data.node_tree.nodes.remove(nd)
print(f"[render-v2] forced {n_white} lights to neutral white")

cam_obj = bpy.data.objects.get("camera_0_0") or bpy.data.objects.get("Camera")
if cam_obj is None or cam_obj.type != "CAMERA":
    print("[render-v2] FATAL: no camera"); sys.exit(3)
bpy.context.scene.camera = cam_obj

# Optional: replace c2w camera with a MuJoCo-style auto-frame camera
# (matches sim/module2_rollout_v2.py params -- lookat=mid(go2,stone),
# azimuth=110, elevation=-18, distance=max(2.5, ||stone-go2||*2.5)).
# Trigger by env var BL_CAM=mj.
import math
# Set for a MuJoCo-style moving tracking camera.  The same azimuth,
# elevation and distance are evaluated against every logged Jackal pose.
mj_follow_camera = False
if os.environ.get("BL_CAM", "").lower() == "mj":
    if robot_name == "jackal":
        mid = Vector(mj_frames[0]["jackal_base_pos"])
        span = 2.0
    elif industrial_robot:
        mid = Vector(mj_frames[0].get("ee_pos", [0.4, -0.4, 1.0]))
        span = 1.2
    else:
        go2_p = Vector(mj_frames[0]["go2_base_pos"])
        stone_name = log.get("stone_object")
        
        # 判断是否有真实的动态石子
        if stone_name and stone_name != "None":
            stone_p = Vector(mj_frames[0]["stone_pos"])
            mid = (go2_p + stone_p) * 0.5
            span = (stone_p - go2_p).length
        else:
            # 无石子情况下，直接聚焦小车
            mid = go2_p
            span = 1.8
    # Default MuJoCo params; can be overridden via BL_CAM_AZ/EL/D for indoor
    # rooms where MuJoCo's free cam pose lands inside a wall.
    az_deg = float(os.environ.get("BL_CAM_AZ", "110.0"))
    el_deg = float(os.environ.get("BL_CAM_EL", "-18.0"))
    d = float(os.environ.get("BL_CAM_D", "0"))
    if d <= 0:
        d = max(2.5, span * 2.5)
    az = math.radians(az_deg); el = math.radians(el_deg)
    # Keep a dedicated scalar: scene-population code below also uses `d`.
    mj_cam_distance = float(d)
    fwd_v = Vector((math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)))
    cam_pos = mid - fwd_v * mj_cam_distance
    # Reuse the blend's stable camera datablock.  Creating a new camera here
    # caused Blender's timeline evaluation to explode its transform; clearing
    # the original rig/animation gives a deterministic tracked camera instead.
    new_cam = cam_obj
    new_cam.parent = None
    new_cam.matrix_parent_inverse = Matrix.Identity(4)
    new_cam.constraints.clear()
    new_cam.animation_data_clear()
    new_cam.delta_location = (0.0, 0.0, 0.0)
    new_cam.delta_rotation_euler = (0.0, 0.0, 0.0)
    new_cam.delta_scale = (1.0, 1.0, 1.0)
    new_cam.location = cam_pos
    look_dir = (mid - cam_pos).normalized()
    new_cam.rotation_mode = "QUATERNION"
    new_cam.rotation_quaternion = look_dir.to_track_quat("-Z", "Y")
    bpy.context.scene.timeline_markers.clear()
    bpy.context.scene.camera = new_cam

    cam_obj = new_cam
    mj_follow_camera = robot_name == "jackal"
    print(f"[render-v2] BL_CAM=mj: lookat={mid} distance={mj_cam_distance:.2f} cam_pos={cam_pos}; tracking={mj_follow_camera}")

cam_world = cam_obj.matrix_world.translation.copy()
fwd = cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
fwd.normalize()
print(f"[render-v2] camera at {cam_world}, fwd {fwd}")


# ---------------------------------------------------------------------------
# Anchor offset: how to map MuJoCo world coords to scene.blend world coords.
# Schema A: ray-cast 6 m ahead of camera onto OpaqueTerrain; offset
#           translates rollout's Go2_base_xy_initial to that anchor.
# Schema B: rollout_log world coords are ALREADY in scene.blend frame
#           (export pipeline preserved world poses). No offset.
# ---------------------------------------------------------------------------
# ===========================================================================
# 严谨的全局干地硬阈值检索与侧向剖面追踪相机算法 (Schema A)
# ===========================================================================
go2_base0 = Vector(mj_frames[0].get("go2_base_pos", [0.0, 0.0, 0.0]))
mj_to_world_offset = Vector((0.0, 0.0, 0.0))

if schema_a:
    deps = bpy.context.evaluated_depsgraph_get()
    best_anchor = None
    best_score = -999999.0
    
    print("\n[render-v2] 正在进行林中空地探测，避开脚下茂密的草丛与叶片...")
    
    # 1. 搜集场景中 100% 的植物、树木、草丛阻挡点
    blockers = []
    # 从 manifest 中提取 procedurally 生成的植被坐标
    for item in groups.get("vegetation", []) + groups.get("decoration", []):
        if "world_pos" in item:
            blockers.append(Vector(item["world_pos"]))
            
    # 从 Blender 占位符集合中提取位置
    for coll_name in ["placeholders:TreeFactory(3040046)", "placeholders:BushFactory(543568399)"]:
        coll = bpy.data.collections.get(coll_name)
        if coll:
            for obj in coll.objects:
                blockers.append(obj.matrix_world.translation.copy())
                
    print(f"[render-v2] 已录入 {len(blockers)} 个植被阻挡物坐标。")

    # 2. 划定相机前方的局部林地扫描区：以相机默认投影点 (-72.0, -62.8) 为中心
    center_x, center_y = -72.0391, -62.8029
    
    # 细密网格扫描（步长 0.25 米）
    for tx in [center_x + i * 0.25 for i in range(-12, 13)]:
        for ty in [center_y + j * 0.25 for j in range(-12, 13)]:
            drop_origin = Vector((tx, ty, 50.0))
            drop_dir = Vector((0.0, 0.0, -1.0))
            
            # 向下投射线寻找 OpaqueTerrain 地表
            ray_origin = drop_origin.copy()
            hit_terrain = False
            terrain_loc = None
            terrain_normal = None
            
            for _ in range(5):
                hit, loc, normal, idx, obj, _ = bpy.context.scene.ray_cast(
                    deps, ray_origin, drop_dir, distance=100.0)
                if not hit:
                    break
                if "OpaqueTerrain" in obj.name:
                    hit_terrain = True
                    terrain_loc = loc
                    terrain_normal = normal
                    break
                ray_origin = loc + drop_dir * 0.05
                
            if hit_terrain:
                # 检查地形坡度是否平坦 (适合机器狗站立)
                if terrain_normal.z > 0.94:
                    
                    # 计算该点到最近的任何植物中心的物理距离
                    min_veg_dist = 999.0
                    if blockers:
                        for b_pos in blockers:
                            dist_to_blocker = (terrain_loc - b_pos).length
                            if dist_to_blocker < min_veg_dist:
                                min_veg_dist = dist_to_blocker
                    
                    # 距离默认中心点的距离（保持构图在相机附近，周围景物不变）
                    dist_from_center = ((tx - center_x)**2 + (ty - center_y)**2)**0.5
                    
                    # 评分机制：
                    # - 距离最近的植物越远越好 (min_veg_dist * 5.0)，保证脚底是一片干净的泥土空地
                    # - 离默认构图中心越近越好 (- dist_from_center * 1.5)，确保依然处于大树、巨石之间
                    score = min_veg_dist * 5.0 - dist_from_center * 1.5
                    
                    if score > best_score:
                        best_score = score
                        best_anchor = terrain_loc.copy()
                        best_veg_dist = min_veg_dist
                        best_tx = tx
                        best_ty = ty

    if best_anchor:
        anchor = best_anchor.copy()
        anchor.z += 0.03  # 微调高度
        print(f"[render-v2] 成功定位林中最佳空地（Bald Spot）！")
        print(f"            世界坐标 (Anchor): {anchor}")
        print(f"            该处海拔 (Height Z): {anchor.z:.2f}m")
        print(f"            脚下安全无草半径: {best_veg_dist:.2f} 米 (已完美避开树叶草丛夹击)\n")
    else:
        # 降级备用
        forward_pt = cam_world + fwd * 6.0
        anchor = Vector((forward_pt.x, forward_pt.y, cam_world.z - 2.0))
        print("[render-v2] WARNING: 空地探测失败，退回到默认投影。")
        
    mj_to_world_offset = anchor - Vector((go2_base0.x, go2_base0.y, 0.0))
    print(f"[render-v2] 最终确定的 Schema-A 偏移量: {mj_to_world_offset}")

    # ===========================================================================
    # 黄金级：侧前方斜俯视写实追踪相机
    # ===========================================================================
    print("[render-v2] 正在林中空地前斜向架设全新追踪相机...")
    
    # 机器狗朝 +X 运动，石块在 +X。
    traj_mid_raw = Vector(mj_frames[len(mj_frames)//2]["go2_base_pos"])
    traj_mid_world = traj_mid_raw + mj_to_world_offset
    
    cam_data = bpy.data.cameras.new("cinematic_side_cam_data")
    cam_data.lens = 48  # 电影写实标准镜头
    side_cam = bpy.data.objects.new("cinematic_side_cam", cam_data)
    bpy.context.scene.collection.objects.link(side_cam)
    
    # 相机机位拉近：面向 Y 轴正方向望去，X轴向前倾斜 0.3 米，Y轴后退 2.4 米，高度抬高 1.25 米。
    # 这个倾斜俯仰角度刚好可以俯视泥土空地，让石子和狗的着地脚非常瞩目，同时将左侧树干、右侧巨石作为绝佳的侧边画幅框景（Framing）。
    cam_pos = traj_mid_world + Vector((0.3, -2.4, 1.25))
    side_cam.location = cam_pos
    
    side_cam.rotation_mode = 'QUATERNION'
    look_dir = (traj_mid_world - cam_pos).normalized()
    side_cam.rotation_quaternion = look_dir.to_track_quat('-Z', 'Y')
    
    # 激活新相机
    bpy.context.scene.camera = side_cam
    print(f"[render-v2] 侧向追踪相机激活成功：位置 {cam_pos} -> 注视点 {traj_mid_world}\n")

else:
    print("[render-v2] schema-B: 1:1 world frame, no offset")


# ---------------------------------------------------------------------------
# Foliage repopulation (only if outdoor TreeFactory/BushFactory placeholders
# exist -- indoor scenes skip this entirely).
# ---------------------------------------------------------------------------
def populate(placeholder_coll_name, asset_coll_name, min_tris, max_tris, max_dist):
    ph = bpy.data.collections.get(placeholder_coll_name)
    asset = bpy.data.collections.get(asset_coll_name)
    if ph is None or asset is None:
        return 0
    asset.hide_viewport = False; asset.hide_render = False
    cands = [o for o in asset.objects
             if o.type == "MESH" and o.data
             and min_tris <= len(o.data.polygons) <= max_tris]
    if not cands:
        cands = [o for o in asset.objects
                 if o.type == "MESH" and o.data and len(o.data.polygons) >= 500]
        if not cands:
            return 0
    cands.sort(key=lambda o: len(o.data.polygons))
    template = cands[0]; template.hide_render = False
    pop = bpy.data.collections.new(
        f"populated_{placeholder_coll_name.split(':')[1]}")
    bpy.context.scene.collection.children.link(pop)
    n = 0
    for src in list(ph.objects):
        if src.type != "MESH":
            continue
        
        # ⚠️ 修正 1: 无条件隐藏所有的原始占位符，防止它们在渲染中露脸阻挡视线
        src.hide_render = True
        src.hide_viewport = True
        
        d = src.matrix_world.translation - cam_world
        # ⚠️ 修正 2: 适当增加安全距离（3.5米），防止生成的高精度树木枝叶穿入相机镜头
        if d.dot(fwd) < 3.5 or d.length > max_dist:
            continue
            
        new_obj = template.copy()
        new_obj.data = template.data
        new_obj.modifiers.clear()
        new_obj.matrix_world = src.matrix_world.copy()
        new_obj.scale = (1.0, 1.0, 1.0)
        new_obj.name = f"pop_{placeholder_coll_name.split(':')[1]}_{n}"
        new_obj.hide_render = False
        new_obj.hide_viewport = False
        pop.objects.link(new_obj)
        n += 1
    return n


n_trees = populate("placeholders:TreeFactory(3040046)",
                   "unique_assets:TreeFactory(3040046)",
                   50_000, 200_000, 60.0)
n_bushes = populate("placeholders:BushFactory(543568399)",
                    "unique_assets:BushFactory(543568399)",
                    500, 200_000, 30.0)
print(f"[render-v2] populate: {n_trees} trees, {n_bushes} bushes")

# ======= 在导入循环前加入递归克隆函数，确保子部件跟随机械运动 [1.1.2] =======
def clone_obj_recursive(src_obj, dest_coll):
    """
    递归克隆父物体及其所有子零部件，保留相对位置、材质和局部坐标矩阵。 [1.1.2]
    """
    new_obj = src_obj.copy()
    if src_obj.data:
        new_obj.data = src_obj.data.copy()
    dest_coll.objects.link(new_obj)
    
    # 落地实例取消隐藏
    new_obj.hide_render = False
    new_obj.hide_viewport = False
    
    for child in src_obj.children:
        new_child = clone_obj_recursive(child, dest_coll)
        new_child.parent = new_obj
        # 核心：保留子零部件相对于父物体的局部矩阵（Local Matrix），实现绝对对齐！ [1.1.2]
        new_child.matrix_local = child.matrix_local.copy()
        
    return new_obj
# ---------------------------------------------------------------------------
# Robot: import each visual geom as a Blender object, share mesh data via cache.
# Go2 uses the CLI robot dir; industrial arms use module2/robot_local/assets.
# ---------------------------------------------------------------------------
robot_coll = bpy.data.collections.new(robot_name or "unitree_go2")
bpy.context.scene.collection.children.link(robot_coll)
robot_assets_dir = (os.path.join(EXPORT_DIR, "module2", "robot_local", "assets")
                    if industrial_robot and robot_name != "jackal" else os.path.join(GO2_DIR, "assets"))
mesh_cache = {}
geom_objs = []
# ======= 替换为：智能自适应 3D 建模渲染版本 =======
# ======= 替换为：超写实 3D 智能重构版本 (自动倒角底盘 + 传动轴 + 悬挂支撑 + 双色轮毂) =======
for i, g in enumerate(VISUAL_GEOMS):
    mf = g["mesh_file"]
    path = os.path.join(robot_assets_dir, mf)
    # Primitive visual geoms have no asset filename; cache them independently.
    cache_key = mf or f"{g.get('primitive', 'generated')}_{i}"
    
    if cache_key not in mesh_cache:
        # 如果磁盘上没有 .obj 文件（原生的 box 和 cylinder），自动在 Blender 中重构出高精度模型！ [7.1.3]
        if g.get("primitive") == "box":
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            imp = bpy.context.active_object
            imp.scale = tuple(2.0 * float(v) for v in g["size"])
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            imp.hide_render = True; imp.hide_viewport = True
            mesh_cache[cache_key] = imp
        elif g.get("primitive") == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=float(g["size"][0]), depth=2.0 * float(g["size"][1]))
            imp = bpy.context.active_object
            imp.hide_render = True; imp.hide_viewport = True
            mesh_cache[cache_key] = imp
        elif not os.path.exists(path):
            print(f"[render-v2] 未找到 3D 物理文件 {mf}，正在进行「超写实越野车」三维重建...")
            
            if "wheel" in g["body_name"] or "cylinder" in mf:
                # ==== 1. 重构高写实双色越野轮胎 (含深色橡胶外胎 + 银灰色金属轮毂) ====
                # 外胎
                bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.04)
                tire = bpy.context.active_object
                tire.name = "tire_outer"
                
                # 内侧轮毂
                bpy.ops.mesh.primitive_cylinder_add(radius=0.07, depth=0.042)
                rim = bpy.context.active_object
                rim.name = "rim_inner"
                
                # 轮轴中心盖
                bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=0.046)
                hub = bpy.context.active_object
                hub.name = "hub_center"
                
                # 创建轮胎材质 (深黑色磨砂橡胶)
                mat_tire = bpy.data.materials.new(name="mat_tire_rubber")
                mat_tire.use_nodes = True
                bsdf_t = mat_tire.node_tree.nodes.get("Principled BSDF")
                if bsdf_t:
                    bsdf_t.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
                    bsdf_t.inputs["Roughness"].default_value = 0.92
                tire.data.materials.append(mat_tire)
                
                # 创建轮毂材质 (金属亮银色)
                mat_rim = bpy.data.materials.new(name="mat_tire_rim")
                mat_rim.use_nodes = True
                bsdf_r = mat_rim.node_tree.nodes.get("Principled BSDF")
                if bsdf_r:
                    bsdf_r.inputs["Base Color"].default_value = (0.65, 0.65, 0.65, 1.0)
                    bsdf_r.inputs["Metallic"].default_value = 0.90
                    bsdf_r.inputs["Roughness"].default_value = 0.18
                rim.data.materials.append(mat_rim)
                hub.data.materials.append(mat_rim)
                
                # 合并轮胎的所有子部件
                bpy.ops.object.select_all(action='DESELECT')
                tire.select_set(True)
                rim.select_set(True)
                hub.select_set(True)
                bpy.context.view_layer.objects.active = tire
                bpy.ops.object.join()
                
                imp = tire
                
            else:
                bpy.ops.mesh.primitive_cube_add(size=1.0)
                chassis = bpy.context.active_object
                chassis.name = "chassis_main"
                chassis.scale = (0.3, 0.2, 0.08)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                
                # 给底盘边缘加倒角 (Bevel) (16个空格缩进)
                bev = chassis.modifiers.new(name="Bevel", type="BEVEL")
                bev.width = 0.012
                bev.segments = 3
                
                # 创建前传动轴 (Cylinder) (16个空格缩进)
                bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=0.48, rotation=(1.5708, 0, 0), location=(0.2, 0, -0.01))
                axle_f = bpy.context.active_object
                axle_f.name = "axle_front"
                
                # 创建后传动轴 (16个空格缩进)
                bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=0.48, rotation=(1.5708, 0, 0), location=(-0.2, 0, -0.01))
                axle_b = bpy.context.active_object
                axle_b.name = "axle_back"
                
                # 创建 4 个悬挂固定承重块 (16个空格缩进)
                sh_parts = []
                for sx, sy in [(0.2, 0.16), (0.2, -0.16), (-0.2, 0.16), (-0.2, -0.16)]:
                    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, sy, -0.04))
                    sp = bpy.context.active_object
                    sp.scale = (0.05, 0.04, 0.06)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                    sh_parts.append(sp)
                    
                # 创建底盘涂装材质 (工程越野亮黄色漆) (16个空格缩进)
                mat_chassis = bpy.data.materials.new(name="mat_chassis_yellow")
                mat_chassis.use_nodes = True
                bsdf_c = mat_chassis.node_tree.nodes.get("Principled BSDF")
                if bsdf_c:
                    bsdf_c.inputs["Base Color"].default_value = (0.95, 0.65, 0.05, 1.0) # 越野亮黄色
                    bsdf_c.inputs["Roughness"].default_value = 0.12
                    bsdf_c.inputs["Metallic"].default_value = 0.10
                chassis.data.materials.append(mat_chassis)
                
                # 创建钢制传动轴材质 (16个空格缩进)
                mat_steel = bpy.data.materials.new(name="mat_axle_steel")
                mat_steel.use_nodes = True
                bsdf_s = mat_steel.node_tree.nodes.get("Principled BSDF")
                if bsdf_s:
                    bsdf_s.inputs["Base Color"].default_value = (0.06, 0.06, 0.06, 1.0)
                    bsdf_s.inputs["Roughness"].default_value = 0.45
                    bsdf_s.inputs["Metallic"].default_value = 0.85
                axle_f.data.materials.append(mat_steel)
                axle_b.data.materials.append(mat_steel)
                for sp in sh_parts:
                    sp.data.materials.append(mat_steel)
                    
                # 核心：采用 Parenting 绑定关系 [1.1.2] (16个空格缩进)
                axle_f.parent = chassis
                axle_b.parent = chassis
                for sp in sh_parts:
                    sp.parent = chassis
                    
                # 隐藏模板 [1] (16个空格缩进)
                chassis.hide_render = True; chassis.hide_viewport = True
                axle_f.hide_render = True; axle_f.hide_viewport = True
                axle_b.hide_render = True; axle_b.hide_viewport = True
                for sp in sh_parts:
                    sp.hide_render = True; sp.hide_viewport = True
                    
                imp = chassis
            
            # 放入缓存中，设置为模板 (12个空格缩进)
            mesh_cache[cache_key] = imp
            
        else:
            # 如果存在 .obj 文件，正常加载 (12个空格缩进)
            # Blender 4.5's STL importer does not reliably populate
            # selected_objects.  Resolve the imported mesh by object-set delta.
            before_import = set(bpy.data.objects)
            if path.lower().endswith(".stl"):
                if hasattr(bpy.ops.wm, "stl_import"):
                    bpy.ops.wm.stl_import(filepath=path)
                else:
                    bpy.ops.import_mesh.stl(filepath=path)
            elif hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
            else:
                bpy.ops.import_scene.obj(filepath=path, axis_forward="Y", axis_up="Z")
            imported = [o for o in bpy.data.objects if o not in before_import and o.type == "MESH"]
            if not imported:
                raise RuntimeError(f"Robot mesh import produced no mesh object: {path}")
            imp = imported[0]
            imp.name = f"robot_template_{mf}"
            for c in imp.users_collection:
                c.objects.unlink(imp)
            imp.hide_render = True; imp.hide_viewport = True
            mesh_cache[cache_key] = imp

    # ────────────────【核心修改：将实例化移至 if/else 外部，采用 4 个空格缩进】────────────────
    # 这确保了无论是初次创建还是从 Cache 中读取模板，都能正确实例化小车的所有零件！ [1]
    template = mesh_cache[cache_key]
    obj = clone_obj_recursive(template, robot_coll)  # 递归克隆父子级 [1.1.2]
    obj.name = f"robot_geom_{i:02d}_{g['body_name']}_{g['mesh_name']}"
    obj.rotation_mode = "QUATERNION"
            
    rgba = g.get("rgba") or [0.6, 0.6, 0.6, 1.0]
    mat = bpy.data.materials.new(name=f"robot_mat_{g['mesh_name']}_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
        is_metal = (g.get("material") == "metal")
        bsdf.inputs["Metallic"].default_value = 0.85 if is_metal else 0.05
        bsdf.inputs["Roughness"].default_value = 0.30 if is_metal else 0.55
    obj.data = obj.data.copy()
    obj.data.materials.clear(); obj.data.materials.append(mat)
    geom_objs.append(obj)

# ---------------------------------------------------------------------------
# Jackal: use a single rigid, engineering-style Blender assembly driven by
# MuJoCo's free-body pose.  The upstream STL meshes use a different local-axis
# convention after MuJoCo compilation; animating them as independent objects
# tears the vehicle apart.  This assembly keeps every visible part in a stable
# local frame while preserving the physical base pose from the rollout.
# ---------------------------------------------------------------------------
jackal_root = None
if robot_name == "jackal":
    for obj in geom_objs:
        obj.hide_render = True
        obj.hide_viewport = True
    jackal_coll = bpy.data.collections.new("jackal_assembled")
    bpy.context.scene.collection.children.link(jackal_coll)
    jackal_root = bpy.data.objects.new("jackal_physical_root", None)
    jackal_root.rotation_mode = "QUATERNION"
    jackal_coll.objects.link(jackal_root)

    def _jackal_mat(name, color, metallic=0.0, roughness=0.45):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        return mat

    jackal_black = _jackal_mat("jackal_assembly_black", (0.035, 0.04, 0.035, 1), 0.15, 0.32)
    jackal_yellow = _jackal_mat("jackal_assembly_yellow", (0.82, 0.55, 0.015, 1), 0.20, 0.30)
    jackal_rubber = _jackal_mat("jackal_assembly_rubber", (0.012, 0.014, 0.012, 1), 0.0, 0.82)
    jackal_metal = _jackal_mat("jackal_assembly_metal", (0.20, 0.23, 0.24, 1), 0.80, 0.22)

    def _box(name, loc, size, mat, bevel=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = tuple(2.0 * v for v in size)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if bevel:
            mod = obj.modifiers.new("edge_rounding", "BEVEL")
            mod.width = bevel; mod.segments = 3
        obj.data.materials.append(mat)
        obj.parent = jackal_root
        return obj

    def _cylinder(name, loc, radius, depth, mat, rot=(0.0, 0.0, 0.0)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=depth,
                                            location=loc, rotation=rot)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(mat)
        obj.parent = jackal_root
        return obj

    # Chassis and visible engineering details, all in the canonical Jackal body frame.
    _box("jackal_main_chassis", (0.0, 0.0, 0.124), (0.284, 0.209, 0.124), jackal_black, 0.025)
    _box("jackal_top_deck", (0.0, 0.0, 0.267), (0.245, 0.165, 0.026), jackal_yellow, 0.012)
    _box("jackal_left_guard", (0.0, 0.277, 0.159), (0.304, 0.024, 0.054), jackal_yellow, 0.008)
    _box("jackal_right_guard", (0.0, -0.277, 0.159), (0.304, 0.024, 0.054), jackal_yellow, 0.008)
    _box("jackal_front_bumper", (0.317, 0.0, 0.101), (0.034, 0.236, 0.047), jackal_black, 0.010)
    _cylinder("jackal_lidar_pedestal", (0.061, 0.0, 0.314), 0.043, 0.049, jackal_metal)
    _cylinder("jackal_lidar", (0.061, 0.0, 0.350), 0.065, 0.032, jackal_black)
    _cylinder("jackal_lidar_band", (0.061, 0.0, 0.351), 0.067, 0.006, jackal_yellow)

    for label, x, y in (("front_left", 0.177, 0.254), ("front_right", 0.177, -0.254),
                        ("rear_left", -0.177, 0.254), ("rear_right", -0.177, -0.254)):
        _cylinder(f"jackal_{label}_wheel", (x, y, 0.0466), 0.1323, 0.054,
                  jackal_rubber, (math.pi / 2.0, 0.0, 0.0))
        _cylinder(f"jackal_{label}_hub", (x, y, 0.0466), 0.070, 0.057,
                  jackal_metal, (math.pi / 2.0, 0.0, 0.0))
    # Replace the simple chassis/wheel proxies with three meshes preassembled
    # from Clearpath's official Jackal URDF.  They are already in the body
    # frame, so only the one physical root is animated.
    for obj in jackal_coll.objects:
        if (obj.name == "jackal_main_chassis" or "_wheel" in obj.name
                or "_hub" in obj.name):
            obj.hide_render = True
            obj.hide_viewport = True

    def _official_jackal_mesh(name, filename, mat):
        before = set(bpy.data.objects)
        asset_path = os.path.join(GO2_DIR, "assets", filename)
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=asset_path)
        else:
            bpy.ops.import_mesh.stl(filepath=asset_path)
        imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        if not imported:
            raise RuntimeError(f"Official Jackal import produced no mesh: {asset_path}")
        obj = imported[0]
        obj.name = name
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        jackal_coll.objects.link(obj)
        obj.parent = jackal_root
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.data.materials.clear(); obj.data.materials.append(mat)
        return obj

    _official_jackal_mesh("jackal_official_body", "jackal_official_body.stl", jackal_black)
    _official_jackal_mesh("jackal_official_fenders", "jackal_official_fenders.stl", jackal_yellow)
    _official_jackal_mesh("jackal_official_wheels", "jackal_official_wheels.stl", jackal_rubber)

    # Micro-offroad details remain in the canonical (unscaled) Jackal frame.
    # The root below applies the same 3x factor as MuJoCo's wheel collisions.
    for label, x, y, side in (("front_left", 0.177, 0.254, 1), ("front_right", 0.177, -0.254, -1),
                              ("rear_left", -0.177, 0.254, 1), ("rear_right", -0.177, -0.254, -1)):
        _cylinder(f"jackal_{label}_beadlock", (x, y + side * 0.029, 0.0466), 0.071, 0.008,
                  jackal_metal, (math.pi / 2.0, 0.0, 0.0))
        _cylinder(f"jackal_{label}_rim_ring", (x, y + side * 0.034, 0.0466), 0.091, 0.005,
                  jackal_yellow, (math.pi / 2.0, 0.0, 0.0))
    _box("jackal_left_led", (0.338, 0.135, 0.150), (0.010, 0.030, 0.018), jackal_yellow, 0.003)
    _box("jackal_right_led", (0.338, -0.135, 0.150), (0.010, 0.030, 0.018), jackal_yellow, 0.003)
    _box("jackal_left_roof_rail", (0.0, 0.155, 0.270), (0.155, 0.010, 0.010), jackal_black, 0.003)
    _box("jackal_right_roof_rail", (0.0, -0.155, 0.270), (0.155, 0.010, 0.010), jackal_black, 0.003)
    jackal_root.scale = (3.0, 3.0, 3.0)

    # Render the exact heightfield sampled by MuJoCo, rather than Blender's
    # independently evaluated terrain modifier stack.  This makes physical
    # wheel contact and visible ground elevation share one surface.
    terrain_src = bpy.data.objects.get("OpaqueTerrain_fine")
    keys_path = os.path.join(EXPORT_DIR, "module1", "keys.json")
    with open(keys_path) as _keys_file:
        _hf = json.load(_keys_file)["terrain_hfield"]
    _heights = np.loadtxt(os.path.join(EXPORT_DIR, _hf["elevation_rel"]))
    # MuJoCo normalizes the hfield samples so their largest value is exactly
    # one before applying size.z.  Mirror that compiler behavior here.
    _heights /= max(float(np.max(_heights)), 1e-12)
    _rows, _cols = _heights.shape
    _xmin, _xmax = float(_hf["x_min"]), float(_hf["x_max"])
    _ymin, _ymax = float(_hf["y_min"]), float(_hf["y_max"])
    _zmin, _zmax = float(_hf["z_min"]), float(_hf["z_max"])
    _verts = [(_xmin + (_xmax - _xmin) * c / (_cols - 1),
               _ymax - (_ymax - _ymin) * r / (_rows - 1),
               _zmin + (_zmax - _zmin) * float(_heights[r, c]))
              for r in range(_rows) for c in range(_cols)]
    _faces = []
    for r in range(_rows - 1):
        for c in range(_cols - 1):
            a = r * _cols + c; b = a + 1; d = (r + 1) * _cols + c; e = d + 1
            _faces.extend(((a, d, b), (b, d, e)))
    _mesh = bpy.data.meshes.new("mujoco_hfield_visual_mesh")
    _mesh.from_pydata(_verts, [], _faces); _mesh.update()
    _terrain = bpy.data.objects.new("mujoco_hfield_visual", _mesh)
    bpy.context.scene.collection.objects.link(_terrain)
    # Reuse the original high-fidelity material.  Its three POINT attributes
    # are sampled onto the exact hfield vertices in XY, so the visual ground
    # keeps the Blend scene's erosion/ground/mountain variation while its
    # elevation remains exactly the MuJoCo collision surface.
    if terrain_src:
        from mathutils.kdtree import KDTree
        _src_mesh = terrain_src.data
        _src_world = terrain_src.matrix_world
        _kd = KDTree(len(_src_mesh.vertices))
        for _i, _v in enumerate(_src_mesh.vertices):
            _p = _src_world @ _v.co
            _kd.insert((_p.x, _p.y, 0.0), _i)
        _kd.balance()
        _nearest = [_kd.find((_v[0], _v[1], 0.0))[1] for _v in _verts]
        for _attr_name in ("ground_collection", "eroded", "mountain_collection"):
            _src_attr = _src_mesh.attributes.get(_attr_name)
            if _src_attr is None:
                continue
            _dst_attr = _mesh.attributes.new(_attr_name, "FLOAT", "POINT")
            for _i, _src_i in enumerate(_nearest):
                _dst_attr.data[_i].value = _src_attr.data[_src_i].value
        if terrain_src.data.materials:
            _terrain.data.materials.append(terrain_src.data.materials[0])
        terrain_src.hide_render = True
        terrain_src.hide_viewport = True
    for _poly in _mesh.polygons:
        _poly.use_smooth = True
    print(f"[render-v2] replaced visual terrain with exact normalized MuJoCo hfield ({_rows}x{_cols})")
    print("[render-v2] built official URDF Jackal mesh assembly driven by MuJoCo base pose")

# ---------------------------------------------------------------------------
# Dynamic prop:
#   schema A: import the source boulder OBJ from manifest, scale by size_scale,
#             apply a fallback rock material.
#   schema B: hide the original prop_dynamic object in scene.blend; create a
#             copy that we keyframe per-frame. The copy keeps the original
#             materials so the bottle / mug looks right.
# ---------------------------------------------------------------------------
size_scale = float(log.get("stone_size_scale", 1.0))
stone_obj = None
if industrial_robot:
    print("[render-v2] industrial robot path: no animated dynamic prop")
elif not src_name or src_name == "None":
    # 完美拦截：无石子情况下安全越过，不让程序继续往下漏进 else 从而引发 AttributeError [1]
    print("[render-v2] no dynamic prop (stone) detected in rollout, skipping import")
elif schema_a:
    obj_path = os.path.join(EXPORT_DIR, boulder_meta["mesh_rel"])
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="Y", axis_up="Z")
    stone_obj = bpy.context.selected_objects[0]
    stone_obj.name = "kick_stone"
    stone_obj.scale = (size_scale, size_scale, size_scale)
    stone_obj.rotation_mode = "QUATERNION"
    
    # ⚠️ 黄金级克隆：直接获取原版巨石对象，深度借用其高精度苔藓岩石材质
    borrowed_mats = []
    orig_obj = bpy.data.objects.get(src_name)
    
    if orig_obj and orig_obj.data and orig_obj.data.materials:
        borrowed_mats = list(orig_obj.data.materials)
        print(f"[render-v2] 成功从小石头的模板巨石 '{src_name}' 克隆了 {len(borrowed_mats)} 个高精度材质贴图！")
    else:
        # 备用方案：如果特定巨石没找到，搜索场景中其他带有 Rock/Boulder 的物体
        for o in bpy.data.objects:
            if o.type == "MESH" and o.data and o.data.materials:
                if any(kw in o.name.lower() for kw in ["boulder", "rock", "stone", "terrain_fine"]):
                    borrowed_mats = list(o.data.materials)
                    print(f"[render-v2] 备用方案激活：从场景物体 '{o.name}' 借用了材质贴图。")
                    break

    # 应用材质
    stone_obj.data = stone_obj.data.copy()
    stone_obj.data.materials.clear()
    
    if borrowed_mats:
        for m in borrowed_mats:
            stone_obj.data.materials.append(m)
    else:
        # 终极暗色调色兜底（深褐色岩石色），防止出现刺眼的灰色
        stone_mat = bpy.data.materials.new(name="kick_stone_mat")
        stone_mat.use_nodes = True
        bsdf = stone_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.22, 0.18, 0.15, 1.0)  # 深褐土岩色
            bsdf.inputs["Roughness"].default_value = 0.95
        stone_obj.data.materials.append(stone_mat)
        print("[render-v2] 未能借用到材质，应用了深褐色岩石兜底材质。")
        
    print(f"[render-v2] schema-A stone: imported '{obj_path}' 且已应用真实环境材质纹理。")
else:
    # Find the prop_dynamic object's source name in scene.blend and clone it.
    src_meta = dyn_props.get(src_name)
    blend_obj_name = src_meta.get("source_name", src_meta.get("name"))
    src_in_blend = bpy.data.objects.get(blend_obj_name)
    if src_in_blend is None:
        print(f"[render-v2] FATAL: '{blend_obj_name}' not in scene.blend")
        sys.exit(3)
    # Find the prop_dynamic object's source name in scene.blend, hide it,
    # and import the SAME mesh OBJ we exported for MuJoCo.  This mirrors
    # schema-A: a clean newly imported object whose origin is at the OBJ
    # origin (= MuJoCo body-frame origin), with no inherited delta_*
    # offsets, parent, constraints, or actions.  We borrow the original
    # Bottle's materials so it still looks right.
    src_meta = dyn_props.get(src_name)
    blend_obj_name = src_meta.get("source_name", src_meta.get("name"))
    src_in_blend = bpy.data.objects.get(blend_obj_name)
    if src_in_blend is None:
        print(f"[render-v2] FATAL: '{blend_obj_name}' not in scene.blend")
        sys.exit(3)
    src_in_blend.hide_render = True
    src_in_blend.hide_viewport = True
    borrowed_mats = list(src_in_blend.data.materials) if src_in_blend.data else []

    obj_rel = src_meta.get("mesh_rel")
    obj_path = os.path.join(EXPORT_DIR, obj_rel)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="Y", axis_up="Z")
    stone_obj = bpy.context.selected_objects[0]
    stone_obj.name = f"animated_{blend_obj_name}"
    stone_obj.rotation_mode = "QUATERNION"
    stone_obj.location = (0.0, 0.0, 0.0)
    stone_obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    stone_obj.scale = (1.0, 1.0, 1.0)
    stone_obj.hide_render = False
    stone_obj.hide_viewport = False
    if borrowed_mats:
        stone_obj.data = stone_obj.data.copy()
        stone_obj.data.materials.clear()
        for m in borrowed_mats:
            stone_obj.data.materials.append(m)
    print(f"[render-v2] schema-B stone: imported '{obj_path}', borrowed {len(borrowed_mats)} mats from '{blend_obj_name}' (original hidden)")


# ---------------------------------------------------------------------------
# Camera: schema-A uses the existing scene camera (already set above).
# Schema-B: keep the c2w original camera (camera_0_0) untouched. The rig
# parent chain makes manipulating .location unreliable. The user can
# manually pick a different camera in scene.blend if needed; for now we
# trust c2w's choice — they framed the scene for the prompt scenario.
# ---------------------------------------------------------------------------
#print(f"[render-v2] using camera '{bpy.context.scene.camera.name}' at world {bpy.context.scene.camera.matrix_world.translation}")
print(f"[render-v2] using camera '{bpy.context.scene.camera.name}' at world {bpy.context.scene.camera.matrix_world.translation}")

# ---------------------------------------------------------------------------
# Animate.
# ---------------------------------------------------------------------------
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = len(mj_frames)
scene.render.fps = log["fps"]

# Drive the camera with a frame-change handler rather than F-curves.  The
# source blend's original camera rig has nontrivial transform inheritance;
# keyframing it caused evaluated positions to explode.  A world matrix written
# after timeline evaluation stays exact and is used by render(animation=True).
if mj_follow_camera:
    def _sync_mj_tracking_camera(_scene, *_unused):
        _idx = max(0, min(_scene.frame_current - 1, len(mj_frames) - 1))
        _mid = Vector(mj_frames[_idx]["jackal_base_pos"]) + mj_to_world_offset
        _pos = _mid - fwd_v * mj_cam_distance
        _rot = (_mid - _pos).normalized().to_track_quat("-Z", "Y")
        cam_obj.matrix_world = Matrix.LocRotScale(_pos, _rot, None)
        print(f"[render-v2] camera-sync frame={_scene.frame_current} desired={tuple(round(v,4) for v in _pos)} actual={tuple(round(v,4) for v in cam_obj.matrix_world.translation)}")
    bpy.app.handlers.frame_change_post.append(_sync_mj_tracking_camera)
    _sync_mj_tracking_camera(scene)

for f_idx, frame in enumerate(mj_frames):
    bf = f_idx + 1
    scene.frame_set(bf)
    if jackal_root is not None:
        jackal_root.location = Vector(frame["jackal_base_pos"]) + mj_to_world_offset
        jackal_root.rotation_quaternion = Quaternion(frame["jackal_base_quat"])
        jackal_root.keyframe_insert(data_path="location", frame=bf)
        jackal_root.keyframe_insert(data_path="rotation_quaternion", frame=bf)
    else:
        xforms = frame[robot_xform_key] if industrial_robot else frame["go2_geom_xforms"]
        for obj, x in zip(geom_objs, xforms):
            if obj is None:
                continue
            loc = Vector(x["pos"]) + mj_to_world_offset
            quat = Quaternion(x["quat"])
            obj.location = loc
            obj.rotation_quaternion = quat
            obj.keyframe_insert(data_path="location", frame=bf)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=bf)
    if not industrial_robot and stone_obj is not None:
        sloc = Vector(frame["stone_pos"]) + mj_to_world_offset
        squat = Quaternion(frame["stone_quat"])
        stone_obj.location = sloc
        stone_obj.rotation_quaternion = squat
        stone_obj.keyframe_insert(data_path="location", frame=bf)
        stone_obj.keyframe_insert(data_path="rotation_quaternion", frame=bf)


# Diagnostic for the MuJoCo-follow camera: confirms the actual evaluated
# transform and the first scene object hit by its center ray.
if mj_follow_camera:
    scene.frame_set(1)
    _deps = bpy.context.evaluated_depsgraph_get()
    _cam_pos = cam_obj.matrix_world.translation.copy()
    _cam_fwd = (cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    _hit, _loc, _normal, _idx, _obj, _matrix = bpy.context.scene.ray_cast(_deps, _cam_pos, _cam_fwd, distance=500.0)
    print(f"[render-v2] mj-camera probe pos={tuple(round(v,4) for v in _cam_pos)} fwd={tuple(round(v,4) for v in _cam_fwd)} hit={_hit} obj={_obj.name if _hit else None} loc={tuple(round(v,4) for v in _loc) if _hit else None}")


# ---------------------------------------------------------------------------
# Cycles GPU render.
# ---------------------------------------------------------------------------
render_engine = os.environ.get("BL_RENDER_ENGINE", "CYCLES").upper()
scene.render.engine = "BLENDER_EEVEE_NEXT" if render_engine == "EEVEE" else "CYCLES"
prefs = bpy.context.preferences
cprefs = prefs.addons["cycles"].preferences
for backend in ("OPTIX", "CUDA"):
    try:
        cprefs.compute_device_type = backend; break
    except TypeError:
        continue
cprefs.refresh_devices()
n_gpu = 0
for dev in cprefs.devices:
    if dev.type in {"OPTIX", "CUDA"}:
        dev.use = True; n_gpu += 1
    else:
        dev.use = False
print(f"[render-v2] Cycles {cprefs.compute_device_type}, {n_gpu} GPU(s)")
if scene.render.engine == "CYCLES":
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
else:
    scene.render.image_settings.color_mode = "RGBA"
print(f"[render-v2] engine={scene.render.engine}")
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUTPUT_DIR, "frame_")
print(f"[render-v2] rendering {scene.frame_end} frames...")

# Ultimate Render-Time Bypass: Hide muddy water and disable terrain displacement
for obj in bpy.data.objects:
    if "liquid" in obj.name.lower() or "water" in obj.name.lower() or "lake" in obj.name.lower():
        obj.hide_render = True
        obj.hide_viewport = True
        print("Drained water at render time: " + obj.name)
    # Preserve native terrain geometry and geometry nodes for the visual pass.

# The frame-change handler installs the MuJoCo camera matrix after each
# timeline update; Cycles can therefore render the sequence as one batch.
bpy.ops.render.render(animation=True)
print(f"[render-v2] DONE -> {OUTPUT_DIR}")