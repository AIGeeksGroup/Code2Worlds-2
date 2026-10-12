"""Module 2 rollout for the core Go2 path and specialized robot branches.

Reads dynamics_program.json + scene_m2.xml, runs MuJoCo simulation with
PD-controlled Go2, records per-frame physics + visual geom transforms.

Supports:
  robot_action="kick" -> three-phase kick (HOME/LIFT/STRIKE/RELAX)
  robot_action="watch" -> static home pose

Outputs:
  EXPORT_DIR/module2/rollout.mp4
  EXPORT_DIR/module2/rollout_first.png  rollout_last.png
  EXPORT_DIR/module2/rollout_log.json   (per-frame physics + Go2 visual xforms)
  EXPORT_DIR/module2/rollout_log_blender.json  (postproc for Blender)

Run:
    MUJOCO_GL=osmesa python sim/module2_rollout_v2.py EXPORT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import imageio.v3 as iio  # noqa: E402
import mujoco             # noqa: E402
import numpy as np        # noqa: E402


if len(sys.argv) < 2:
    print("usage: module2_rollout_v2.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
M2_DIR = os.path.join(EXPORT_DIR, "module2")
SCENE_XML = os.environ.get("C2W_M2_SCENE_XML", os.path.join(M2_DIR, "scene_m2.xml"))
PROGRAM = os.path.join(M2_DIR, "dynamics_program.json")
RENDER = os.environ.get("C2W_RENDER", "1") != "0"
RENDER_WIDTH = int(os.environ.get("C2W_RENDER_WIDTH", "1280"))
RENDER_HEIGHT = int(os.environ.get("C2W_RENDER_HEIGHT", "720"))

with open(PROGRAM) as f:
    program = json.load(f)


def _is_industrial_arm_program(program: dict) -> bool:
    robot = program.get("robot", {})
    return (robot.get("kind") == "industrial_arm"
            or str(program.get("robot_model", "")).startswith("universal_robots"))


def _interp_waypoints(waypoints: list[list[float]], alpha: float) -> np.ndarray:
    if len(waypoints) == 1:
        return np.asarray(waypoints[0], dtype=float)
    alpha = max(0.0, min(1.0, alpha))
    scaled = alpha * (len(waypoints) - 1)
    i = min(int(np.floor(scaled)), len(waypoints) - 2)
    t = scaled - i
    a = np.asarray(waypoints[i], dtype=float)
    b = np.asarray(waypoints[i + 1], dtype=float)
    return (1.0 - t) * a + t * b


def _collect_visual_geoms(model: mujoco.MjModel, robot_name: str) -> list[dict]:
    visual_geoms = []
    for gid in range(model.ngeom):
        if model.geom_group[gid] != 2:
            continue
        # 允许 Mesh 和 Cylinder 两种几何体 [7.1.3]
        if model.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue

        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

        # 修正逻辑判断 [1.2.5]
        if not ("car" in body_name or "wheel" in body_name):
            continue

        mesh_id = int(model.geom_dataid[gid])
        mesh_name = "cylinder"
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)

        mat_id = int(model.geom_matid[gid])
        mat_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
                    if mat_id >= 0 else None)
        rgba = (model.mat_rgba[mat_id].tolist() if mat_id >= 0
                else model.geom_rgba[gid].tolist())
        visual_geoms.append({
            "geom_id": int(gid), "mesh_name": mesh_name,
            "mesh_file": f"{mesh_name}.obj", "body_name": body_name,
            "material": mat_name, "rgba": rgba,
        })
    return visual_geoms


def _geom_xforms(model: mujoco.MjModel, data: mujoco.MjData, visual_geoms: list[dict]) -> list[dict]:
    mat_buf = np.zeros(9)
    quat_buf = np.zeros(4)
    out = []
    for g in visual_geoms:
        gid = g["geom_id"]
        mat_buf[:] = np.asarray(data.geom_xmat[gid]).reshape(-1)
        mujoco.mju_mat2Quat(quat_buf, mat_buf)
        out.append({"pos": data.geom_xpos[gid].tolist(), "quat": quat_buf.tolist()})
    return out


def _rollout_industrial_arm() -> None:
    robot = program.get("robot", {})
    control = robot.get("control", {})
    robot_name = robot.get("name") or robot.get("model") or "ur5e"
    robot_name = "ur5e" if "ur5" in robot_name else robot_name
    duration = float(control.get("duration") or program.get("duration") or 4.0)
    fps = 30
    n_frames = int(duration * fps)
    print(f"[m2-roll-v2] industrial duration={duration}s robot={robot_name}")
    print(f"[m2-roll-v2] load {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("m2_init").id)
    mujoco.mj_forward(model, data)

    joint_names = control.get("joint_names") or [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    qpos_addr = [model.joint(n).qposadr[0] for n in joint_names]
    joint_ranges = [model.jnt_range[model.joint(n).id].tolist() for n in joint_names]
    waypoints = control.get("waypoints") or program.get("joint_waypoints")
    if not waypoints:
        waypoints = [[float(data.qpos[a]) for a in qpos_addr]]
    visual_geoms = _collect_visual_geoms(model, robot_name)
    site_id = -1
    for sname in (robot.get("ee_site"), "attachment_site"):
        if not sname:
            continue
        try:
            site_id = model.site(sname).id
            break
        except KeyError:
            pass

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    if site_id >= 0:
        cam.lookat[:] = data.site_xpos[site_id]
    cam.distance = 2.8
    cam.azimuth = 135
    cam.elevation = -22
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH) if RENDER else None
    frames = []
    steps_per_frame = max(1, int((1.0 / fps) / model.opt.timestep))
    f6 = np.zeros(6)
    log = {
        "fps": fps, "duration": duration,
        "robot_model": robot.get("model") or program.get("robot_model"),
        "robot_action": program.get("robot_action"),
        "task_description": program.get("task_description"),
        "weld_path": program.get("weld_path"),
        "joint_names": joint_names,
        "joint_ranges": joint_ranges,
        f"{robot_name}_visual_geoms": visual_geoms,
        "frames": [], "issues": [],
    }
    max_contacts = 0; max_force = 0.0; joint_limit_violations = 0
    for fi in range(n_frames):
        t = fi / fps
        target = _interp_waypoints(waypoints, fi / max(n_frames - 1, 1))
        for _ in range(steps_per_frame):
            data.ctrl[:len(target)] = target[:model.nu]
            mujoco.mj_step(model, data)
        q = np.asarray([data.qpos[a] for a in qpos_addr])
        for value, (lo, hi) in zip(q, joint_ranges):
            if value < lo - 1e-3 or value > hi + 1e-3:
                joint_limit_violations += 1
        contact_force = 0.0
        for ci in range(data.ncon):
            mujoco.mj_contactForce(model, data, ci, f6)
            contact_force = max(contact_force, abs(float(f6[0])))
        max_contacts = max(max_contacts, int(data.ncon))
        max_force = max(max_force, contact_force)
        fr = {
            "frame": fi + 1, "t": round(t, 4),
            "target_qpos": target[:len(joint_names)].tolist(),
            "qpos": q.tolist(), "ctrl": data.ctrl[:len(joint_names)].tolist(),
            "ee_pos": data.site_xpos[site_id].tolist() if site_id >= 0 else [0, 0, 0],
            "ncon": int(data.ncon),
            "contact_force_max": round(contact_force, 4),
            f"{robot_name}_geom_xforms": _geom_xforms(model, data, visual_geoms),
        }
        log["frames"].append(fr)
        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        if fi % 15 == 0 or fi == n_frames - 1:
            print(f"  frame {fi:3d} t={t:.2f}s q0={q[0]:.3f} ncon={data.ncon}")
    if joint_limit_violations:
        log["issues"].append(f"joint_limit_violations={joint_limit_violations}")
    log["summary"] = {
        "max_contacts": max_contacts,
        "max_contact_force": round(max_force, 4),
        "joint_limit_violations": joint_limit_violations,
        "rollout_ok": not log["issues"],
    }
    if frames:
        iio.imwrite(os.path.join(M2_DIR, "rollout_first.png"), frames[0])
        iio.imwrite(os.path.join(M2_DIR, "rollout_last.png"), frames[-1])
        iio.imwrite(os.path.join(M2_DIR, "rollout.mp4"), frames, fps=fps, codec="libx264")
    with open(os.path.join(M2_DIR, "rollout_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log["summary"], indent=2))


if _is_industrial_arm_program(program):
    _rollout_industrial_arm()
    sys.exit(0)

if "jackal" in str(program.get("robot_model", "")).lower() or "jackal" in str(program.get("robot", {}).get("model", "")).lower():
    from module2_rollout_jackal import run as _rollout_jackal
    _rollout_jackal(EXPORT_DIR, program)
    sys.exit(0)

raw_dyn = program.get("simulator_dynamics", [])
sim_dyn = raw_dyn.get("dynamic_objects", []) if isinstance(raw_dyn, dict) else raw_dyn
if not sim_dyn:
    raise ValueError("Go2 rollout needs at least one simulator_dynamics object")

duration = float(sim_dyn[0].get("duration", 4.0))
robot_action = program.get("robot_action", "watch")
print(f"[m2-roll-v2] duration={duration}s robot_action={robot_action}")

W, H = RENDER_WIDTH, RENDER_HEIGHT
FPS = 30
N_FRAMES = int(duration * FPS)


print(f"[m2-roll-v2] load {SCENE_XML}")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
print(f"[m2-roll-v2] nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom}")

kid = model.key("m2_init").id
mujoco.mj_resetDataKeyframe(model, data, kid)
mujoco.mj_forward(model, data)

stone_obj_name = sim_dyn[0]["object"]
stone_body = model.body(stone_obj_name).id
go2_base = model.body("base").id
print(f"[m2-roll-v2] stone='{stone_obj_name}' id={stone_body} go2_base={go2_base}")


# Discover Go2 leg joint slices dynamically (no LEG_QPOS=14 assumption).
# Find all 12 leg joint qpos indices by searching joints with names
# matching FL/FR/RL/RR pattern. Joints in order: hip, thigh, calf x 4 legs.
LEG_NAMES = []
for prefix in ("FL", "FR", "RL", "RR"):
    for joint in ("_hip_joint", "_thigh_joint", "_calf_joint"):
        LEG_NAMES.append(prefix + joint)
LEG_QPOS_IDX = []
LEG_QVEL_IDX = []
for jn in LEG_NAMES:
    try:
        j = model.joint(jn)
        LEG_QPOS_IDX.append(j.qposadr[0])
        LEG_QVEL_IDX.append(j.dofadr[0])
    except KeyError:
        pass
LEG_QPOS_IDX = np.array(LEG_QPOS_IDX)
LEG_QVEL_IDX = np.array(LEG_QVEL_IDX)
print(f"[m2-roll-v2] {len(LEG_QPOS_IDX)} leg joints found "
      f"(qpos[{LEG_QPOS_IDX[0] if len(LEG_QPOS_IDX) else '-'}..])")


HOME_LEG = np.array([0.0, 0.9, -1.8])
LIFT_LEG = np.array([0.0, 1.6, -2.6])
KICK_LEG = np.array([0.0, 0.3, -1.8])
WALK_STANCE = np.array([0.0, 0.82, -1.62])
WALK_SWING = np.array([0.0, 1.16, -2.18])
KP = np.array([55.0, 85.0, 85.0] * 4)
KV = np.array([2.0, 3.5, 3.5] * 4)


def make_target_kick(t: float) -> np.ndarray:
    """Outdoor: front-left leg swings through PRESTANCE/LIFT/STRIKE."""
    target = np.tile(HOME_LEG, 4).astype(np.float64)
    if 0.6 <= t < 0.9:
        target[0:3] = LIFT_LEG
    elif 0.9 <= t < 1.3:
        target[0:3] = KICK_LEG
    return target


def make_target_watch(t: float) -> np.ndarray:
    """Indoor / passive: hold home pose for the entire rollout."""
    return np.tile(HOME_LEG, 4).astype(np.float64)


WALK_VX = 0.3  # m/s target forward speed
ROLL_KP = 0.6  # roll feedback gain (rad -> hip rad)

def make_target_walk(t: float) -> np.ndarray:
    """Periodic trot + roll stabilisation.

    Trot: WALK_STANCE <-> WALK_SWING alternate between (FL+RR) and (FR+RL).
    Roll feedback: read base xmat, compute roll, push left/right hips to resist.
    """
    phase = (t % 0.52) / 0.52
    swing_a = phase < 0.5
    alpha = phase / 0.5 if swing_a else (phase - 0.5) / 0.5
    swing = WALK_STANCE * (1.0 - alpha) + WALK_SWING * alpha
    st    = WALK_STANCE.copy()
    sw    = swing.copy()
    if swing_a:  # FL+RR swing, FR+RL stance
        tgt = np.concatenate([sw, st, st, sw]).astype(np.float64)
    else:         # FR+RL swing, FL+RR stance
        tgt = np.concatenate([st, sw, sw, st]).astype(np.float64)
    # Roll correction: base xmat row 2 gives world-frame z in body frame
    xmat = data.body("base").xmat.reshape(3, 3)
    roll = float(np.arctan2(xmat[2, 1], xmat[2, 2]))   # rad, + = right lean
    hip_corr = np.clip(-ROLL_KP * roll, -0.3, 0.3)
    # FL=idx0, FR=idx3, RL=idx6, RR=idx9  (hip is first of each 3-tuple)
    tgt[0]  += hip_corr   # FL hip: resist right lean
    tgt[9]  += hip_corr   # RR hip
    tgt[3]  -= hip_corr   # FR hip
    tgt[6]  -= hip_corr   # RL hip
    return tgt


def pd_torque(target_legs: np.ndarray) -> np.ndarray:
    err = target_legs - data.qpos[LEG_QPOS_IDX]
    return np.clip(KP * err - KV * data.qvel[LEG_QVEL_IDX], -23.7, 23.7)


if robot_action == "kick":
    make_target = make_target_kick
elif robot_action == "walk":
    make_target = make_target_walk
else:
    make_target = make_target_watch


# Camera: auto-frame around the dynamic body so indoor/outdoor both look ok.
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
stone_pos0 = data.body(stone_obj_name).xpos.copy()
go2_pos0 = data.body("base").xpos.copy()
mid = (stone_pos0 + go2_pos0) / 2
cam.lookat[:] = mid
cam.distance = max(2.5, float(np.linalg.norm(stone_pos0 - go2_pos0)) * 2.5)
cam.azimuth = 110.0
cam.elevation = -18.0
print(f"[m2-roll-v2] cam lookat={mid} distance={cam.distance:.2f}")

renderer = mujoco.Renderer(model, height=H, width=W) if RENDER else None
frames: list[np.ndarray] = []


# Discover Go2 visual geoms (group=2 + type=mesh) -- mirrors v1, used by Blender.
go2_visual_geoms = []
for gid in range(model.ngeom):
    if model.geom_group[gid] != 2:
        continue
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    body_id = int(model.geom_bodyid[gid])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if not (body_name == "base" or body_name.startswith(
            ("FL_", "FR_", "RL_", "RR_"))):
        continue
    mesh_id = int(model.geom_dataid[gid])
    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    mat_id = int(model.geom_matid[gid])
    mat_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
                if mat_id >= 0 else None)
    rgba = (model.mat_rgba[mat_id].tolist() if mat_id >= 0
            else model.geom_rgba[gid].tolist())
    go2_visual_geoms.append({
        "geom_id": int(gid), "mesh_name": mesh_name,
        "mesh_file": f"{mesh_name}.obj", "body_name": body_name,
        "material": mat_name, "rgba": rgba,
    })
print(f"[m2-roll-v2] {len(go2_visual_geoms)} Go2 visual geoms")


# Rollout main loop.
log = {
    "fps": FPS, "duration": duration, "stone_object": stone_obj_name,
    "stone_source_boulder": sim_dyn[0].get("source_boulder")
                            or sim_dyn[0].get("reuse_existing"),
    "stone_size_scale": sim_dyn[0].get("size_scale", 1.0),
    "robot_action": robot_action,
    "go2_visual_geoms": go2_visual_geoms,
    "frames": [],
}

# Find stone freejoint qvel slice (it's the freejoint of the dynamic body).
stone_joint_id = None
for j in range(model.njnt):
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        continue
    if int(model.jnt_bodyid[j]) == stone_body:
        stone_joint_id = j
        break
assert stone_joint_id is not None, "stone freejoint not found"
stone_qvel_start = int(model.jnt_dofadr[stone_joint_id])
print(f"[m2-roll-v2] stone freejoint qvel[{stone_qvel_start}:{stone_qvel_start+6}]")

steps_per_frame = int((1.0 / FPS) / model.opt.timestep)
print(f"[m2-roll-v2] steps_per_frame={steps_per_frame}")

t_start = time.time()
for fi in range(N_FRAMES):
    t = fi / FPS
    for sub in range(steps_per_frame):
        sub_t = t + sub * model.opt.timestep
        data.ctrl[:] = pd_torque(make_target(sub_t))
        if robot_action == "walk":
            data.qvel[6] = min(data.qvel[6] + 0.05, WALK_VX)  # soft ramp
        mujoco.mj_step(model, data)

    spos = data.body(stone_obj_name).xpos.copy()
    squat = data.body(stone_obj_name).xquat.copy()
    sv = data.qvel[stone_qvel_start:stone_qvel_start+3]
    sw = data.qvel[stone_qvel_start+3:stone_qvel_start+6]

    contact_count = 0; fmax = 0.0; ftot = 0.0
    f6 = np.zeros(6)
    for ci in range(data.ncon):
        c = data.contact[ci]
        if (model.geom_bodyid[c.geom1] == stone_body
                or model.geom_bodyid[c.geom2] == stone_body):
            contact_count += 1
            mujoco.mj_contactForce(model, data, ci, f6)
            fn = abs(float(f6[0]))
            ftot += fn
            if fn > fmax: fmax = fn

    geom_xforms = []
    mat_buf = np.zeros(9); quat_buf = np.zeros(4)
    for g in go2_visual_geoms:
        gid = g["geom_id"]
        pos = data.geom_xpos[gid].tolist()
        mat_buf[:] = np.asarray(data.geom_xmat[gid]).reshape(-1)
        mujoco.mju_mat2Quat(quat_buf, mat_buf)
        geom_xforms.append({"pos": pos, "quat": quat_buf.tolist()})

    joint_limit_violations = []
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id] or model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        position = float(data.qpos[model.jnt_qposadr[joint_id]])
        low, high = model.jnt_range[joint_id]
        if position < low - 1e-4 or position > high + 1e-4:
            joint_limit_violations.append(model.joint(joint_id).name)

    log["frames"].append({
        "t": round(t, 4),
        "stone_pos": spos.tolist(), "stone_quat": squat.tolist(),
        "stone_lin_speed": float(np.linalg.norm(sv)),
        "stone_ang_speed": float(np.linalg.norm(sw)),
        "stone_contacts": contact_count,
        "stone_contact_force_max": round(fmax, 3),
        "stone_contact_force_total": round(ftot, 3),
        "joint_limit_violations": joint_limit_violations,
        "go2_base_pos": data.body("base").xpos.tolist(),
        "go2_base_quat": data.body("base").xquat.tolist(),
        "go2_base_z": float(data.body("base").xpos[2]),
        "go2_geom_xforms": geom_xforms,
    })

    if robot_action == "walk":
        cam.lookat[:] = data.body("base").xpos
    if renderer is not None:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    if fi % 15 == 0 or fi == N_FRAMES - 1:
        print(f"  frame {fi:3d} t={t:.2f}s spos={spos} |v|={np.linalg.norm(sv):.3f}"
              f" ncon={contact_count} Fmax={fmax:.1f}N")

elapsed = time.time() - t_start
print(f"[m2-roll-v2] sim+render in {elapsed:.1f}s ({N_FRAMES/elapsed:.1f} fps)")

stone_pos_final = data.body(stone_obj_name).xpos.copy()
displacement = float(np.linalg.norm(stone_pos_final - stone_pos0))
log["stone_initial_pos"] = stone_pos0.tolist()
log["stone_final_pos"] = stone_pos_final.tolist()
log["stone_displacement_m"] = round(displacement, 4)

if frames:
    iio.imwrite(os.path.join(M2_DIR, "rollout_first.png"), frames[0])
    iio.imwrite(os.path.join(M2_DIR, "rollout_last.png"), frames[-1])
    iio.imwrite(os.path.join(M2_DIR, "rollout.mp4"), frames, fps=FPS, codec="libx264")
with open(os.path.join(M2_DIR, "rollout_log.json"), "w") as f:
    json.dump(log, f, indent=2)
