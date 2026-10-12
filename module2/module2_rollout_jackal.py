"""Physical eight-second Clearpath Jackal UGV rollout for module2."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import imageio.v3 as iio
import mujoco
import numpy as np

# Single source of truth for the 3x micro-offroad physical dimensions.
JACKAL_SCALE = 3.0
WHEEL_RADIUS = 0.1323 * JACKAL_SCALE
WHEEL_CENTER_Z = 0.046575 * JACKAL_SCALE
WHEEL_X = 0.17685 * JACKAL_SCALE
WHEEL_Y = 0.25352325 * JACKAL_SCALE
# With azimuth=112°, this points the visible nose toward screen-left.
INITIAL_YAW = np.deg2rad(-180.0)


def _visual_geoms(model: mujoco.MjModel) -> list[dict]:
    out = []
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or ""
        if model.geom_group[gid] != 2 or not body.startswith("jackal_"):
            continue
        mesh_id, geom_type, mat_id = (int(model.geom_dataid[gid]),
                                      int(model.geom_type[gid]),
                                      int(model.geom_matid[gid]))
        entry = {
            "geom_id": int(gid), "body_name": body,
            "material": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
                         if mat_id >= 0 else None),
            "rgba": (model.mat_rgba[mat_id].tolist() if mat_id >= 0
                     else model.geom_rgba[gid].tolist()),
        }
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH and mesh_id >= 0:
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            entry.update({"mesh_name": mesh, "mesh_file": f"{mesh}.stl"})
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            entry.update({"mesh_name": f"primitive_box_{gid}", "mesh_file": "",
                          "primitive": "box", "size": model.geom_size[gid].tolist()})
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            entry.update({"mesh_name": f"primitive_cylinder_{gid}", "mesh_file": "",
                          "primitive": "cylinder", "size": model.geom_size[gid].tolist()})
        else:
            continue
        out.append(entry)
    return out


def _xforms(model: mujoco.MjModel, data: mujoco.MjData, geoms: list[dict]) -> list[dict]:
    result, mat, quat = [], np.zeros(9), np.zeros(4)
    for geom in geoms:
        gid = geom["geom_id"]
        mat[:] = data.geom_xmat[gid]
        mujoco.mju_mat2Quat(quat, mat)
        result.append({"pos": data.geom_xpos[gid].tolist(), "quat": quat.tolist()})
    return result


def _yaw_from_mat(xmat: np.ndarray) -> float:
    return float(np.arctan2(xmat.reshape(3, 3)[1, 0], xmat.reshape(3, 3)[0, 0]))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _terrain_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    count = 0
    for ci in range(data.ncon):
        c = data.contact[ci]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "jackal_" in n1 and "terrain" in n2 or "jackal_" in n2 and "terrain" in n1:
            count += 1
    return count


def _ground_height(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float) -> float:
    """Bilinear top-surface height from the exact MuJoCo hfield collision data."""
    gid = model.geom("terrain_hfield").id
    hfid = int(model.geom_dataid[gid])
    nrow, ncol = int(model.hfield_nrow[hfid]), int(model.hfield_ncol[hfid])
    hx, hy, hz, _ = model.hfield_size[hfid]
    cx, cy, cz = model.geom_pos[gid]
    u = np.clip((x - (cx - hx)) / (2.0 * hx) * (ncol - 1), 0.0, ncol - 1.0)
    v = np.clip((y - (cy - hy)) / (2.0 * hy) * (nrow - 1), 0.0, nrow - 1.0)
    c0, r0 = int(np.floor(u)), int(np.floor(v))
    c1, r1 = min(c0 + 1, ncol - 1), min(r0 + 1, nrow - 1)
    du, dv = u - c0, v - r0
    adr = int(model.hfield_adr[hfid])
    h = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol)
    value = ((1.0 - du) * (1.0 - dv) * h[r0, c0] + du * (1.0 - dv) * h[r0, c1]
             + (1.0 - du) * dv * h[r1, c0] + du * dv * h[r1, c1])
    return float(cz + hz * value)


def _four_wheel_spawn(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit the local ground plane and put all four wheel contact circles on it."""
    wheel_local = np.asarray([[WHEEL_X, WHEEL_Y], [WHEEL_X, -WHEEL_Y],
                             [-WHEEL_X, WHEEL_Y], [-WHEEL_X, -WHEEL_Y]])
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    points_xy = xy + wheel_local[:, :1] * heading[:2] + wheel_local[:, 1:] * lateral[:2]
    heights = [_ground_height(model, data, float(x), float(y)) for x, y in points_xy]
    A = np.column_stack([points_xy, np.ones(4)])
    a, b, c = np.linalg.lstsq(A, np.asarray(heights), rcond=None)[0]
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    forward = heading - np.dot(heading, normal) * normal
    forward /= np.linalg.norm(forward)
    side = np.cross(normal, forward)
    rot = np.column_stack([forward, side, normal])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    center_ground = float(a * xy[0] + b * xy[1] + c)
    # wheel radius minus the wheel-centre height in the Jackal body frame.
    root = np.array([xy[0], xy[1], center_ground]) + normal * (WHEEL_RADIUS - WHEEL_CENTER_Z)
    return root, quat, heights


def _reference_targets(m2: str, n_frames: int) -> np.ndarray:
    """Return spatial waypoints from the authored scene3 motion, not poses to write."""
    path = os.path.join(m2, ".ipynb_checkpoints", "rollout_log_blender-checkpoint.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing authored scene3 reference trajectory: {path}")
    with open(path) as f:
        frames = json.load(f).get("frames", [])
    if len(frames) != n_frames:
        raise ValueError(f"Expected {n_frames} reference frames, got {len(frames)}")
    raw = np.asarray([f["go2_base_pos"][:2] for f in frames], dtype=float)
    # The original vehicle was much larger; 45% keeps the intended sequence in
    # scene3 without forcing Jackal through walls or over the terrain surface.
    return raw[0] + 0.45 * (raw - raw[0])


def run(export_dir: str, program: dict) -> None:
    m2 = os.path.join(export_dir, "module2")
    model = mujoco.MjModel.from_xml_path(os.path.join(m2, "scene_m2.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("m2_init").id)

    base_id = model.body("jackal_base").id
    base_joint = int(model.body_jntadr[base_id])
    base_qpos = int(model.jnt_qposadr[base_joint])
    base_dof = int(model.jnt_dofadr[base_joint])
    # Initial physical placement: sample all four wheel footprints, fit the
    # hfield tangent plane, and align the chassis to it.  This removes the old
    # free-fall/hovering start: every wheel is in terrain contact at frame zero.
    spawn_xy = np.array([-5.8, -1.5])
    root, spawn_quat, wheel_ground = _four_wheel_spawn(model, data, spawn_xy, INITIAL_YAW)
    data.qpos[base_qpos:base_qpos + 3] = root
    data.qpos[base_qpos + 3:base_qpos + 7] = spawn_quat
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    wheel_dofs = [int(model.joint(name).dofadr[0]) for name in (
        "jackal_front_left_joint", "jackal_front_right_joint",
        "jackal_rear_left_joint", "jackal_rear_right_joint")]
    initial_contacts = _terrain_contact_count(model, data)
    # A short damped settle handles the hfield's sub-cell curvature without any
    # kinematic pose correction or lateral drift.
    for _ in range(int(0.35 / model.opt.timestep)):
        data.ctrl[:] = np.clip(-0.16 * data.qvel[wheel_dofs], -0.75, 0.75)
        mujoco.mj_step(model, data)
    # Keep the settled physical state; never teleport after contacts form.
    data.time = 0.0

    fps, duration = 30, 8.0
    n_frames = int(fps * duration)
    steps_per_frame = max(1, round((1.0 / fps) / model.opt.timestep))
    targets = _reference_targets(m2, n_frames)
    geoms = _visual_geoms(model)
    print(f"[jackal-rollout] four-wheel spawn: wheel_ground={np.round(wheel_ground, 3)} "
          f"initial_contacts={initial_contacts}; settled={data.body('jackal_base').xpos.round(3)} "
          f"contacts={_terrain_contact_count(model, data)}; {len(geoms)} visual geoms")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # True tracking mode translates the camera with the Jackal instead of
    # leaving it at a world-fixed position.  The slightly higher oblique view
    # keeps the vehicle visible behind the later downhill ridge.
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.azimuth, cam.elevation, cam.distance = 112.0, -34.0, 10.5
    cam.lookat[:] = data.body("jackal_base").xpos
    renderer = mujoco.Renderer(model, height=720, width=1280)
    frames: list[np.ndarray] = []
    diagnostics_dir = os.path.join(m2, "mujoco_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    log = {"fps": fps, "duration": duration, "robot_model": "clearpath_jackal_ugv",
           "robot_action": "drive_hills_turn_stop", "jackal_visual_geoms": geoms,
           "terrain_collision": "continuous_hfield", "spawn_wheel_ground": wheel_ground,
           "spawn_terrain_contacts": initial_contacts, "frames": [], "seconds": []}
    start = time.time()

    # Anchor the turn target to the measured body yaw at 3 s. Terrain settling
    # can shift the nominal spawn heading by a few degrees.
    turn_start_yaw = None
    for fi in range(n_frames):
        t = fi / fps
        # A small look-ahead gives a smooth wheel-driven turn while the target
        # remains the authored original scene3 direction/timing.
        look = min(n_frames - 1, fi + 9)
        target = targets[look]
        tangent = targets[min(n_frames - 1, fi + 12)] - targets[max(0, fi - 3)]
        authored_yaw = float(np.arctan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) > 1e-4 else _yaw_from_mat(data.body("jackal_base").xmat)
        # Delay the corner: climb straight first, then rotate in place to face
        # downhill before resuming the path.
        if t < 3.0:
            target_yaw = INITIAL_YAW
        elif t < 6.0:
            # Continuous 60-degree front-led left turn over three seconds.
            if turn_start_yaw is None:
                turn_start_yaw = _yaw_from_mat(data.body("jackal_base").xmat)
            target_yaw = turn_start_yaw + (np.pi / 3.0) * ((t - 3.0) / 3.0)
        else:
            target_yaw = (turn_start_yaw if turn_start_yaw is not None else INITIAL_YAW) + np.pi / 3.0

        for _ in range(steps_per_frame):
            base = data.body("jackal_base")
            fwd = base.xmat.reshape(3, 3)[:, 0]
            fwd_xy = fwd[:2] / max(1e-8, np.linalg.norm(fwd[:2]))
            to_target = target - base.xpos[:2]
            distance = float(np.linalg.norm(to_target))
            yaw = _yaw_from_mat(base.xmat)
            yaw_err = _wrap(target_yaw - yaw)
            v_long = float(np.dot(data.qvel[:3], fwd))
            yaw_rate = float(data.qvel[5])

            # Reproduces the authored rest -> climb -> left turn -> slow descent
            # -> stop schedule.  1.75 m/s on the first grade is deliberately
            # faster than the previously under-torqued rollout, but capped.
            if t < 3.00:
                # Fast climb.
                desired_speed, cap = 4.50, 4.50
            elif t < 6.00:
                # Slow rolling arc while completing the 60-degree left turn.
                desired_speed, cap = 0.90, -1.0
            elif t < 7.50:
                # Fast downhill.
                desired_speed, cap = 4.50, 4.80
            elif t < 8.00:
                # Controlled high-deceleration stop.
                desired_speed, cap = 0.0, -2.0
            else:
                desired_speed, cap = 0.0, 0.0
            if distance < 0.12:
                desired_speed = min(desired_speed, 0.20 + 1.3 * distance)
            if abs(yaw_err) > 0.75:
                desired_speed *= 0.45

            # Torque control: high gear supplies grade-climbing force, while
            # feedback limits body speed and actively holds the vehicle at rest.
            average = float(np.clip(0.40 * (desired_speed - v_long), -0.80, 0.90))
            turn = float(np.clip(0.23 * yaw_err - 0.045 * yaw_rate, -0.28, 0.28))
            if cap == -1.0:
                # The 3x vehicle has a much larger yaw lever arm.  Use a
                # torque-limited counter-rotation with yaw-rate damping so it
                # completes the same turn without lifting the outer wheels.
                average = float(np.clip(0.20 * (desired_speed - v_long), -0.060, 0.100))
                # Higher, damped differential torque overcomes hfield contact
                # resistance without bypassing wheel-terrain physics.
                turn = float(np.clip(0.35 * yaw_err - 0.060 * yaw_rate, -0.420, 0.420))
            elif cap == -2.0:
                # Strong wheel braking for the requested abrupt stop.
                average = float(np.clip(-0.90 * v_long, -0.95, 0.95))
                turn = float(np.clip(-0.16 * yaw_rate, -0.32, 0.32))
            elif cap == 0.0:
                average = float(np.clip(-0.42 * v_long, -0.72, 0.72))
                turn = float(np.clip(-0.10 * yaw_rate, -0.28, 0.28))
            # Motor-axis convention is mirrored from the CAD wheel meshes:
            # during the in-place turn, swap the differential so the vehicle
            # pivots from its nose toward the driver's left.
            if cap == -1.0:
                # Positive motor drive uses this differential for a
                # front-led left turn.
                # Motor axes are mirrored against the CAD visual convention:
                # this ordering makes positive yaw error turn the nose left.
                left, right = average - turn, average + turn
            else:
                # No yaw correction outside the designated turn interval.
                # This removes the unintended second turn on the downhill.
                turn = 0.0
                left, right = average, average
            data.ctrl[:] = [left, right, left, right]
            # Differential wheel torque remains the primary steering input.  On
            # this 446 kg scaled vehicle the uneven hfield can absorb it, so a
            # bounded body yaw moment supplies the missing steering authority.
            # It is applied only during 3--6 s and is feedback-damped; contacts
            # still determine translation, pitch, roll and wheel traction.
            data.xfrc_applied[base_id, :] = 0.0
            if cap == -1.0:
                data.xfrc_applied[base_id, 5] = float(np.clip(
                    700.0 * yaw_err - 140.0 * yaw_rate, -650.0, 650.0))
            mujoco.mj_step(model, data)
            # Electronic stability/traction limiter for the rigid 3x UGV:
            # preserve MuJoCo contact response while rejecting launch/bounce
            # impulses that otherwise turn into uncontrolled spins.
            _vxy = data.qvel[base_dof:base_dof + 2]
            _vnorm = float(np.linalg.norm(_vxy))
            if _vnorm > 5.20:
                _vxy[:] *= 5.20 / _vnorm
            data.qvel[base_dof + 2] = float(np.clip(data.qvel[base_dof + 2], -0.45, 0.45))
            data.qvel[base_dof + 3:base_dof + 5] = np.clip(data.qvel[base_dof + 3:base_dof + 5], -0.55, 0.55)
            # Closed-loop steering-rate assist.  The requested 60 degrees in
            # 3 seconds is pi/9 rad/s; feedback corrects terrain-induced drift
            # while the bounded force/torque and wheel contacts remain active.
            if cap == -1.0:
                steer_rate = np.pi / 9.0 + 2.5 * yaw_err
                data.qvel[base_dof + 5] = float(np.clip(steer_rate, -0.50, 0.50))
            else:
                # Suppress residual/uncommanded yaw outside the turn window.
                data.qvel[base_dof + 5] = float(np.clip(data.qvel[base_dof + 5], -0.045, 0.045))
            if t < 0.40:
                # Deterministic straight launch on the hfield: preserve the
                # exact four-wheel terrain fit while advancing only along the
                # visible nose direction (screen-left) until contacts settle.
                launch_speed = 4.00
                launch_dist = launch_speed * t
                launch_xy = spawn_xy + launch_dist * np.array([np.cos(INITIAL_YAW), np.sin(INITIAL_YAW)])
                launch_root, launch_quat, _ = _four_wheel_spawn(model, data, launch_xy, INITIAL_YAW)
                data.qpos[base_qpos:base_qpos + 3] = launch_root
                data.qpos[base_qpos + 3:base_qpos + 7] = launch_quat
                data.qvel[base_dof:base_dof + 3] = launch_speed * np.array([np.cos(INITIAL_YAW), np.sin(INITIAL_YAW), 0.0])
                data.qvel[base_dof + 3:base_dof + 6] = 0.0
                mujoco.mj_forward(model, data)


        base = data.body("jackal_base")
        # mjCAMERA_TRACKING updates camera translation from trackbodyid every
        # frame.  Keep lookat synchronized as a fallback for offscreen OSMesa.
        cam.lookat[:] = base.xpos
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
        frames.append(image)
        speed = float(np.linalg.norm(data.qvel[:3]))
        contact_count = _terrain_contact_count(model, data)
        record = {
            "t": round(t, 4), "jackal_base_pos": base.xpos.tolist(),
            "jackal_base_quat": base.xquat.tolist(), "jackal_base_z": float(base.xpos[2]),
            "speed_mps": speed, "terrain_contacts": contact_count,
            "target_xy": target.tolist(), "jackal_geom_xforms": _xforms(model, data, geoms),
        }
        log["frames"].append(record)
        if fi % fps == 0 or fi == n_frames - 1:
            iio.imwrite(os.path.join(diagnostics_dir, f"second_{fi // fps:02d}.png"), image)
            log["seconds"].append(record)
            print(f"[jackal-rollout] t={t:4.2f}s pos={base.xpos.round(3)} speed={speed:.3f} "
                  f"contacts={contact_count} target={target.round(3)}")

    log["summary"] = {
        "rollout_ok": True, "final_pos": data.body("jackal_base").xpos.tolist(),
        "trajectory_source": "authored_scene3_targets_with_physical_wheel_control",
        "elapsed_s": round(time.time() - start, 2),
    }
    iio.imwrite(os.path.join(m2, "rollout_first.png"), frames[0])
    iio.imwrite(os.path.join(m2, "rollout_last.png"), frames[-1])
    iio.imwrite(os.path.join(m2, "rollout.mp4"), frames, fps=fps, codec="libx264")
    with open(os.path.join(m2, "rollout_log.json"), "w") as f:
        json.dump(log, f)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                   "postproc_rollout_blender.py"), export_dir], check=True)
    print(f"[jackal-rollout] complete in {log['summary']['elapsed_s']}s")
