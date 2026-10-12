"""Module 2, sub-step 4: simulator-in-the-loop critic.

Reads:
  EXPORT_DIR/module2/rollout_log.json   (per-frame physics signals)
  EXPORT_DIR/module2/dynamics_program.json
  EXPORT_DIR/module2/rollout.mp4        (for VLM frame sampling)

Computes the seven physical critic signals described in the manuscript:
  1. penetration depth (max stone-vs-terrain interpenetration)
  2. collision pairs (stone contact count over time)
  3. object stability (post-strike linear/angular speed decay)
  4. contact force peak
  5. support relation (does the stone come to rest on terrain?)
  6. joint over-limit and Go2 uprightness
  7. robot reachability (did the kick produce stone displacement?)

Then samples 3 frames from rollout.mp4 and asks Qwen-VL whether the
visual evolution looks physically plausible. Combines both into a
single valid + feedback verdict written to critic_report.json. Missing
visual judgment never counts as fused acceptance.

Run:
    DASHSCOPE_API_KEY=sk-... python sim/module2_sim_critic.py EXPORT_DIR
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys

import numpy as np
import requests


if len(sys.argv) < 2:
    print("usage: module2_sim_critic.py EXPORT_DIR")
    sys.exit(2)

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not API_KEY:
    print("WARN: DASHSCOPE_API_KEY not set; VLM critic will be skipped.")

EXPORT_DIR = os.path.abspath(sys.argv[1])
M2_DIR = os.path.join(EXPORT_DIR, "module2")
LOG_PATH = os.path.join(M2_DIR, "rollout_log.json")
PROGRAM_PATH = os.path.join(M2_DIR, "dynamics_program.json")
MP4_PATH = os.path.join(M2_DIR, "rollout.mp4")
OUT_PATH = os.path.join(M2_DIR, "critic_report.json")

with open(LOG_PATH) as f:
    log = json.load(f)
with open(PROGRAM_PATH) as f:
    program = json.load(f)

frames = log["frames"]
raw_dyn = program.get("simulator_dynamics", [])
sim_dyn = raw_dyn.get("dynamic_objects", []) if isinstance(raw_dyn, dict) else raw_dyn
fps = log["fps"]
duration = log.get("duration", len(frames) / fps)
print(f"[critic] {len(frames)} frames @ {fps} fps "
      f"({duration:.2f}s); {len(sim_dyn)} dynamic objects")


# ----- 1. physics critic ---------------------------------------------------
def physics_critic(frames, program):
    requires_interaction = program.get("robot_action", "kick") in {"kick", "push"}
    stone_pos = np.array([f["stone_pos"] for f in frames])
    lin = np.array([f["stone_lin_speed"] for f in frames])
    ang = np.array([f["stone_ang_speed"] for f in frames])
    contacts = np.array([f["stone_contacts"] for f in frames])
    fmax = np.array([f["stone_contact_force_max"] for f in frames])
    ftot = np.array([f["stone_contact_force_total"] for f in frames])
    base_z = np.array([f.get("go2_base_z", 0.33) for f in frames])
    base_q = np.array([f.get("go2_base_quat", [1, 0, 0, 0])
                       for f in frames])

    # Strike window heuristic: t in [0.9, 1.3] from rollout script.
    strike_mask = np.array(
        [(0.9 <= f["t"] < 1.3) for f in frames])
    settle_mask = np.array([(f["t"] >= 1.5) for f in frames])

    initial_pos = stone_pos[0]
    final_pos = stone_pos[-1]
    displacement = float(np.linalg.norm(final_pos - initial_pos))
    peak_speed = float(lin.max())

    # 1) penetration depth: we don't have direct geom-vs-geom signed dist
    #    in MuJoCo log; proxy = how far stone z went below terrain at
    #    initial position. If never below initial, treat as 0.
    pen_depth = float(max(0.0, stone_pos[0, 2] - stone_pos[:, 2].min()))

    # 2) collision pairs: time-averaged + peak
    contact_avg = float(contacts.mean())
    contact_peak = int(contacts.max())

    # 3) stability: |v|, |w| in settle window
    if settle_mask.any():
        settle_lin = float(lin[settle_mask].mean())
        settle_ang = float(ang[settle_mask].mean())
    else:
        settle_lin = settle_ang = None

    # 4) contact force peak (only during strike)
    if strike_mask.any():
        strike_fmax = float(fmax[strike_mask].max())
    else:
        strike_fmax = 0.0
    overall_fmax = float(fmax.max())

    # 5) support relation: resting near the starting support with contact.
    final_contact = bool(settle_mask.any() and (contacts[settle_mask][-min(5, int(settle_mask.sum())):] > 0).any())
    end_on_ground = bool(
        (settle_lin is not None and settle_lin < 0.05)
        and (settle_ang is not None and settle_ang < 0.5)
        and (final_pos[2] < initial_pos[2] + 0.10)
        and final_contact
    )

    # 6) A yaw turn changes quaternion w but does not mean tip-over.
    base_up_z = 1.0 - 2.0 * (base_q[:, 1] ** 2 + base_q[:, 2] ** 2)
    base_min_up_z = float(base_up_z.min()) if len(base_up_z) else 1.0
    base_z_min = float(base_z.min())
    base_z_max = float(base_z.max())
    joint_limit_violations = sorted({name for frame in frames for name in frame.get("joint_limit_violations", [])})
    go2_collapsed = bool(base_z_min < 0.10 or base_min_up_z < 0.766)

    # 7) robot reachability / interaction success
    moved = bool(displacement > 0.02 and peak_speed > 0.10)

    return {
        "displacement_m": displacement,
        "peak_lin_speed_m_s": peak_speed,
        "penetration_depth_m": pen_depth,
        "stone_contacts_avg": contact_avg,
        "stone_contacts_peak": contact_peak,
        "settle_lin_speed_m_s": settle_lin,
        "settle_ang_speed_rad_s": settle_ang,
        "strike_window_force_peak_N": strike_fmax,
        "rollout_force_peak_N": overall_fmax,
        "end_on_ground": end_on_ground,
        "final_support_contact": final_contact,
        "go2_base_z_min": base_z_min,
        "go2_base_z_max": base_z_max,
        "go2_base_min_up_z": base_min_up_z,
        "go2_collapsed": go2_collapsed,
        "joint_limit_violations": joint_limit_violations,
        "interaction_succeeded": moved,
        "requires_interaction": requires_interaction,
    }


pcrit = physics_critic(frames, program)
print("[critic] physics signals:")
for k, v in pcrit.items():
    print(f"  {k}: {v}")


# ----- 2. visual critic via Qwen-VL ---------------------------------------
def sample_frames_from_mp4(mp4_path: str, n: int = 4) -> list[bytes]:
    """Sample n equally-spaced frames as PNG bytes via imageio."""
    import io
    import imageio.v3 as iio
    from PIL import Image
    meta = iio.immeta(mp4_path)
    n_total = int(round(duration * fps))
    idxs = [int((i + 0.5) * n_total / n) for i in range(n)]
    out = []
    arr_iter = iio.imiter(mp4_path)
    target = set(idxs)
    cache = {}
    for i, frame in enumerate(arr_iter):
        if i in target:
            cache[i] = frame
        if len(cache) == n:
            break
    for i in idxs:
        img = Image.fromarray(cache[i])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def call_vlm_video_critic(images: list[bytes], pcrit: dict,
                          program: dict) -> dict:
    """Send sampled frames + numerical physics to Qwen-VL."""
    if not API_KEY:
        return {"skipped": True, "reason": "no DASHSCOPE_API_KEY"}
    endpoint = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    raw_dyn = program.get("simulator_dynamics", [])
    sim_dyn = raw_dyn.get("dynamic_objects", []) if isinstance(raw_dyn, dict) else raw_dyn
    obj_descr = ", ".join(
        f"{d['object']} (source={d.get('source_boulder')}, "
        f"mass={d.get('mass')}kg)" for d in sim_dyn) or "(none)"
    prompt = (
        "You are a physics-aware critic for a robot simulation rollout. "
        f"The scene is an outdoor terrain with a Unitree Go2 quadruped. "
        f"Dynamic objects: {obj_descr}. "
        f"Numerical signals from MuJoCo:\n"
        f"  stone displacement: {pcrit['displacement_m']:.3f} m\n"
        f"  peak linear speed: {pcrit['peak_lin_speed_m_s']:.3f} m/s\n"
        f"  strike-window force peak: "
        f"{pcrit['strike_window_force_peak_N']:.1f} N\n"
        f"  settle linear speed: {pcrit['settle_lin_speed_m_s']}\n"
        f"  go2 collapsed: {pcrit['go2_collapsed']}\n"
        f"  interaction succeeded: {pcrit['interaction_succeeded']}\n\n"
        "I am sending 4 frames sampled across time (early -> late). "
        "Judge whether the visual evolution looks physically plausible. "
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "valid": <true|false>,\n'
        '  "issues": [<short string>, ...],\n'
        '  "feedback_for_dyn_program": "<one short sentence>",\n'
        '  "confidence": <0..1>\n'
        "}\n"
    )
    content = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(img).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    payload = {
        "model": "qwen-vl-max",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"raw": text[:500], "parse_error": True}
    return json.loads(m.group(0))


vlm_report = {"skipped": True, "reason": "no rollout.mp4"}
if os.path.exists(MP4_PATH):
    try:
        print(f"[critic] sampling 4 frames from {MP4_PATH}")
        imgs = sample_frames_from_mp4(MP4_PATH, n=4)
        print(f"[critic] querying Qwen-VL ({len(imgs)} frames)...")
        vlm_report = call_vlm_video_critic(imgs, pcrit, program)
    except Exception as e:
        vlm_report = {"failed": True, "error": str(e)}
print(f"[critic] vlm verdict: {vlm_report}")


# ----- 3. fuse + write report ---------------------------------------------
def fuse_verdict(pcrit: dict, vlm: dict) -> dict:
    issues = []
    if pcrit["go2_collapsed"]:
        issues.append("go2 collapsed (base z too low or tilted)")
    if pcrit["joint_limit_violations"]:
        issues.append("joint limits exceeded: " + ", ".join(pcrit["joint_limit_violations"]))
    if pcrit["stone_contacts_peak"] == 0:
        issues.append("dynamic object never contacted another body")
    if pcrit["requires_interaction"] and not pcrit["interaction_succeeded"]:
        issues.append("kick produced negligible stone motion")
    if pcrit["penetration_depth_m"] > 0.05:
        issues.append(
            f"stone penetrated {pcrit['penetration_depth_m']*100:.1f}cm "
            "below initial pose")
    if pcrit["strike_window_force_peak_N"] > 5000.0:
        issues.append("contact force peak unrealistically high")
    if (pcrit["end_on_ground"] is False
            and pcrit["interaction_succeeded"]):
        issues.append("stone did not settle to a stable pose")
    if isinstance(vlm, dict) and vlm.get("issues"):
        issues.extend([f"vlm: {x}" for x in vlm["issues"]])

    physics_valid = (
        not pcrit["go2_collapsed"]
        and not pcrit["joint_limit_violations"]
        and pcrit["stone_contacts_peak"] > 0
        and (not pcrit["requires_interaction"] or pcrit["interaction_succeeded"])
        and pcrit["penetration_depth_m"] < 0.05
        and pcrit["strike_window_force_peak_N"] < 5000.0
        and pcrit["end_on_ground"]
    )
    vlm_valid = bool(isinstance(vlm, dict) and vlm.get("valid") is True
                     and not vlm.get("skipped") and not vlm.get("failed"))
    if not vlm_valid:
        issues.append("visual validation missing or failed")
    overall = bool(physics_valid and vlm_valid)
    return {
        "valid": overall,
        "physics_valid": physics_valid,
        "vlm_valid": vlm_valid,
        "issues": issues,
        "feedback_for_dyn_program": (
            vlm.get("feedback_for_dyn_program", "")
            if isinstance(vlm, dict) else ""),
    }


verdict = fuse_verdict(pcrit, vlm_report)


# ----- 4. plots: displacement / speed / contact-force timeseries ---------
def plot_timeseries(frames, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.array([f["t"] for f in frames])
    pos = np.array([f["stone_pos"] for f in frames])
    p0 = pos[0]
    disp = np.linalg.norm(pos - p0, axis=1)
    vlin = np.array([f["stone_lin_speed"] for f in frames])
    vang = np.array([f["stone_ang_speed"] for f in frames])
    fmax = np.array([f["stone_contact_force_max"] for f in frames])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(ts, disp * 100, "-", color="C0")
    axes[0].set_ylabel("displacement (cm)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ts, vlin, "-", label="|v| (m/s)", color="C1")
    axes[1].plot(ts, vang, "-", label="|w| (rad/s)", color="C3", alpha=0.6)
    axes[1].set_ylabel("speed")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(ts, fmax, "-", color="C2")
    axes[2].set_ylabel("peak contact force (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Module 2 rollout signals: {os.path.basename(EXPORT_DIR)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[critic] plotted -> {out_path}")


PLOT_PATH = os.path.join(M2_DIR, "rollout_signals.png")
try:
    plot_timeseries(frames, PLOT_PATH)
except Exception as e:
    print(f"[critic] WARN plot failed: {e}")
    PLOT_PATH = None


report = {
    "physics": pcrit,
    "vlm": vlm_report,
    "verdict": verdict,
    "rollout_log": os.path.basename(LOG_PATH),
    "rollout_mp4": os.path.basename(MP4_PATH),
    "plots": os.path.basename(PLOT_PATH) if PLOT_PATH else None,
}
with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=2, allow_nan=False)

print()
print(f"[critic] verdict.valid = {verdict['valid']}")
print(f"[critic] issues = {verdict['issues']}")
print(f"[critic] -> {OUT_PATH}")
