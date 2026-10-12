"""Module 2, sub-step 1: PostProcess Agent dual-output.

Reads:
  EXPORT_DIR/manifest.json          (C2W static scene)
  EXPORT_DIR/module1/keys.json      (KEY boulders + physical attrs)
  USER_PROMPT (sys.argv[2])

Calls DashScope (Qwen-Coder) and asks it to translate the prompt into a
`dynamics_program.json` matching the Module 2 schema in the TPAMI notes:

  {
    "simulator_dynamics": [
      {"object": "...", "type": "rigid_body", "source_boulder": "...",
       "initial_pose":  {"pos": [x,y,z], "quat": [w,x,y,z]},
       "mass": kg, "friction": mu,
       "initial_velocity":         [vx,vy,vz],
       "initial_angular_velocity": [wx,wy,wz],
       "duration": seconds}
    ],
    "articulated_dynamics": [...],
    "visual_effects":       [...]
  }

The LLM is given the list of KEY boulders + their VLM-estimated physical
attrs and must pick one as a "source" for the rolling boulder, plus a
release pose + initial velocity that physically makes sense for the
prompt (e.g. "rolls down toward Go2" -> upslope of Go2's path, velocity
pointed at Go2).

Output: EXPORT_DIR/module2/dynamics_program.json

Run:
    python sim/module2_dyn_program.py EXPORT_DIR "A boulder breaks loose ..."
"""
from __future__ import annotations

import json
import os
import re
import sys

from openai import OpenAI


if len(sys.argv) < 3:
    print('usage: module2_dyn_program.py EXPORT_DIR "user prompt"')
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
USER_PROMPT = sys.argv[2]
M2_DIR = os.path.join(EXPORT_DIR, "module2")
os.makedirs(M2_DIR, exist_ok=True)

API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("C2W_TEXT_MODEL", "qwen3-coder-plus")

if not API_KEY:
    print("FATAL: set DASHSCOPE_API_KEY (or OPENAI_API_KEY)")
    sys.exit(3)

with open(os.path.join(EXPORT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
with open(os.path.join(EXPORT_DIR, "module1", "keys.json")) as f:
    keys = json.load(f)

key_summary = []
for kb in keys.get("key_boulders", []):
    name = kb["name"]
    phys = keys.get("physical", {}).get(name, {})
    vlm = phys.get("vlm", {})
    geom = phys.get("geometry", {})
    key_summary.append({
        "name": name,
        "world_pos": kb["world_pos"],
        "world_size": kb["world_size"],
        "material": vlm.get("material"),
        "mass_kg": geom.get("mass_kg"),
        "friction": vlm.get("friction"),
    })

groups = manifest.get("groups", {})
terrain = manifest.get("terrain") or next(iter(groups.get("background_terrain", [])), {})
terrain_aabb = terrain.get("world_aabb", [0]*6)
dynamic_candidates = groups.get("prop_dynamic", [])
boulders = manifest.get("boulders", groups.get("prop_concave_large", []))

scene_summary = {
    "terrain_aabb_xyz_minmax": terrain_aabb,
    "n_boulders_total": len(boulders),
    "key_boulders": key_summary,
    "dynamic_candidates": [
        {"name": item["name"], "source_name": item.get("source_name"),
         "world_pos": item.get("world_pos"), "world_size": item.get("world_size")}
        for item in dynamic_candidates
    ],
}

SYSTEM_PROMPT = """You are the PostProcess Agent for Code2Worlds-2 Embodied World Simulation.

You translate a natural-language dynamic-event description into a STRICT JSON
program that drives BOTH a MuJoCo physics rollout AND Blender visual rendering.

# Scene context
You are given:
  - terrain AABB
  - the list of KEY boulders (the only ones with VLM-grounded mass/friction).
  - HARD RULE: ALL boulders that appear in C2W's manifest -- including every
    KEY boulder -- are PERMANENTLY FUSED into the terrain by Module 1. They
    NEVER become dynamic. You may only reference a KEY boulder's NAME as a
    visual+geometry TEMPLATE for a brand-new, independent small stone.
  - A Unitree Go2 quadruped sits at origin (x=0, y=0, z=0.27) and walks
    forward along +X.

# Task
Output a JSON object with three top-level keys, each a list:

  "simulator_dynamics": NEW small kickable/pushable stones, independent
                        of the static terrain. For each:
      {
        "object": "<unique name, e.g. 'kick_stone_0'>",
        "type":   "rigid_body",
        "source_boulder": "<name of a KEY boulder whose CoACD mesh + material
                           are reused as the visual+collision TEMPLATE. The
                           static KEY boulder remains in the terrain; this
                           is just a stylistic clone, scaled down.>",
        "size_scale": <float 0.05-0.20: how much to shrink the source mesh.
                       The KEY boulders are 2-3 m across, so 0.10-0.15
                       gives ~20-40 cm kickable rocks.>,
        "initial_pose": {"pos": [x,y,z], "quat": [w,x,y,z]},
        "mass":     <kg, 2.0 - 20.0 -- a stone Go2 can actually displace>,
        "friction": <mu, 0.5 - 0.9>,
        "initial_velocity":         [vx,vy,vz],   "m/s, usually [0,0,0]",
        "initial_angular_velocity": [wx,wy,wz],   "rad/s, usually [0,0,0]",
        "duration": <seconds of MuJoCo rollout>
      }

  "articulated_dynamics": joint-based dynamics (door, drawer). Empty for
                          outdoor scenes.

  "visual_effects": Blender-only effects (rain, fog, dust). For each:
      {"effect": "...", "type": "particle|shader|...", ...free schema...}

# Hard constraints
- Output ONE valid JSON object inside a ```json``` fence and nothing else.
- All numbers are SI units.
- For "Go2 walks across rocky terrain and kicks a small stone aside":
  place 1-3 small stones (size_scale 0.10-0.15, mass 5-15 kg) on Go2's
  forward path at x = +0.4 .. +1.2 m, y in [-0.15, +0.15] (one of them
  near a foot strike). Stones at rest: initial_velocity = [0,0,0],
  initial_angular_velocity = [0,0,0]. z = 0.05 + size_scale * 1.5
  (ground level + half stone height).
- duration: 3-5 s.
- source_boulder names MUST come from the provided key_boulders list.
- NEVER write "release boulder_XXX" or use a manifest boulder name as
  the dynamic object name. The dynamic object is ALWAYS new, e.g.
  "kick_stone_0", "kick_stone_1".
"""

USER_TEMPLATE = """User instruction: {instruction}

Scene context (JSON):
{scene}

Generate the dynamics_program JSON now."""

INDOOR_SYSTEM_PROMPT = """You are the Code2Worlds-2 dynamics planner for an existing indoor Blender scene. Return one JSON object with simulator_dynamics, articulated_dynamics, visual_effects, and robot_action. Select an existing prop_dynamic by its exact name using reuse_existing. Give the dynamic object a distinct object name, an initial_pose near that source object's world_pos, positive mass and friction, zero or small initial velocities, and a duration in seconds. Keep static props and room geometry fixed. The robot_action is kick, walk, or watch. Never include benchmark probes, validators, or success thresholds. Use SI units. Visual effects stay in Blender and do not become MuJoCo bodies."""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
print(f"[m2-dyn] LLM={MODEL_NAME}  prompt={USER_PROMPT!r}")
print(f"[m2-dyn] {len(key_summary)} KEY boulders to choose from")

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": INDOOR_SYSTEM_PROMPT if dynamic_candidates else SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            instruction=USER_PROMPT,
            scene=json.dumps(scene_summary, indent=2),
        )},
    ],
    temperature=0.2,
)
raw = resp.choices[0].message.content.strip()

m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
js_text = m.group(1).strip() if m else raw

try:
    program = json.loads(js_text)
except json.JSONDecodeError as e:
    print(f"[m2-dyn] FAILED to parse JSON: {e}")
    print(raw)
    sys.exit(4)

valid_names = {b["name"] for b in key_summary}
valid_dynamic = {item["name"] for item in dynamic_candidates}
raw_dynamic = program.get("simulator_dynamics", [])
sim_items = raw_dynamic.get("dynamic_objects", []) if isinstance(raw_dynamic, dict) else raw_dynamic
for item in sim_items:
    src = item.get("source_boulder")
    if src and src not in valid_names:
        raise ValueError(f"source_boulder {src!r} not in KEY list")
    reused = item.get("reuse_existing")
    if reused and reused not in valid_dynamic:
        raise ValueError(f"reuse_existing {reused!r} not in dynamic candidates")

out_path = os.path.join(M2_DIR, "dynamics_program.json")
with open(out_path, "w") as f:
    json.dump(program, f, indent=2)

n_sim = len(sim_items)
n_art = len(program.get("articulated_dynamics", []))
n_vis = len(program.get("visual_effects", []))
print(f"[m2-dyn] DONE -> {out_path}")
print(f"[m2-dyn]  {n_sim} simulator_dynamics, {n_art} articulated, {n_vis} visual_effects")
for it in sim_items:
    print(f"          rigid: {it.get('object')} <- {it.get('source_boulder')}  "
          f"pos={it.get('initial_pose',{}).get('pos')}  "
          f"v={it.get('initial_velocity')}  "
          f"w={it.get('initial_angular_velocity')}")
