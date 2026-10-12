"""Module 1, sub-step 4b: VLM Physical Attribute Agent (DashScope Qwen-VL).

For each KEY boulder we send 4 rendered views + bbox metadata to Qwen-VL
and ask it to return JSON with:
  - dominant_material   (e.g. "granite", "sandstone")
  - density_kg_m3       (e.g. 2700)
  - friction            (mu, e.g. 0.7)
  - restitution         (e.g. 0.10)
  - confidence          (0..1)

We compute mass and inertia from mesh volume + density:
    mass = density * volume
    inertia = density * I_unit  (trimesh provides I_unit via mesh.moment_inertia
                                  scaled by density and volume)

Output: <export_dir>/module1/physical_attrs.json plus inline update of
        keys.json["physical"] = {boulder_name: {...}}.

Run:
    DASHSCOPE_API_KEY=sk-... python sim/module1_vlm_attrs.py EXPORT_DIR
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time

import numpy as np
import requests
import trimesh


if len(sys.argv) < 2:
    print("usage: module1_vlm_attrs.py EXPORT_DIR")
    sys.exit(2)

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
OFFLINE_REGISTRY = os.environ.get("C2W_OFFLINE_ATTRS", "0") == "1" or not API_KEY

EXPORT_DIR = os.path.abspath(sys.argv[1])
M1_DIR = os.path.join(EXPORT_DIR, "module1")
keys_path = os.path.join(M1_DIR, "keys.json")
with open(keys_path) as f:
    keys = json.load(f)

VIEW_DIR = os.path.join(M1_DIR, "vlm_views")
OUT_PATH = os.path.join(M1_DIR, "physical_attrs.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asset_registry import material_from_name  # noqa: E402

# DashScope OpenAI-compatible endpoint
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-vl-max"   # high-end VL model; cheap enough for 10 calls

PROMPT = """You are a physics-aware perception assistant. Below are 4 views (iso / front / side / top) of a single object rendered as a triangulated mesh from a synthetic 3D scene. The object's longest world dimension is {longest_dim_m:.2f} m.

The object's source name is: {source_name}

Identify the object's dominant material and physical properties. Pick the SINGLE BEST category from this generic list (not just rocks):

  rock_granite, rock_basalt, rock_sandstone, rock_limestone, rock_generic
  wood_hardwood, wood_softwood, wood_painted
  metal_steel, metal_aluminum, metal_iron
  plastic_hard, plastic_soft, rubber
  ceramic, glass, fabric, leather
  concrete, brick
  generic

Return ONLY a JSON object (no commentary, no markdown fences):

{{
  "dominant_material": "<one of the categories above>",
  "density_kg_m3": <number; realistic bulk density for that material>,
  "friction": <number 0.2..0.95, sliding friction vs another rigid object>,
  "restitution": <number 0.0..0.4, bounciness>,
  "confidence": <number 0..1>,
  "reasoning": "<one short sentence justifying the material choice>"
}}

Realistic density ranges (kg/m^3): rock 2200-2800, wood 350-900, steel 7800, aluminum 2700, plastic 900-1400, glass 2400-2600, ceramic 2300-2700, concrete 2200-2400, brick 1800-2000, fabric 200-400, rubber 1100-1500. Use the source name as a strong hint: "BoulderFactory" -> rock_*; "TableFactory"/"CabinetFactory"/"ShelfFactory" -> wood_*; "BedFactory" -> wood_* (frame); "BottleFactory" -> glass or plastic; "MugFactory"/"CupFactory" -> ceramic.
"""


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(images: list[str], prompt: str) -> dict:
    content = [{"type": "text", "text": prompt}]
    for p in images:
        b64 = img_to_b64(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # Tolerate ```json ... ``` fences if VLM ignores instructions
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in VLM response: {text[:200]}")
    return json.loads(m.group(0))


def compute_inertia_and_mass(mesh: trimesh.Trimesh, density_kg_m3: float):
    # trimesh assumes density=1 for raw moment_inertia; scale to real density
    if not mesh.is_watertight:
        mesh.fill_holes()
    vol_m3 = max(float(mesh.volume), 1e-6)
    mass = density_kg_m3 * vol_m3
    # principal moments of inertia at COM, axes aligned to mesh local frame
    mesh_density_scaled = mesh.copy()
    mesh_density_scaled.density = density_kg_m3
    inertia = mesh_density_scaled.moment_inertia
    com = mesh.center_mass.tolist()
    return {
        "volume_m3": vol_m3,
        "mass_kg": mass,
        "com_local": com,
        "inertia_3x3": inertia.tolist(),
        "diag_inertia": np.diag(inertia).tolist(),
    }


if OFFLINE_REGISTRY:
    print(f"[m1-vlm] offline registry mode enabled; using deterministic asset-registry "
          f"fallback for {len(keys['key_boulders'])} KEY props")
else:
    print(f"[m1-vlm] querying VLM for {len(keys['key_boulders'])} key boulders "
          f"(model={MODEL})")

physical = {}
for i, b in enumerate(keys["key_boulders"]):
    name = b["name"]
    longest = max(b["world_size"])
    view_dir = os.path.join(VIEW_DIR, name)
    images = [os.path.join(view_dir, f) for f in
              ["view_iso.png", "view_front.png", "view_side.png", "view_top.png"]]
    if not OFFLINE_REGISTRY and not all(os.path.exists(p) for p in images):
        print(f"[m1-vlm] SKIP {name}: missing views")
        continue

    t0 = time.time()
    source_name = b.get("source_name", name)
    try:
        if OFFLINE_REGISTRY:
            category, density, friction, restitution = material_from_name(source_name)
            attrs = {
                "dominant_material": category,
                "density_kg_m3": density,
                "friction": friction,
                "restitution": restitution,
                "confidence": 0.75 if category != "generic" else 0.35,
                "reasoning": f"Deterministic registry fallback for {source_name}",
                "source": "asset_registry_offline",
            }
        else:
            attrs = call_vlm(images, PROMPT.format(
                longest_dim_m=longest, source_name=source_name))
    except Exception as e:
        print(f"[m1-vlm] ERROR {name}: {e}; using generic default")
        attrs = {
            "dominant_material": "generic",
            "density_kg_m3": 1500,
            "friction": 0.6,
            "restitution": 0.1,
            "confidence": 0.0,
            "reasoning": f"VLM call failed: {e}",
        }
    dt = time.time() - t0

    # Clamp to sane physical ranges (covers wood..steel; fabric/rubber too)
    attrs["density_kg_m3"] = float(np.clip(attrs.get("density_kg_m3", 1500),
                                           150, 8500))
    attrs["friction"] = float(np.clip(attrs.get("friction", 0.6), 0.2, 1.0))
    attrs["restitution"] = float(np.clip(attrs.get("restitution", 0.1), 0.0, 0.5))

    # Compute mass + inertia from mesh + density
    mesh_path = os.path.join(EXPORT_DIR, b["mesh_rel"])
    mesh = trimesh.load(mesh_path, force="mesh")
    geo = compute_inertia_and_mass(mesh, attrs["density_kg_m3"])

    physical[name] = {
        "vlm": attrs,
        "geometry": geo,
        "longest_dim_m": longest,
        "vlm_seconds": round(dt, 2),
    }
    print(f"[m1-vlm]  #{i:02d} {name}: {attrs['dominant_material']:<14s} "
          f"rho={attrs['density_kg_m3']:.0f}  mu={attrs['friction']:.2f}  "
          f"e={attrs['restitution']:.2f}  vol={geo['volume_m3']:.3f}m^3  "
          f"mass={geo['mass_kg']:.1f}kg  ({dt:.1f}s)")

with open(OUT_PATH, "w") as f:
    json.dump(physical, f, indent=2)

keys["physical"] = physical
with open(keys_path, "w") as f:
    json.dump(keys, f, indent=2)

print(f"[m1-vlm] DONE: wrote {OUT_PATH} and updated {keys_path}")
