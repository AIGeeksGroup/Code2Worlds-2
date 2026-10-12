"""Module 1, sub-step 1.5 (generic): select KEY props + DYNAMIC target.

Scene-agnostic: reads manifest["groups"] from export_scene_to_obj_v2.py.

KEY props (CoACD + VLM target):
  - drawn from groups["prop_concave_large"]
  - filtered by size + tri count (drop microscopic / massive / degenerate)
  - ranked by centrality wrt scene background (terrain or floor union AABB)
  - greedy spaced selection so the N picks are not clustered

DYNAMIC target (the object Module 2 will animate):
  - if groups["prop_dynamic"] is non-empty (indoor scenes have a hand-placed
    Bottle/Mug/etc.), pick the largest one as the target
  - else fall back to a small KEY prop in [0.3, 0.7]m as a kickable surrogate

Output keys.json has the same schema as v1 for backward compatibility:
  {key_boulders: [...], target_boulder: <name>, target_boulder_info: {...},
   franka_anchor_xyz: [...], scene_centre_xy: [...]}
where "boulder" is now any prop_concave_large entry.

Run:
    python sim/module1_select_keys_v2.py EXPORT_DIR
"""
from __future__ import annotations

import json
import math
import os
import sys


if len(sys.argv) < 2:
    print("usage: module1_select_keys_v2.py EXPORT_DIR")
    sys.exit(2)

EXPORT_DIR = os.path.abspath(sys.argv[1])
manifest_path = os.path.join(EXPORT_DIR, "manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

groups = manifest.get("groups", {})
key_candidates = groups.get("prop_concave_large", [])
dyn_candidates = groups.get("prop_dynamic", [])
print(f"[m1-keys-v2] {len(key_candidates)} concave-large props, "
      f"{len(dyn_candidates)} dynamic props in manifest")


# --- 1. size + tri filter ----------------------------------------------
SIZE_MIN, SIZE_MAX = 0.20, 3.50    # widened: indoor furniture spans 0.3-3 m
TRI_MIN = 60
candidates = []
for p in key_candidates:
    longest = max(p["world_size"])
    if not (SIZE_MIN <= longest <= SIZE_MAX):
        continue
    if p.get("n_tris", 0) < TRI_MIN:
        continue
    candidates.append(p)
print(f"[m1-keys-v2] {len(candidates)} after size [{SIZE_MIN},{SIZE_MAX}]m + "
      f"tris>={TRI_MIN}")


# --- 2. compute scene-centre AABB (background union, fall back to props) -
def collect_aabbs(*group_lists):
    boxes = []
    for grp in group_lists:
        for entry in grp:
            if "world_aabb" in entry:
                boxes.append(entry["world_aabb"])
    return boxes


bg_boxes = collect_aabbs(
    groups.get("background_terrain", []),
    groups.get("background_floor", []),
    groups.get("background_wall", []))
if bg_boxes:
    ax = min(b[0] for b in bg_boxes); bx = max(b[3] for b in bg_boxes)
    ay = min(b[1] for b in bg_boxes); by = max(b[4] for b in bg_boxes)
    print(f"[m1-keys-v2] background AABB x[{ax:.2f},{bx:.2f}] "
          f"y[{ay:.2f},{by:.2f}]")
elif candidates:
    xs = [p["world_pos"][0] for p in candidates]
    ys = [p["world_pos"][1] for p in candidates]
    ax, bx = min(xs), max(xs); ay, by = min(ys), max(ys)
    print(f"[m1-keys-v2] no background; using prop AABB "
          f"x[{ax:.2f},{bx:.2f}] y[{ay:.2f},{by:.2f}]")
else:
    ax = bx = ay = by = 0.0
cx, cy = (ax + bx) / 2, (ay + by) / 2
half_diag = max(bx - ax, by - ay) / 2 or 1.0


# --- 3. centrality + greedy spaced selection ---------------------------
def centrality(p):
    px, py, _ = p["world_pos"]
    return -math.hypot(px - cx, py - cy) / half_diag


candidates.sort(key=centrality, reverse=True)
N_KEYS = 10
MIN_SEPARATION = 1.5  # m  (indoor scenes are tighter than the 4 m we used outdoors)
picked = []
for c in candidates:
    if len(picked) >= N_KEYS:
        break
    cx_, cy_, _ = c["world_pos"]
    too_close = False
    for p in picked:
        px_, py_, _ = p["world_pos"]
        if math.hypot(cx_ - px_, cy_ - py_) < MIN_SEPARATION:
            too_close = True
            break
    if not too_close:
        picked.append(c)

print(f"[m1-keys-v2] picked {len(picked)} KEY props "
      f"(separated >= {MIN_SEPARATION}m)")
for i, p in enumerate(picked):
    print(f"  #{i:02d} {p['name']}  pos={[round(x,2) for x in p['world_pos']]}  "
          f"size={[round(x,2) for x in p['world_size']]}  "
          f"n_tris={p.get('n_tris','?')}")


# --- 4. dynamic target ------------------------------------------------
target = None
target_source = None
if dyn_candidates:
    # Prefer the dyn candidate closest to a camera if cameras exist; else
    # just the largest by volume.
    def vol(p):
        return p["world_size"][0] * p["world_size"][1] * p["world_size"][2]
    dyn_sorted = sorted(dyn_candidates, key=vol, reverse=True)
    target = dyn_sorted[0]
    target_source = "prop_dynamic"
elif picked:
    PUSH_MIN, PUSH_MAX = 0.20, 0.70
    push_cands = [p for p in picked
                  if PUSH_MIN <= max(p["world_size"]) <= PUSH_MAX]
    if not push_cands:
        push_cands = list(picked)
    target = push_cands[0]
    target_source = "prop_concave_large_fallback"

if target:
    print(f"\n[m1-keys-v2] DYNAMIC TARGET ({target_source}): {target['name']}  "
          f"pos={target['world_pos']}  size={target['world_size']}")
else:
    print("[m1-keys-v2] WARN: no dynamic target found; "
          "Module 2 will need a synthetic kick_stone")


# --- 5. emit keys.json (schema-compatible with v1) -------------------
ANCHOR_OFFSET = 0.7
if target is not None:
    tx, ty, tz = target["world_pos"]
    franka_anchor = [tx - ANCHOR_OFFSET, ty, tz + 0.5]
else:
    franka_anchor = [cx, cy, 0.5]

out = {
    "scene_centre_xy": [cx, cy],
    "size_filter": [SIZE_MIN, SIZE_MAX],
    "min_separation_m": MIN_SEPARATION,
    "key_boulders": picked,           # naming kept for v1 compat
    "target_boulder": target["name"] if target else None,
    "target_boulder_info": target,
    "target_source": target_source,
    "franka_anchor_xyz": franka_anchor,
}

out_dir = os.path.join(EXPORT_DIR, "module1")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "keys.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"[m1-keys-v2] wrote {out_path}")
