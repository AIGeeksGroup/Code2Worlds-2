"""Asset-class registry for Code2Worlds scenes.

Module 1's "per-asset collision plan" needs a way to classify each
Blender collection / object into an asset class. Different asset
classes get different collision strategies in Module 1 emit
(`module1_emit_mjcf.py`) and different dynamic-eligibility flags in
Module 2 (`module2_dyn_program.py`).

Adding a new scene type (industrial, hospital, kitchen, ...) is a
data-only patch: append rules to ASSET_RULES and BACKGROUND_RULES.
No pipeline code changes.

Asset classes:
  - background_terrain: outdoor terrain mesh (one per scene). Sliced
    into a grid of convex prisms by module1_terrain_grid.py.
  - background_floor:   indoor floor / room shell. Same grid strategy
    as terrain (locally near-convex per cell).
  - background_wall:    indoor walls / ceilings / structural surfaces.
    Single-mesh static collider; thin enough that MuJoCo's auto-convex
    is acceptable.
  - prop_concave_large: large freestanding object that may have real
    concavities (boulder, sofa, tree). KEY candidates for CoACD.
  - prop_convex_small:  small object that is roughly convex, treat as
    visual-mesh-as-collider (no CoACD). Default for most clutter.
  - prop_dynamic:       small graspable / kickable object that the
    LLM is allowed to spawn as a dynamic body in Module 2 (mug,
    bottle, cup).
  - vegetation:         visual-only, no collider (foliage explodes
    MuJoCo collision performance; we drop it from the simulation
    scene entirely but render it in Blender).
  - decoration:         visual-only, very small or thin (< 5 cm).
"""
from __future__ import annotations

import re
from typing import Optional


# Each rule: (pattern_regex, asset_class). First match wins. The
# pattern is matched against the **collection name** (case-sensitive
# substring search via re.search).
ASSET_RULES: list[tuple[str, str]] = [
    # Hand-authored Code2Worlds-compatible demo collections.
    (r"robot_visual_only|service_robot_visual_only", "decoration"),
    (r"water_grass_visual_only", "vegetation"),
    # --- Outdoor (scene_1, future natural scenes) ---
    # Infinigen keeps reusable source meshes in a hidden unique_assets
    # collection. Only the placed spawn_placeholder instances belong to the
    # simulated world; exporting the templates creates duplicate rocks at the
    # origin and can invalidate contact tests.
    (r"^unique_assets:BoulderFactory", "skip"),
    (r"BoulderFactory",       "prop_concave_large"),
    (r"TreeFactory",          "vegetation"),
    (r"BushFactory",          "vegetation"),
    (r"FernFactory",          "vegetation"),
    (r"GrassTuftFactory",     "vegetation"),
    (r"MonocotFactory",       "vegetation"),
    (r"SucculentFactory",     "vegetation"),
    (r"FlowerFactory|TreeFlowerFactory", "vegetation"),
    (r"LeafFactory|BranchFactory|GenericTreeFactory", "vegetation"),
    (r"FruitFactory",         "decoration"),
    (r"BlenderRockFactory",   "decoration"),
    # --- Indoor (scene_8/9/10, future indoor scenes) ---
    # Furniture: large concave-ish, KEY candidates for CoACD.
    (r"SofaFactory|CouchFactory",        "prop_concave_large"),
    (r"TableFactory|DeskFactory|CoffeeTable", "prop_concave_large"),
    (r"ChairFactory|StoolFactory",       "prop_concave_large"),
    (r"BedFactory|MattressFactory|BlanketFactory|BoxComforterFactory", "prop_concave_large"),
    (r"Cabinet|Shelf|Bookcase|Wardrobe|Drawer|CellShelfFactory", "prop_concave_large"),
    (r"ServiceCounterFactory|ReceptionDeskFactory|GateDeskFactory|AirportSeatFactory", "prop_concave_large"),
    (r"MirrorFactory",                  "prop_concave_large"),
    # Small, kickable / graspable -> dynamic candidates by default.
    (r"BottleFactory|MugFactory|CupFactory|GlassFactory", "prop_dynamic"),
    (r"BookFactory|BookStackFactory|VaseFactory|BowlFactory|PlateFactory", "prop_dynamic"),
    (r"AppleFactory|OrangeFactory",                       "prop_dynamic"),
    (r"BallFactory|ToyFactory",                           "prop_dynamic"),
    (r"PackageBoxFactory|LuggageFactory",                 "prop_dynamic"),
    # Indoor decoration / soft furnishings (rugs, curtains, pillows).
    (r"PillowFactory|RugFactory|CurtainFactory",          "decoration"),
    (r"PaintingFactory|FrameFactory|LampFactory|DeskLampFactory|FloorLampFactory", "decoration"),
    # Indoor structure as Infinigen factories (sometimes shown as
    # "WallFactory" / "FloorFactory" etc when scattered procedurally).
    (r"WallFactory|FloorFactory|CeilingFactory",          "background_wall"),
    (r"DoorFactory|WindowFactory|PanelDoorFactory",       "background_wall"),
    # Indoor furniture (real Infinigen indoor names from scene_9 inspect).
    (r"KitchenCabinetFactory",          "prop_concave_large"),
    (r"TVStandFactory",                 "prop_concave_large"),
    (r"LargePlantContainerFactory",     "prop_concave_large"),
    (r"NatureShelfTrinketsFactory",     "decoration"),
    # Infinigen indoor sub-collections that hold the room mesh parts.
    # Use unique_assets:room_floor as the canonical floor; placeholders:room_meshes
    # is a duplicate room shell mesh -> skip it to avoid double-counting.
    (r"unique_assets:room_floor",       "background_floor"),
    (r"placeholders:room_meshes|placeholders:room_shells", "skip"),
    (r"unique_assets:room_wall",        "background_wall"),
    (r"unique_assets:room_ceiling|unique_assets:room_exterior", "background_wall"),
    (r"unique_assets:doors|unique_assets:windows|unique_assets:door_casings", "background_wall"),
    (r"placeholders:portal_cutters",    "skip"),  # door/window opening helpers
    (r"^door_base_elements",            "skip"),  # door panel decorations
    (r"^skirting$",                     "decoration"),  # skirting board
    (r"^RigidBodyWorld$",               "skip"),  # Blender physics system, not a real collection
]

# Background = the scene's static structural shell. These collections
# (or named meshes) form the room or terrain you walk on, not props.
# Rules apply to **object names** because background meshes often live
# in the default `Collection`, not in a factory-prefixed collection.
BACKGROUND_RULES: list[tuple[str, str]] = [
    # Outdoor terrain
    # Reusable BoulderFactory source templates are not placed scene objects.
    # This must precede the generic BoulderFactory fallback in ASSET_RULES.
    (r"^BoulderFactory.*\.spawn_asset\(", "skip"),
    (r"OpaqueTerrain",                 "background_terrain"),
    (r"^Terrain",                      "background_terrain"),
    (r"^ShoreRockFactory",             "prop_concave_large"),
    # Hand-authored urban outdoor scenes.
    (r"^Road$|^Street$|^Sidewalk$|^PlazaGround$|^Crosswalk$|^Curb", "background_terrain"),
    # Indoor structure: Infinigen names rooms as
    # `living-room_0/0.floor`, `bedroom_0/0.wall` etc.
    (r"\.floor$",                      "background_floor"),
    (r"\.wall$",                       "background_wall"),
    (r"\.ceiling$",                    "background_wall"),
    (r"\.exterior$",                   "background_wall"),
    (r"\.meshed$",                     "skip"),  # alt-rep, duplicate of room shell
    # Generic word-based fallbacks (caught above for Infinigen, but
    # kept for hand-named scenes).
    (r"^Floor$|^floor",                "background_floor"),
    (r"^Wall$|^wall\.|_wall$|^back_wall$|^left_wall$|^right_wall$", "background_wall"),
    (r"^Ceiling|^ceiling",             "background_wall"),
    (r"^window\.|^door\.|^Window|^Door", "background_wall"),
    (r"^skirtingboard",                "decoration"),
    (r"crosswalk|lane_mark|road_mark|manhole|drain_cover|painted_arrow", "decoration"),
    (r"^Bench|^Planter|^LampPost|^Bollard|^BusStop|^Kiosk|^TrashCan|^TrafficBarrier", "prop_convex_small"),
    (r"^QueuePost|^CheckInKiosk|^ElevatorDoor|^WindowFrame|^WayfindingPillar", "prop_convex_small"),
    # Procedural industrial rooms. These rules let industrial workcells share
    # the same Module 1 registry instead of using a separate handoff pipeline.
    (r"^ceiling_slab$|^ceiling_t_grid_|^ceiling_edge_trim|^ceiling_rear_edge_trim", "background_wall"),
    (r"^ceiling_area_light_|^Camera$|^camera$", "skip"),
    (r"text|label|screen_text|floor_id|scene_identity", "decoration"),
    (r"light|diffuser|fixture", "decoration"),
    (r"floor_warning|floor_mark|floor_lane|scuff|stain|paint_chip|expansion_joint|oil_stain", "decoration"),
    (r"curtain|cable|hose|duct|pipe|tray|hanger|rail|chain", "decoration"),
    (r"^robot_", "decoration"),
    (r"welding|control_panel|gas_cylinder|fume_|shielding_|wire_feeder|torch_|spatter|bead|clamp|table_|tool_|parts_bin|storage_|rack|cabinet|console", "prop_concave_large"),
    (r"emergency_stop|push_button|selector_switch|pressure_gauge|regulator|knob|caster|foot|screw|fastener|filler_rod", "prop_convex_small"),
    # Hand-placed dynamic prop (c2w PostProcess Agent put it here).
    (r"^Bottle$|^Mug$|^Cup$|^Ball$|^coffee_cup$|^teacup$|^TrafficCone$|^PackageBox$|^PackageBoxFactory|^LuggageFactory",   "prop_dynamic"),
    # Skip rules: visual-only artifacts from Infinigen export.
    (r"^liquid|^atmosphere|^emitter|^scatter:",  "skip"),
    (r"^hoof_parent_temp",                       "skip"),  # Infinigen rigging helper
    # Indoor: keep only spawn_asset (the high-poly visible mesh).
    # spawn_placeholder / bbox_placeholder are low-poly proxies that
    # Infinigen uses internally; including them double-counts each prop.
    (r"\.spawn_placeholder\(",                   "skip"),
    (r"\.bbox_placeholder\(",                    "skip"),
    # Door panel sub-meshes (Cube/Mesh/Plane/BezierCurve in door_base_elements).
    (r"^Cube|^Mesh\.|^Plane\.|Bézier", "skip"),
    # Empty room-shell wireframes (the bare placeholder object is named
    # "living-room_0/0" with no suffix; the real meshes carry .floor etc).
    (r"^[a-z\-]+_\d+/\d+$",            "skip"),
]


def classify_collection(name: str) -> Optional[str]:
    for pat, cls in ASSET_RULES:
        if re.search(pat, name):
            return cls
    return None


def classify_object(name: str) -> Optional[str]:
    """Classify a Blender object by name.

    Step 1: try BACKGROUND_RULES (terrain / floor / wall / structural).
    Step 2: try ASSET_RULES against the object name -- this catches
            Infinigen indoor scenes where placeholder objects sit
            directly in `Collection` / `placeholders` / `unique_assets`
            rather than in a factory-prefixed sub-collection.
    """
    for pat, cls in BACKGROUND_RULES:
        if re.search(pat, name):
            return cls
    for pat, cls in ASSET_RULES:
        if re.search(pat, name):
            return cls
    return None


# Asset classes that should appear in manifest["groups"] of the export.
# Order is documentary; consumers iterate manifest["groups"].
GROUP_ORDER = [
    "background_terrain",
    "background_floor",
    "background_wall",
    "prop_concave_large",
    "prop_convex_small",
    "prop_dynamic",
    "vegetation",
    "decoration",
]

FUSED_STATIC_CLASSES = {
    "background_terrain",
    "background_floor",
    "background_wall",
    "prop_concave_large",
    "prop_convex_small",
}


# ---------------------------------------------------------------------------
# Material lookup by name: cheap heuristic for prop_dynamic objects so we
# don't burn a VLM call to identify a clearly-named bottle. Returns
# (material, density_kg_m3, friction, restitution). Used by Module 2 emit
# when there is no per-prop VLM physical_attrs entry available.
# ---------------------------------------------------------------------------
# (regex pattern, material, density, friction, restitution)
MATERIAL_RULES: list[tuple[str, str, float, float, float]] = [
    # Hand-placed indoor dynamic objects.
    (r"^Bottle$|BottleFactory",   "glass",       2500, 0.30, 0.20),
    (r"^Mug$|^Cup$|^teacup$|MugFactory|CupFactory|^coffee_cup$",
                                  "ceramic",     2400, 0.45, 0.10),
    (r"GlassFactory",             "glass",       2500, 0.30, 0.20),
    (r"BowlFactory|PlateFactory", "ceramic",     2400, 0.45, 0.10),
    (r"^Ball$|BallFactory",       "rubber",      1200, 0.85, 0.70),
    (r"AppleFactory|OrangeFactory|FruitFactory", "fruit", 950, 0.55, 0.20),
    (r"BookFactory|BookStackFactory",            "paper", 800, 0.50, 0.05),
    # Outdoor (kept for back-compat; outdoor uses VLM by default).
    (r"BoulderFactory|ShoreRockFactory", "rock_generic", 2700, 0.70, 0.10),
    # Indoor furniture defaults (used if VLM unavailable).
    (r"SofaFactory|CouchFactory", "fabric",       400, 0.70, 0.05),
    (r"TableFactory|DeskFactory|ChairFactory|BedFactory|"
     r"Cabinet|Shelf|Bookcase|Wardrobe|TVStand|MattressFactory|"
     r"BlanketFactory|BoxComforterFactory|LargePlantContainer",
                                  "wood_hardwood", 700, 0.55, 0.10),
    (r"DeskLampFactory|FloorLampFactory|MirrorFactory|WallArtFactory",
                                  "metal_steel",   2500, 0.40, 0.15),
    (r"NatureShelfTrinketsFactory","ceramic",      2400, 0.45, 0.10),
]


def material_from_name(name: str) -> tuple[str, float, float, float]:
    """Heuristic material lookup. Returns (material, density, friction, e)."""
    for pat, mat, rho, mu, e in MATERIAL_RULES:
        if re.search(pat, name):
            return mat, rho, mu, e
    return "generic", 1500.0, 0.5, 0.1
