"""Build the small Blender input used by the Module 1/2 quick start.

Run from the repository root:
    blender --background --python demo_scene/build_scene.py -- demo_scene/scene.blend
"""
from __future__ import annotations

import os
import sys

import bpy
from mathutils import Vector


def material(name: str, color: tuple[float, float, float, float]):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    return value


def apply_scale(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

floor_mat = material("concrete_floor", (0.57, 0.61, 0.64, 1.0))
cabinet_mat = material("painted_wood", (0.55, 0.35, 0.20, 1.0))
ball_mat = material("rubber_ball", (0.85, 0.25, 0.12, 1.0))

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.05))
floor = bpy.context.object
floor.name = "Floor"
floor.scale = (4.0, 4.0, 0.1)
apply_scale(floor)
floor.data.materials.append(floor_mat)

bpy.ops.mesh.primitive_cube_add(size=1, location=(1.05, 0.45, 0.30))
cabinet = bpy.context.object
cabinet.name = "Cabinet_demo"
cabinet.scale = (0.60, 0.50, 0.60)
apply_scale(cabinet)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=3)
bpy.ops.object.mode_set(mode="OBJECT")
cabinet.data.materials.append(cabinet_mat)

bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.12, location=(0, 0, 0.12))
ball = bpy.context.object
ball.name = "Ball"
ball.data.materials.append(ball_mat)

bpy.ops.object.camera_add(location=(2.8, -3.8, 2.6))
camera = bpy.context.object
direction = Vector((0, 0, 0.2)) - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = camera

bpy.ops.object.light_add(type="AREA", location=(0, 0, 3.4))
bpy.context.object.data.energy = 450
bpy.context.object.data.shape = "DISK"
bpy.context.object.data.size = 4

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
target = os.path.abspath(args[0] if args else "demo_scene/scene.blend")
os.makedirs(os.path.dirname(target), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=target)
print(f"demo scene saved: {target}")
