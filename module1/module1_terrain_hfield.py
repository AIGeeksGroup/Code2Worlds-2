"""Build a continuous MuJoCo collision height field from the terrain mesh."""
from __future__ import annotations
import json, os, sys
import numpy as np
import trimesh
from PIL import Image

if len(sys.argv) < 2:
    raise SystemExit('usage: module1_terrain_hfield.py EXPORT_DIR [RESOLUTION]')
export_dir = os.path.abspath(sys.argv[1]); resolution = int(sys.argv[2]) if len(sys.argv) > 2 else 384
m1 = os.path.join(export_dir, 'module1')
with open(os.path.join(export_dir, 'manifest.json')) as f: manifest = json.load(f)
with open(os.path.join(m1, 'keys.json')) as f: keys = json.load(f)
terrain = manifest.get('terrain') or manifest['groups']['background_terrain'][0]
rel = terrain.get('collision_obj') or terrain['visual_obj']
mesh = trimesh.load(os.path.join(export_dir, rel), force='mesh')
(xmin,ymin,zmin),(xmax,ymax,zmax) = mesh.bounds
xs = np.linspace(xmin, xmax, resolution); ys = np.linspace(ymin, ymax, resolution)
# MuJoCo hfield rows correspond to y; sample the same top surface used by Blender terrain.
origins = np.stack([np.tile(xs, resolution), np.repeat(ys, resolution),
                    np.full(resolution*resolution, zmax + 100)], axis=1)
dirs = np.tile(np.array([[0.,0.,-1.]]), (len(origins),1))
intr = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
locs, ray_ids, _ = intr.intersects_location(origins, dirs, multiple_hits=True)
z = np.full(len(origins), zmin, dtype=np.float64)
for ri, loc in zip(ray_ids, locs): z[ri] = max(z[ri], loc[2])
heights = z.reshape(resolution, resolution)[::-1, :]
# MuJoCo accepts RGB png hfields. 8-bit is enough here: vertical quantization
# is < 11 cm over this scene and spatial sampling is 39 cm.
normalized = np.clip((heights-zmin) / max(zmax-zmin, 1e-8), 0, 1)
u8 = np.rint(normalized*255).astype(np.uint8)
elevation_rel = 'module1/terrain_hfield_elevation.txt'
with open(os.path.join(export_dir, elevation_rel), 'w') as f:
    np.savetxt(f, normalized, fmt='%.7f')
keys['terrain_hfield'] = {'elevation_rel':elevation_rel,'resolution':resolution,
 'x_min':float(xmin),'x_max':float(xmax),'y_min':float(ymin),'y_max':float(ymax),
 'z_min':float(zmin),'z_max':float(zmax),'base_depth':10.0}
with open(os.path.join(m1,'keys.json'),'w') as f: json.dump(keys,f,indent=2)
print(f'[m1-hfield] {resolution}x{resolution} -> {os.path.join(export_dir, elevation_rel)}')
print(f'[m1-hfield] xy=({xmin:.3f},{ymin:.3f})..({xmax:.3f},{ymax:.3f}) z=({zmin:.3f},{zmax:.3f})')
