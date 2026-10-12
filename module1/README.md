# Module 1: SimReady Scene Compiler

Module 1 compiles a Code2Worlds Blender scene into MuJoCo MJCF without changing the original object and scene generation streams. The compiler exports scene meshes and a manifest, selects key interaction objects, makes CoACD collision pieces, estimates material properties from rendered views or a local registry, constructs support-aware scene geometry, and emits paired default and enriched MJCF worlds.

## Inputs and outputs

- Input: a `.blend` authoring scene with recognizable object and collection names.
- Intermediate files: `<export>/manifest.json`, meshes, key-object selection, collision pieces, material estimates, optional terrain cells.
- Output: `<export>/module1/scene_default.xml` and `scene_enriched.xml`.

## Compile a scene

The repository includes an editable [Blender example](../demo_scene/scene.blend). From the repository root, after installing `requirements-method.txt` and Blender, run the checked-in example through both modules:

```bash
python demo_scene/run_demo.py --output /tmp/code2worlds2-demo
```

Inspect `/tmp/code2worlds2-demo/module1/scene_default.xml`, `scene_enriched.xml`, `keys.json`, and `support_graph.json`. See the [example guide](../demo_scene/README.md) for individual steps. For your own Blender scene, the Module 1 sequence is:

```bash
blender /path/to/scene.blend --background \
  --python module1/export_scene_to_obj_v2.py -- ./scene_export
python module1/module1_select_keys_v2.py ./scene_export
python module1/support_graph.py ./scene_export
python module1/module1_coacd.py ./scene_export
C2W_OFFLINE_ATTRS=1 python module1/module1_vlm_attrs.py ./scene_export
# For an outdoor terrain scene, also run:
# python module1/module1_terrain_grid.py ./scene_export
python module1/module1_emit_mjcf_v2.py ./scene_export
```

`support_graph.py` writes an auditable geometric support proposal. It labels
support, floating/embedded, containment, attachment, and likely hinge
relations from exported AABBs. Rotated thin objects and variable-height
terrain need simulator contact confirmation.

For an MJCF with free-joint objects, run a separate gravity pass:

```bash
python module1/settle_scene.py ./scene_export/module2/scene_m2.xml \
  --body kick_stone_0 --duration-s 5
```

This creates a settled MJCF beside its input and reports whether the selected
body reached low speed with contact. Set `C2W_M2_SCENE_XML` to the settled file
for a subsequent Go2 rollout. M1's paired exports are static; Module 2 adds
dynamic free bodies.

To request VLM physical attributes, run `module1/module1_vlm_render.py` first, set `DASHSCOPE_API_KEY`, and call `module1/module1_vlm_attrs.py` without `C2W_OFFLINE_ATTRS=1`. No key is stored in this repository. The terrain grid stage is used only for outdoor scenes. Compare the two MJCF exports to inspect the effect of enriched collision and physical attributes. Both output files rely on the mesh files in the export directory, so keep the directory intact.

The public repository includes this small editable example and the frozen benchmark runtime assets. It does not include the multi-gigabyte Blender authoring scenes used in the paper. Run this pipeline on the example, your own generated scene, or a separately obtained source scene.
