# Editable Module 1/2 example

[`scene.blend`](scene.blend) contains a floor, cabinet, and ball. [`build_scene.py`](build_scene.py) generates the checked-in Blender file and can be edited to make variants. [`dynamics_program.json`](dynamics_program.json) tells Module 2 to reuse the exported ball as a free rigid body and run a Go2 kick action. This example exercises the public compiler and physics rollout without a model API or the paper's large authoring scenes.

![The bundled Blender example scene](preview.png)

From the repository root, install Blender 4.0 or newer and `requirements-method.txt`, then run:

```bash
python demo_scene/run_demo.py --output /tmp/code2worlds2-demo
```

The runner saves all generated files outside the checkout, validates the default, enriched, and dynamic MJCF models with MuJoCo, and prints paths to the rollout log and critic report. It uses the Go2 model already included under `benchmark/simcode4d_unified/vendor/`. It turns video rendering off so it can run without a graphics backend. The critic may report physical failures and always lacks visual acceptance in this mode; the report is feedback for refining a scene, not a benchmark success claim.

## Run the stages individually

The following commands are equivalent to the runner. Use them to inspect or edit intermediate results:

```bash
OUT=/tmp/code2worlds2-demo
GO2=benchmark/simcode4d_unified/vendor/handover9_20260921/export_singleroom_lr_go2/module2/go2_local

# Module 1: Blender scene -> exported meshes -> SimReady MJCF.
blender demo_scene/scene.blend --background \
  --python module1/export_scene_to_obj_v2.py -- "$OUT"
python module1/module1_select_keys_v2.py "$OUT"
python module1/support_graph.py "$OUT"
python module1/module1_coacd.py "$OUT"
C2W_OFFLINE_ATTRS=1 python module1/module1_vlm_attrs.py "$OUT"
python module1/module1_emit_mjcf_v2.py "$OUT"

# Module 2: supplied program -> dynamic MJCF -> physics log and critic.
mkdir -p "$OUT/module2"
cp demo_scene/dynamics_program.json "$OUT/module2/dynamics_program.json"
python module2/module2_emit_dynamic_v2.py "$OUT" "$GO2"
C2W_RENDER=0 python module2/module2_rollout_v2.py "$OUT"
python module2/postproc_rollout_blender.py "$OUT"
python module2/module2_sim_critic.py "$OUT"
```

Module 1 writes `module1/scene_default.xml` and `module1/scene_enriched.xml`; Module 2 writes `module2/scene_m2.xml`, `rollout_log.json`, and `critic_report.json` under `$OUT`. The offline material lookup replaces VLM attribute prediction for this quick start. The checked-in dynamics program replaces LLM program generation. Run with a configured model provider and visual rendering to exercise those optional stages.

## Make it your own

Open `scene.blend` in Blender, or edit `build_scene.py` and regenerate a copy with:

```bash
blender --background --python demo_scene/build_scene.py -- /tmp/my_scene.blend
```

For a new scene, run the Module 1 commands with your `.blend` path. Inspect `<export>/manifest.json` for the `prop_dynamic` name and set `reuse_existing` in a copied dynamics program to that exact name before running Module 2. For outdoor scenes, run `module1/module1_terrain_grid.py` before MJCF emission. The runner itself always uses the checked-in scene and program; use the individual commands for customized scenes.
