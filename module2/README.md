# Module 2: Simulator-in-the-Loop 4D Refiner

Module 2 turns a dynamics program into a MuJoCo scene, executes a robot/object rollout, records physical state and contact signals, and produces a simulator-critic report. Visual effects remain in Blender. The core Go2 pipeline supports outdoor object spawning and indoor reuse of an exported dynamic prop. Additional executors cover Jackal and UR5e; the frozen NAT-03 amphibious task is evaluated by SimCode4D.

## Inputs and outputs

- Input: a Module 1 export directory, `dynamics_program.json`, and a Go2 model directory for the core Go2 path. The bundled example uses the Go2 model included with the benchmark.
- Output: `<export>/module2/scene_m2.xml`, `rollout_log.json`, `rollout_log_blender.json`, and `critic_report.json`. The rollout may also produce an MP4 and preview frames.

## Run the core path

For an offline physics run using the included `.blend` scene, program, and Go2 model:

```bash
python demo_scene/run_demo.py --output /tmp/code2worlds2-demo
```

This runs Module 1 first and writes `/tmp/code2worlds2-demo/module2/scene_m2.xml`, `rollout_log.json`, and `critic_report.json`. See the [example guide](../demo_scene/README.md) for each command. For a custom scene after Module 1, run:

```bash
export DASHSCOPE_API_KEY=your_key
python module2/module2_dyn_program.py ./scene_export "A robot pushes an object"
python module2/module2_emit_dynamic_v2.py ./scene_export /path/to/unitree_go2
MUJOCO_GL=egl python module2/module2_rollout_v2.py ./scene_export
python module2/postproc_rollout_blender.py ./scene_export
python module2/module2_sim_critic.py ./scene_export
```

You may provide `<export>/module2/dynamics_program.json` yourself and skip the LLM generation step. The simulator critic can compute physics signals without an API key, but a missing visual verdict cannot make fused acceptance pass. Its seven-signal implementation targets the Go2/object interaction path. For headless hosts without EGL, try `MUJOCO_GL=osmesa` if OSMesa is installed.

For a physics-only smoke test on a host without usable OpenGL, set
`C2W_RENDER=0` when running the rollout. It still writes the state/contact
log; no MP4 or fused visual acceptance is produced. For video on memory-limited
hosts, set `C2W_RENDER_WIDTH` and `C2W_RENDER_HEIGHT` (for example 640 and
360). Run `postproc_rollout_blender.py` as a separate command after rollout so
large MuJoCo models are not resident twice.

To run the bounded automatic loop on that core path:

```bash
MUJOCO_GL=egl python module2/run_refinement.py \
  ./scene_export /path/to/unitree_go2 --max-revisions 3
```

Each round archives the program, rollout log, and critic verdict under
`<export>/module2/refinement/`. A model key is required to revise a rejected
plan. Revisions cannot change object identity, source geometry, robot action,
or visual effects. Jackal and UR5e execution paths do not use
this Go2-specific repair loop.

For Blender rerendering, use the corrected `rollout_log_blender.json`, which accounts for MuJoCo mesh transforms:

```bash
blender /path/to/scene.blend --background \
  --python module2/module2_render_blender_v2.py -- \
  ./scene_export/module2/rollout_log_blender.json \
  /path/to/unitree_go2 ./scene_export ./scene_export/module2/frames 75
```

The custom-scene pipeline needs a source scene and a compatible robot asset; the example supplies both. For a standalone test of released benchmark contracts and metrics, use the [SimCode4D benchmark](../benchmark/simcode4d_unified/README.md).
