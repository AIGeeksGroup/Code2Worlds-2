#!/usr/bin/env python3
"""Run the bundled Blender scene through both public method modules.

This is an offline physics smoke test. It deliberately leaves rendering and
model-based scene authoring out of the quick start; the simulator critic can
still report physical failures in its generated report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "demo_scene" / "scene.blend"
PROGRAM = ROOT / "demo_scene" / "dynamics_program.json"
GO2 = (
    ROOT
    / "benchmark/simcode4d_unified/vendor/handover9_20260921"
    / "export_singleroom_lr_go2/module2/go2_local"
)


def run(*args: str | Path, env: dict[str, str] | None = None) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "demo_scene",
        help="Directory for generated meshes, MJCF, and rollout results",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    blender = shutil.which("blender")
    if blender is None:
        parser.error("Blender is required on PATH (Blender 4.0 was used for this scene)")
    if not SCENE.is_file() or not PROGRAM.is_file() or not (GO2 / "go2.xml").is_file():
        parser.error("The bundled scene, dynamics program, or Go2 model is missing")
    output.mkdir(parents=True, exist_ok=True)

    run(blender, SCENE, "--background", "--python", ROOT / "module1/export_scene_to_obj_v2.py", "--", output)
    for name in ("module1_select_keys_v2.py", "support_graph.py", "module1_coacd.py"):
        run(sys.executable, ROOT / "module1" / name, output)
    offline = os.environ.copy()
    offline["C2W_OFFLINE_ATTRS"] = "1"
    run(sys.executable, ROOT / "module1/module1_vlm_attrs.py", output, env=offline)
    run(sys.executable, ROOT / "module1/module1_emit_mjcf_v2.py", output)

    # This run is physics-only. An inherited EGL setting may fail even on
    # machines where MuJoCo simulation itself works without OpenGL.
    os.environ.pop("MUJOCO_GL", None)
    import mujoco

    for name in ("scene_default.xml", "scene_enriched.xml"):
        mujoco.MjModel.from_xml_path(str(output / "module1" / name))

    module2 = output / "module2"
    module2.mkdir(exist_ok=True)
    shutil.copyfile(PROGRAM, module2 / "dynamics_program.json")
    run(sys.executable, ROOT / "module2/module2_emit_dynamic_v2.py", output, GO2)
    mujoco.MjModel.from_xml_path(str(module2 / "scene_m2.xml"))
    physics = os.environ.copy()
    physics["C2W_RENDER"] = "0"
    run(sys.executable, ROOT / "module2/module2_rollout_v2.py", output, env=physics)
    run(sys.executable, ROOT / "module2/postproc_rollout_blender.py", output)
    run(sys.executable, ROOT / "module2/module2_sim_critic.py", output)

    report = json.loads((module2 / "critic_report.json").read_text())
    print(f"\nModule 1 MJCF: {output / 'module1/scene_enriched.xml'}")
    print(f"Module 2 MJCF: {module2 / 'scene_m2.xml'}")
    print(f"Rollout log:   {module2 / 'rollout_log.json'}")
    print(f"Critic report: {module2 / 'critic_report.json'}")
    print(f"Critic valid:  {report.get('verdict', {}).get('valid')} (physical and visual checks)")


if __name__ == "__main__":
    main()
