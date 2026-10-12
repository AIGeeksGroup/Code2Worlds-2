#!/usr/bin/env python3
"""Run a gravity settling pass and write accepted free-body poses to MJCF.

Use this on the compiled scene before task rollout. The output stays beside the
input XML so local MJCF asset references keep the same base directory.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def free_joint(model: mujoco.MjModel, body_name: str) -> int:
    body = model.body(body_name).id
    for joint in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            return joint
    raise ValueError(body_name + " has no free joint")


def descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def settle(xml_path: Path, output_xml: Path, bodies: list[str], duration_s: float,
           prelift_m: float) -> dict:
    xml_path, output_xml = xml_path.resolve(), output_xml.resolve()
    if xml_path.parent != output_xml.parent:
        raise ValueError("settled XML must be written beside its input XML")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("m2_init").id)
    except KeyError:
        mujoco.mj_resetData(model, data)
    if not bodies:
        bodies = [model.body(int(model.jnt_bodyid[j])).name for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not bodies:
        raise ValueError("scene has no free-joint bodies to settle")
    selected = {name: free_joint(model, name) for name in bodies}
    for joint in selected.values():
        data.qpos[int(model.jnt_qposadr[joint]) + 2] += prelift_m
    mujoco.mj_forward(model, data)
    steps = int(math.ceil(duration_s / model.opt.timestep))
    contact_seen = {name: False for name in bodies}
    final_penetration_m = {name: 0.0 for name in bodies}
    terminal_steps = max(1, int(math.ceil(0.05 / model.opt.timestep)))
    for step in range(steps):
        mujoco.mj_step(model, data)
        for contact in data.contact:
            left = int(model.geom_bodyid[contact.geom1]);right = int(model.geom_bodyid[contact.geom2])
            for name in bodies:
                target = model.body(name).id
                if step >= steps - terminal_steps and (descendant(model, left, target) or descendant(model, right, target)):
                    contact_seen[name] = True
                    final_penetration_m[name] = max(final_penetration_m[name], max(0.0, -float(contact.dist)))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("non-finite state during settling")
    root = ET.parse(xml_path)
    results = []
    for name, joint in selected.items():
        qaddr = int(model.jnt_qposadr[joint]);vaddr = int(model.jnt_dofadr[joint])
        pose = data.qpos[qaddr:qaddr + 7].tolist()
        linear = float(np.linalg.norm(data.qvel[vaddr:vaddr + 3]))
        angular = float(np.linalg.norm(data.qvel[vaddr + 3:vaddr + 6]))
        accepted = bool(linear <= 0.05 and angular <= 0.5 and contact_seen[name]
                        and final_penetration_m[name] <= 0.001)
        results.append({"body": name, "accepted": accepted, "position": pose[:3],
                        "quaternion": pose[3:], "linear_speed_mps": linear,
                        "angular_speed_radps": angular,
                        "contact_in_final_50ms": contact_seen[name],
                        "final_window_max_penetration_m": final_penetration_m[name]})
        if not accepted:
            continue
        body_element = root.find(".//body[@name='" + name + "']")
        if body_element is None:
            raise ValueError(name + " body is not in the top-level XML; flatten includes first")
        body_element.set("pos", " ".join(f"{value:.9g}" for value in pose[:3]))
        body_element.set("quat", " ".join(f"{value:.9g}" for value in pose[3:]))
        for key in root.findall(".//keyframe/key"):
            if key.get("qpos"):
                values = [float(value) for value in key.get("qpos").split()]
                if len(values) == model.nq:
                    values[qaddr:qaddr + 7] = pose
                    key.set("qpos", " ".join(f"{value:.9g}" for value in values))
            if key.get("qvel"):
                values = [float(value) for value in key.get("qvel").split()]
                if len(values) == model.nv:
                    values[vaddr:vaddr + 6] = [0.0] * 6
                    key.set("qvel", " ".join(f"{value:.9g}" for value in values))
    root.write(output_xml, encoding="unicode", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output_xml))
    return {"schema_version": "code2worlds2.settling.v1", "input_xml": str(xml_path),
            "settled_xml": str(output_xml), "duration_s": duration_s,
            "prelift_m": prelift_m, "bodies": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--body", action="append", default=[])
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--prelift-m", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration_s <= 0 or not 0 <= args.prelift_m <= 0.1:
        parser.error("duration must be positive and prelift must be in [0,0.1] m")
    xml = args.xml.resolve()
    output = args.output or xml.with_name(xml.stem + "_settled.xml")
    report = settle(xml, output, args.body, args.duration_s, args.prelift_m)
    report_path = output.with_suffix(".settling.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"settled_xml": str(output), "accepted": sum(item["accepted"] for item in report["bodies"]),
                      "body_count": len(report["bodies"])}))
    return 0 if all(item["accepted"] for item in report["bodies"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
