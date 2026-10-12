#!/usr/bin/env python3
"""Build immutable repaired runtime artifacts without modifying legacy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


CASES = {
    "NAT-01-rock-kick": "benchmark/simcode4d_unified/assets/nat01_meshes/module2/scene_m2.xml",
    "NAT-02-hills-turn-stop": "benchmark/simcode4d_unified/source_reexports/NAT-02-hills-turn-stop-v3/module1/scene_enriched.xml",
    "NAT-03-water-land": "deliverables/final_real8s_physical_2026-09-01/module1/scene10_amphibious/simready_scene.xml",
    "SER-01-medicine": "deliverables/final_real8s_physical_2026-09-01/module1/scene_indoor_service/simready_scene.xml",
    "SER-02-airport-luggage": "deliverables/final_real8s_physical_2026-09-01/module1/indoor2_airport/simready_scene.xml",
    "SER-03-hotel-drawer": "deliverables/final_real8s_physical_2026-09-01/module1/indoor3_hotel/simready_scene.xml",
}
NAT01_SETTLED_QPOS = (26.9793591, -15.0211407, 1.10181477, 0.90872691, 0.265148019, -0.298678111, -0.121257233, 26.6556659, -15.1999169, 1.32588315, 0.999490352, -0.000366512125, -0.0319181941, 0.000360344278, 0.00292017051, 0.860353211, -1.8890557, -0.0035973969, 0.861092078, -1.88977172, 0.00386155084, 0.867468593, -1.93454207, -0.00328878759, 0.866961307, -1.93342274)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def harden_contacts(root: ET.Element) -> None:
    for default in root.findall("./default"):
        geom = default.find("geom")
        if geom is not None:
            geom.set("solref", ".002 1")
            geom.set("solimp", ".995 .9999 .0002")
            geom.set("margin", ".003")
    for geom in root.findall(".//geom"):
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue
        geom.set("solref", ".002 1")
        geom.set("solimp", ".995 .9999 .0002")
        margin = max(float(geom.get("margin", "0")), 0.003)
        geom.set("margin", f"{margin:.6g}")


def stabilize_hfield_contacts(root: ET.Element) -> None:
    for default in root.findall("./default"):
        geom = default.find("geom")
        if geom is not None:
            geom.set("solref", ".0025 1")
            geom.set("solimp", ".95 .99 .001")
            geom.set("margin", ".003")
    for geom in root.findall(".//geom"):
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue
        geom.set("solref", ".0025 1")
        geom.set("solimp", ".95 .99 .001")
        geom.set("margin", ".003")


def add_contact_exclusion(root: ET.Element, body1: str, body2: str) -> None:
    contact = root.find("./contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for exclude in contact.findall("exclude"):
        if {exclude.get("body1"), exclude.get("body2")} == {body1, body2}:
            return
    ET.SubElement(contact, "exclude", {"name": "benchmark_grasp_clearance", "body1": body1, "body2": body2})


def add_base_actuators(root: ET.Element, free_joint: str, force_limit: float, yaw_limit: float) -> None:
    actuator = root.find("./actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {child.get("name") for child in actuator}
    channels = (
        ("base_force_x_ctrl", "1 0 0 0 0 0", force_limit),
        ("base_force_y_ctrl", "0 1 0 0 0 0", force_limit),
        ("base_force_z_ctrl", "0 0 1 0 0 0", 5500.0),
        ("base_roll_ctrl", "0 0 0 1 0 0", 5500.0),
        ("base_pitch_ctrl", "0 0 0 0 1 0", 5500.0),
        ("base_yaw_ctrl", "0 0 0 0 0 1", yaw_limit),
    )
    for name, gear, limit in channels:
        if name in existing:
            continue
        ET.SubElement(actuator, "motor", {"name": name, "joint": free_joint, "gear": gear,
                                           "ctrllimited": "true", "ctrlrange": f"{-limit:g} {limit:g}"})


def add_joint_limit_buffer(root: ET.Element, joint_names: tuple[str, ...]) -> None:
    for name in joint_names:
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None or not joint.get("range"):
            continue
        lower, upper = (float(value) for value in joint.get("range").split())
        if abs(lower) <= 1e-12:
            joint.set("range", f"-.01 {upper:g}")


def patch_nat01(root: ET.Element, destination_dir: Path) -> None:
    key = root.find("./keyframe/key[@name='m2_init']")
    if key is None:
        raise RuntimeError("NAT-01 m2_init key is missing")
    source_values = key.get("qpos", "").split()
    if len(source_values) != len(NAT01_SETTLED_QPOS):
        raise RuntimeError("NAT-01 qpos layout changed; settled keyframe must be regenerated")
    key.set("qpos", " ".join(f"{value:.9g}" for value in NAT01_SETTLED_QPOS))
    include = root.find("./include")
    if include is not None:
        source = Path(include.get("file", "")).resolve()
        robot_tree = ET.parse(source)
        harden_contacts(robot_tree.getroot())
        runtime_include = destination_dir / "go2_runtime.xml"
        ET.indent(robot_tree, space="  ")
        robot_tree.write(runtime_include, encoding="unicode")
        include.set("file", str(runtime_include.resolve()))


def patch_nat02(root: ET.Element, workspace: Path, destination_dir: Path) -> None:
    stabilize_hfield_contacts(root)
    source = (workspace / "jackal_ugv/jackal.xml").resolve()
    jackal_tree = ET.parse(source)
    jackal_root = jackal_tree.getroot()
    compiler = jackal_root.find("./compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for joint in jackal_root.findall(".//joint"):
        name = joint.get("name", "")
        if name.startswith("jackal_") and name.endswith("_joint"):
            joint.set("damping", "2")
            joint.set("armature", ".08")
    wheel_suspensions = {
        "jackal_front_left_wheel": "jackal_front_left_suspension",
        "jackal_front_right_wheel": "jackal_front_right_suspension",
        "jackal_rear_left_wheel": "jackal_rear_left_suspension",
        "jackal_rear_right_wheel": "jackal_rear_right_suspension",
    }
    for body in jackal_root.findall(".//body"):
        suspension_name = wheel_suspensions.get(body.get("name", ""))
        if suspension_name:
            suspension = ET.Element(
                "joint",
                {"name": suspension_name, "type": "slide", "axis": "0 0 1",
                 "limited": "true", "range": "-.08 .18", "stiffness": "30000",
                 "damping": "2000", "springref": "0"},
            )
            body.insert(0, suspension)
    wheel_collisions = {
        "jackal_front_left_collision", "jackal_front_right_collision",
        "jackal_rear_left_collision", "jackal_rear_right_collision",
    }
    for geom in jackal_root.findall(".//geom"):
        if geom.get("name") in wheel_collisions:
            geom.set("type", "sphere")
            geom.set("size", ".3969")
            geom.set("friction", ".8 .02 .001")
            geom.attrib.pop("euler", None)
            geom.attrib.pop("quat", None)
    for mesh in jackal_root.findall("./asset/mesh"):
        raw = mesh.get("file")
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = source.parent / "assets" / candidate
        mesh.set("file", str(candidate.resolve()))
    harden_contacts(jackal_root)
    stabilize_hfield_contacts(jackal_root)
    add_base_actuators(jackal_root, "jackal_freejoint", 5000.0, 650.0)
    runtime_include = destination_dir / "jackal_runtime.xml"
    ET.indent(jackal_tree, space="  ")
    jackal_tree.write(runtime_include, encoding="unicode")
    if not any(Path(item.get("file", "")).name == runtime_include.name for item in root.findall("./include")):
        ET.SubElement(root, "include", {"file": str(runtime_include.resolve())})


def patch_nat03(root: ET.Element) -> None:
    shore = root.find("./worldbody/geom[@name='shore_hfield']")
    if shore is not None:
        shore.set("margin", "0")
    for suffix in ("fl", "fr", "rl", "rr"):
        body = root.find(f".//body[@name='wheel_{suffix}']")
        if body is None:
            raise RuntimeError(f"NAT-03 wheel_{suffix} body is missing")
        suspension = ET.Element(
            "joint",
            {"name": f"wheel_{suffix}_suspension", "type": "slide", "axis": "0 0 1",
             "limited": "true", "range": "-.06 .12", "stiffness": "8000",
             "damping": "700", "springref": "0"},
        )
        body.insert(0, suspension)
        wheel = body.find(f"./geom[@name='wheel_{suffix}_geom']")
        if wheel is None:
            raise RuntimeError(f"NAT-03 wheel_{suffix} collision geom is missing")
        visual_attributes = dict(wheel.attrib)
        visual_attributes.update({"name": f"wheel_{suffix}_visual", "mass": "0",
                                  "contype": "0", "conaffinity": "0"})
        body.append(ET.Element("geom", visual_attributes))
        wheel.set("type", "sphere")
        wheel.set("size", ".27")
        wheel.set("group", "3")
        wheel.set("rgba", "0 0 0 0")
        wheel.set("margin", "0")
        wheel.attrib.pop("quat", None)
    actuator = root.find("./actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {child.get("name") for child in actuator}
    additions = (
        ("propeller_x_ctrl", "shore_x", "-2800 2800"),
        ("heave_ctrl", "shore_z", "-1300 1300"),
        ("pitch_ctrl", "shore_pitch", "-900 900"),
    )
    for name, joint, force_range in additions:
        if name in existing:
            continue
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": joint,
                "gear": "1",
                "forcelimited": "true",
                "forcerange": force_range,
                "ctrllimited": "true",
                "ctrlrange": force_range,
            },
        )


def patch_medicine(root: ET.Element) -> None:
    for joint_name, actuator_name in (("lift_joint", "lift_ctrl"), ("arm_joint", "arm_ctrl")):
        joint = root.find(f".//joint[@name='{joint_name}']")
        actuator = root.find(f"./actuator/position[@name='{actuator_name}']")
        if joint is None or actuator is None:
            raise RuntimeError("SER-01 axis or actuator is missing: " + joint_name)
        joint.set("damping", "140")
        actuator.set("kp", "6000")


def patch_airport(root: ET.Element) -> None:
    joint = root.find(".//joint[@name='lift_joint']")
    actuator = root.find("./actuator/position[@name='lift_ctrl']")
    if joint is None or actuator is None:
        raise RuntimeError("SER-02 lift axis or actuator is missing")
    joint.set("damping", "140")
    actuator.set("kp", "6000")


def patch_hotel(root: ET.Element) -> None:
    equality = root.find("./equality/joint[@name='drawer_lock']")
    if equality is not None:
        equality.set("active", "false")
    joint = root.find(".//joint[@name='drawer_joint']")
    if joint is not None:
        joint.set("damping", "120")
        joint.set("frictionloss", "8")
        joint.set("stiffness", "0")
        joint.set("springref", "0")
        joint.set("range", "0 .31")
    actuator = root.find("./actuator/position[@name='drawer_ctrl']")
    if actuator is not None:
        actuator.set("kp", "3000")


def build(workspace: Path, output_root: Path) -> None:
    for case_id, relative_source in CASES.items():
        source = workspace / relative_source
        destination_dir = output_root / case_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "m1_world.xml"
        tree = ET.parse(source)
        root = tree.getroot()
        for include in root.findall(".//include"):
            include_path = Path(include.get("file", ""))
            if not include_path.is_absolute():
                include.set("file", str((source.parent / include_path).resolve()))
        option = root.find("./option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", "0.001")
        option.set("iterations", "120")
        option.set("tolerance", "1e-10")
        harden_contacts(root)
        if case_id == "NAT-01-rock-kick":
            patch_nat01(root, destination_dir)
        elif case_id == "NAT-02-hills-turn-stop":
            patch_nat02(root, workspace, destination_dir)
        elif case_id == "NAT-03-water-land":
            patch_nat03(root)
        elif case_id == "SER-01-medicine":
            patch_medicine(root)
        elif case_id == "SER-02-airport-luggage":
            patch_airport(root)
        elif case_id == "SER-03-hotel-drawer":
            patch_hotel(root)
        if case_id.startswith("SER-"):
            add_joint_limit_buffer(root, ("lift_joint", "arm_joint", "drawer_joint"))
            add_base_actuators(root, "root_free", 900.0, 520.0)
        if case_id in {"SER-01-medicine", "SER-02-airport-luggage"}:
            add_contact_exclusion(root, "task_item", "grasp_frame")
        ET.indent(tree, space="  ")
        tree.write(destination, encoding="unicode")
        overlay = {
            "schema_version": "simcode4d.rc3a-dev1.m2-overlay",
            "case_id": case_id,
            "m1_base_sha256": sha256_file(destination),
            "task_independent": True,
            "kind": "noop",
            "allowed_dynamic_channels": [],
            "note": "No source-declared time-varying M2 component is required. Static contacts and fields remain M1-owned.",
        }
        with (destination_dir / "m2_overlay.json").open("w", encoding="utf-8") as handle:
            json.dump(overlay, handle, indent=2, sort_keys=True)
        with (destination_dir / "legacy_origin.txt").open("w", encoding="utf-8") as handle:
            handle.write(relative_source + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.workspace.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
