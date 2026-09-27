"""Offline checks for recorded bin calibration and static platforms."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import pytest
import yaml

from e3_sorting_system.pick_action_server import (
    APPROACH_JOINT_LIMITS, calibration_slots)

PACKAGE = Path(__file__).resolve().parents[1]
EXPECTED = {
    "red_bin": [1.5780840510, 0.8013771349, -0.9467428297,
                -0.0332285864, 1.7100726856, 1.6302872993],
    "blue_bin": [-1.5779212067, 0.8827475163, -1.0850132822,
                 0.0335590192, 1.7794357656, -1.5141581248],
}


RELEASE_JOINTS = {
    "red_bin.left": [1.7858465337, 0.8430637514, -0.7403920297,
                     -0.0305444414, 1.4737572650, 1.8456349341],
    "red_bin.center": [1.5860181203, 0.8380505628, -0.7422356759,
                       -0.0370827207, 1.4941800225, 1.6558392442],
    "red_bin.right": [1.3685759249, 0.8158492750, -0.6887449752,
                      -0.0423251675, 1.4320815049, 1.4299657847],
    "blue_bin.left": [-1.7954059604, 0.8717468305, -0.8018801372,
                      0.0282933348, 1.4989574536, -1.7356545103],
    "blue_bin.center": [-1.5786027935, 0.8032457336, -0.6907313796,
                        0.0336656462, 1.4543002999, -1.5214733686],
    "blue_bin.right": [-1.3668479619, 0.8719069661, -0.8194452148,
                       0.0267157095, 1.5002830670, -1.2996634482],
}


PICK_JOINTS = {
    "grid_1.pick_joints": [
        0.724953, 0.819108, -0.589521,
        -0.000043, 1.337840, 0.734401],
    "grid_2.pick_joints": [
        0.6010395155, 1.1610209601, -1.2097142381,
        0.0135643530, 1.5434936893, 0.6239476983],
    "grid_3.pick_joints": [
        0.0299717092, 0.4072309339, 0.0607839625,
        -0.0485094300, 1.1382044377, 0.0878735103],
    "grid_4.pick_joints": [
        -0.0108474300, 0.8006787379, -0.5972599467,
        0.0083696301, 1.3114291338, 0.0472812917],
    "grid_5.pick_joints": [
        -0.7690627850, 0.8152957491, -0.5632076370,
        0.0456492524, 1.3115248216, -0.7278116679],
    "grid_6.pick_joints": [
        -0.6028831361, 1.1625077805, -1.2096582565,
        -0.0172381553, 1.5396605038, -0.6220880451],
}


TRANSPORT_JOINTS = {
    "grid_1.pick_joints": [
        0.724953, 0.819108, -0.589521,
        -0.000043, 1.337840, 0.734401],
    "grid_1.lift_stage_1_joints": [
        0.729194, 0.808804, -0.741697,
        -0.004483, 1.508453, 0.737744],
    "grid_1.clearance_joints": [
        0.7002439295, 0.9082026653, -1.2221155915,
        -0.0161130230, 1.8575642789, 0.7573696093],
    "red_bin.approach_joints": EXPECTED["red_bin"],
    "red_bin.center_intermediate_joints": [
        1.5815950842, 0.8196593210, -0.8260416017,
        -0.0362723402, 1.5886831424, 1.6553670456],
    "red_bin.center_release_joints": RELEASE_JOINTS["red_bin.center"],
}


@pytest.mark.parametrize("name,y,color", [
    ("red_bin", 0.18, [0.90, 0.08, 0.08, 1.0]),
    ("blue_bin", -0.18, [0.05, 0.20, 0.90, 1.0])])
def test_static_receiving_platform(name, y, color):
    world = ET.parse(PACKAGE / "worlds/e3_sorting.world")
    model = world.find(f"./world/model[@name='{name}']")
    assert model.findtext("static") == "true"
    assert list(map(float, model.findtext("pose").split())) == (
        [0.0, y, 0.406, 0.0, 0.0, 0.0])
    link, = model.findall("link")
    assert link.get("name") == "region_link"
    for tag in ("visual", "collision"):
        element, = link.findall(tag)
        size = list(map(float, element.findtext("geometry/box/size").split()))
        assert size == (
            [0.18, 0.10, 0.012])
    for tag in ("ambient", "diffuse"):
        assert list(map(float, link.findtext(
            "visual/material/" + tag).split())) == color
    assert model.findall(".//gravity") == []
    assert model.findall(".//inertial") == []


def test_recorded_joint_data_and_safety_defaults():
    source = PACKAGE / "config/sorting_config.yaml"
    config = yaml.safe_load(source.read_text())
    cfg = config["e3_pick_action_server"]["ros__parameters"]
    for name, expected in EXPECTED.items():
        values = cfg[name + ".approach_joints"]
        assert values == expected
        assert len(values) == 6
        for joint, value in zip(cfg["joint_names"], values):
            assert math.isfinite(value)
            lower, upper = APPROACH_JOINT_LIMITS[joint]
            assert lower <= value <= upper
    for name, expected in RELEASE_JOINTS.items():
        model, slot = name.split(".")
        values = cfg[f"{model}.{slot}_release_joints"]
        assert values == expected
        assert len(values) == 6
        for joint, value in zip(cfg["joint_names"], values):
            assert math.isfinite(value)
            lower, upper = APPROACH_JOINT_LIMITS[joint]
            assert lower <= value <= upper
    assert len(RELEASE_JOINTS) == 6
    red, blue = (cfg[name + ".approach_joints"] for name in EXPECTED)
    assert red is not blue
    assert red != blue
    blue_copy = list(blue)
    red[0] = 0.0
    assert blue == blue_copy
    assert cfg["attach_enabled"] is False
    assert cfg["lift_calibration"] is False


def test_transport_path_is_exact_finite_and_within_joint_limits():
    cfg = slot_config()
    assert cfg["classification_transport_enabled"] is False
    for name, expected in TRANSPORT_JOINTS.items():
        values = cfg[name]
        assert values == expected
        assert len(values) == 6
        for joint, value in zip(cfg["joint_names"], values):
            assert math.isfinite(value)
            lower, upper = APPROACH_JOINT_LIMITS[joint]
            assert lower <= value <= upper


def test_calibration_limits_match_shared_urdf():
    source = (PACKAGE.parents[3] / "E5_fixed_pick_place/ros2_ws/src"
              / "e5_arm_sim/urdf/mecharm_270_m5_sim.urdf.xacro")
    root = ET.parse(source)
    for name, bounds in APPROACH_JOINT_LIMITS.items():
        limit = root.find(f"./joint[@name='{name}']/limit")
        assert (float(limit.get("lower")), float(limit.get("upper"))) == bounds


def slot_config():
    source = PACKAGE / "config/sorting_config.yaml"
    return yaml.safe_load(source.read_text())[
        "e3_pick_action_server"]["ros__parameters"]


def test_independent_ordered_slots_and_full_footprints():
    cfg = slot_config()
    slots = calibration_slots(cfg)
    for model, y in (("red_bin", 0.18), ("blue_bin", -0.18)):
        assert slots[model] == [[-0.040, y], [0.0, y], [0.040, y]]
        assert cfg[model + ".slot_release_z"] == 0.438
        assert len({id(slot) for slot in slots[model]}) == 3
        for a, b in zip(slots[model], slots[model][1:]):
            assert b[0] - a[0] == pytest.approx(0.040)
            assert b[0] - a[0] - 0.028 == pytest.approx(0.012)
        for x, _ in slots[model]:
            assert 0.09 - abs(x) - 0.014 > 0
            assert 0.05 - 0.014 > 0
        for x in (-0.040, 0.040):
            assert 0.09 - abs(x) - 0.014 == pytest.approx(0.036)
    slots["red_bin"][0][0] = 99.0
    assert slots["red_bin"][1][0] == 0.0
    assert slots["blue_bin"][0][0] == -0.040
    assert sorted(key for key in cfg if "approach_joints" in key) == [
        "blue_bin.approach_joints", "red_bin.approach_joints"]
    release_keys = {
        key for key in cfg if key.endswith("_release_joints")}
    assert release_keys == {
        f"{model}.{slot}_release_joints"
        for model in ("red_bin", "blue_bin")
        for slot in ("left", "center", "right")}
    assert cfg["bin_slot_calibration"] is False
    assert cfg["calibration_execute"] is False


@pytest.mark.parametrize("values", [
    [], [-0.055, 0.0], [-0.055, 0.0, 0.055, 0.06],
    [float("nan"), 0.0, 0.055], [-0.055, 0.0, float("inf")],
    [0.055, 0.0, -0.055], [-0.055, 0.0, 0.0],
    [-0.028, 0.0, 0.055], [-0.055, 0.0, 0.08]])
@pytest.mark.parametrize("model", ["red_bin", "blue_bin"])
def test_invalid_slot_offsets_rejected(model, values):
    cfg = slot_config()
    cfg[f"{model}.slot_offsets_x"] = values
    with pytest.raises(ValueError):
        calibration_slots(cfg)


@pytest.mark.parametrize("values", [[0.0], [0.0, float("nan")],
                                    [float("inf"), 0.18]])
def test_invalid_slot_world_coordinates_rejected(values):
    cfg = slot_config()
    cfg["red_bin.center_world_xy"] = values
    with pytest.raises(ValueError):
        calibration_slots(cfg)


@pytest.mark.parametrize("suffix", ["slot_offsets_x", "center_world_xy"])
def test_shared_mutable_slot_configuration_rejected(suffix):
    cfg = slot_config()
    cfg[f"blue_bin.{suffix}"] = cfg[f"red_bin.{suffix}"]
    with pytest.raises(ValueError, match="independent"):
        calibration_slots(cfg)


def test_all_grid_pick_joints_and_world_positions_are_persisted():
    cfg = slot_config()
    assert len(PICK_JOINTS) == 6
    for name, expected in PICK_JOINTS.items():
        values = cfg[name]
        assert values == expected
        assert len(values) == 6
        for joint, value in zip(cfg["joint_names"], values):
            assert math.isfinite(value)
            lower, upper = APPROACH_JOINT_LIMITS[joint]
            assert lower <= value <= upper

    expected_poses = {
        "pick_object_1": [0.13, 0.12, 0.4255, 0.0, 0.0, 0.0],
        "pick_object_2": [0.17, 0.12, 0.4255, 0.0, 0.0, 0.0],
        "pick_object_3": [0.13, 0.00, 0.4255, 0.0, 0.0, 0.0],
        "pick_object_4": [0.18, 0.00, 0.4255, 0.0, 0.0, 0.0],
        "pick_object_5": [0.13, -0.12, 0.4255, 0.0, 0.0, 0.0],
        "pick_object_6": [0.17, -0.12, 0.4255, 0.0, 0.0, 0.0],
    }
    world = ET.parse(PACKAGE / "worlds/e3_sorting.world")
    for model_name, expected in expected_poses.items():
        model = world.find(f"./world/model[@name='{model_name}']")
        assert model is not None
        assert list(map(float, model.findtext("pose").split())) == expected
        for tag in ("visual", "collision"):
            size = model.findtext(f"link/{tag}/geometry/box/size")
            assert list(map(float, size.split())) == [0.028, 0.028, 0.045]
