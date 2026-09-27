"""Check installed OMPL configuration and launch loading without ROS nodes."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def test_installed_ompl_and_launch_builder(monkeypatch):
    share = Path(get_package_share_directory("mecharm_270_moveit_config"))
    config_path = share / "config/ompl_planning.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text())
    assert config["planning_plugin"] == "ompl_interface/OMPLPlanner"
    assert config["planner_configs"] == {
        "RRTConnect": {"type": "geometric::RRTConnect", "range": 0.0}}
    assert config["mecharm_arm"]["planner_configs"] == ["RRTConnect"]
    assert config["mecharm_arm"]["default_planner_config"] == "RRTConnect"
    assert "AddTimeOptimalParameterization" in config["request_adapters"]

    spec = importlib.util.spec_from_file_location(
        "e3_move_group_launch", share / "launch/move_group.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Capture the real Builder result without constructing or starting nodes.
    monkeypatch.setattr(module, "generate_move_group_launch", lambda c: c)
    loaded = module.generate_launch_description().planning_pipelines
    assert loaded["planning_pipelines"] == ["ompl"]
    assert loaded["default_planning_pipeline"] == "ompl"
    assert loaded["ompl"] == config


def test_closed_gripper_collision_exceptions_are_unique():
    share = Path(get_package_share_directory("mecharm_270_moveit_config"))
    root = ET.parse(share / "config/firefighter.srdf").getroot()
    forbidden = {"gripper_left2", "gripper_right2"}
    assert not any(
        {e.get("link1"), e.get("link2")} == forbidden
        for e in root.findall("disable_collisions"))
    for suffix in ("3",):
        pair = {f"gripper_left{suffix}", f"gripper_right{suffix}"}
        matches = [
            e for e in root.findall("disable_collisions")
            if {e.get("link1"), e.get("link2")} == pair]
        assert len(matches) == 1
        assert matches[0].get("reason") == "Default"
