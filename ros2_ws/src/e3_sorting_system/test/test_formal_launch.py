"""Static launch checks; do not execute a LaunchService."""

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_formal_launch_configuration():
    source = (ROOT / "launch/e3_sorting.launch.py").read_text()
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "Node"]
    assert len(nodes) == 3
    names = []
    for node in nodes:
        args = {k.arg: k.value for k in node.keywords}
        names.append(ast.literal_eval(args["name"]))
        params = args["parameters"].elts
        assert isinstance(params[0], ast.Name) and params[0].id == "config"
        overrides = ast.literal_eval(params[1])
        assert overrides["use_sim_time"] is True
        if names[-1] == "e3_pick_action_server":
            assert overrides["dry_run"] is False
            assert overrides["attach_enabled"] is True
            assert overrides["classification_transport_enabled"] is True
            for key, value in overrides.items():
                if "calibration" in key:
                    assert value is False
    assert len(set(names)) == 3
    assert source.count('"e3_sim.launch.py"') == 1
    assert source.count('"move_group.launch.py"') == 1
    sim = (ROOT / "launch/e3_sim.launch.py").read_text()
    assert "TimerAction" not in sim
    assert "OnProcessExit" in sim
    moveit = ROOT.parent / "mecharm_270_moveit_config"
    assert '["use_sim_time"] = True' in (
        moveit / "launch/move_group.launch.py").read_text()
    config = yaml.safe_load((ROOT / "config/sorting_config.yaml").read_text())
    assert config["task_manager"]["ros__parameters"]["use_sim_time"]
    assert config["e3_object_detector"]["ros__parameters"]["use_sim_time"]
