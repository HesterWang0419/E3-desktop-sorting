"""Offline tests; no nodes, services or actions are started."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml
from action_msgs.msg import GoalStatus

from e3_sorting_system.task_manager import SortingRound, TaskManager


CENTERS = {
    1: (203.5, 308.0), 2: (204.0, 269.5), 3: (320.0, 308.0),
    4: (320.0, 260.0), 5: (438.5, 308.0), 6: (436.0, 269.5)}


def detection(grid, dx=0, label=None, score=0.9):
    x, y = CENTERS[grid]
    return NS(
        bbox=NS(center=NS(position=NS(x=x + dx, y=y))),
        results=[NS(hypothesis=NS(
            class_id=label or ("comb" if grid % 2 else "mouse"),
            score=score))])


def make_round():
    return SortingRound(CENTERS, 3, 45.0, 0.7, 5.0, 0.0)


def stable(round_):
    for index in range(3):
        round_.observe([detection(g) for g in CENTERS], index)


def result(success=True, final_state="sorting_completed"):
    return NS(success=success, final_state=final_state, message="test")


def test_pixel_calibration():
    cfg = yaml.safe_load((
        Path(__file__).resolve().parents[1] /
        "config/sorting_config.yaml").read_text())
    for grid, center in CENTERS.items():
        assert cfg["task_manager"]["ros__parameters"][
            f"grid_{grid}.pixel_center"] == list(center)


def test_complete_consecutive_frames_required():
    round_ = make_round()
    full = [detection(g) for g in CENTERS]
    round_.observe(full, 0)
    round_.observe(full[:4], 1)
    assert round_.take() is None
    for now in (2, 3):
        round_.observe(full, now)
        assert round_.state == "observing"
    round_.observe(full, 4)
    assert round_.state == "ready"


def test_missing_timeout_reports_grids_and_never_dispatches():
    round_ = make_round()
    for now in (0, 1, 2):
        round_.observe([detection(g) for g in (1, 2, 5, 6)], now)
    round_.tick(5)
    assert round_.state == "failed"
    assert "missing grids=[3, 4]" in round_.message
    stable(round_)
    assert round_.take() is None


@pytest.mark.parametrize("item", [
    detection(1, dx=-46), detection(1, label="unknown"),
    detection(1, label="mouse"), detection(1, score=0.69),
    detection(1, dx=float("nan"))])
def test_invalid_detection_not_assigned(item):
    round_ = make_round()
    round_.observe([item], 0)
    assert round_.last == {}


def test_six_goals_unique_serial_and_disappearance_does_not_wait():
    round_ = make_round()
    stable(round_)
    for grid in CENTERS:
        goal = round_.take()
        assert goal == (
            grid, "comb" if grid % 2 else "mouse", f"pick_object_{grid}")
        assert round_.take() is None
        round_.observe([], 100)
        round_.tick(100)
        round_.finish(GoalStatus.STATUS_SUCCEEDED, result())
    assert round_.state == "sorting_completed"
    stable(round_)
    assert round_.take() is None
    assert round_.done == set(CENTERS)


@pytest.mark.parametrize("status,success", [
    (GoalStatus.STATUS_ABORTED, False),
    (GoalStatus.STATUS_CANCELED, False),
    (GoalStatus.STATUS_SUCCEEDED, False)])
def test_failed_action_stops_round(status, success):
    round_ = make_round()
    stable(round_)
    round_.take()
    round_.finish(status, result(success))
    assert round_.state == "failed"
    assert round_.take() is None
    assert "grid=1" in round_.message


def test_duplicate_detection_cannot_fill_two_grids():
    round_ = make_round()
    round_.observe([detection(1), detection(1)], 0)
    assert round_.last == {}


def test_action_adapter_waits_for_result_before_next_send():
    manager = TaskManager.__new__(TaskManager)
    manager.round = make_round()
    stable(manager.round)
    manager.pending = manager.goal_handle = manager.phase = None
    manager.wait_started = None
    manager.cfg = {"action_wait_timeout": 15.0}
    sent = []
    acceptance = Future()
    completion = Future()

    def send(goal):
        sent.append(goal)
        return acceptance

    manager.client = NS(server_is_ready=lambda: True, send_goal_async=send)
    manager.advance(0)
    manager.advance(1)
    assert len(sent) == 1
    acceptance.set_result(NS(
        accepted=True, get_result_async=lambda: completion))
    manager.advance(2)
    manager.advance(3)
    assert len(sent) == 1
    completion.set_result(NS(
        status=GoalStatus.STATUS_SUCCEEDED, result=result()))
    manager.advance(4)
    assert len(sent) == 1
    manager.advance(5)
    assert len(sent) == 2
    assert sent[1].object_model == "pick_object_2"


@pytest.mark.parametrize("outcome", ["rejected", "aborted", "canceled"])
def test_adapter_failure_never_dispatches_next(outcome):
    manager = TaskManager.__new__(TaskManager)
    manager.round = make_round()
    stable(manager.round)
    manager.pending = manager.goal_handle = manager.phase = None
    manager.wait_started = None
    manager.cfg = {"action_wait_timeout": 15.0}
    sent = []
    acceptance, completion = Future(), Future()

    def send(goal):
        sent.append(goal)
        return acceptance

    manager.client = NS(server_is_ready=lambda: True, send_goal_async=send)
    manager.advance(0)
    acceptance.set_result(NS(
        accepted=outcome != "rejected",
        get_result_async=lambda: completion))
    manager.advance(1)
    if outcome != "rejected":
        completion.set_result(NS(
            status=(GoalStatus.STATUS_ABORTED if outcome == "aborted"
                    else GoalStatus.STATUS_CANCELED),
            result=result(False, outcome)))
        manager.advance(2)
    manager.advance(3)
    assert manager.round.state == "failed"
    assert len(sent) == 1


def test_adapter_unavailable_server_times_out_without_sending():
    manager = TaskManager.__new__(TaskManager)
    manager.round = make_round()
    stable(manager.round)
    manager.pending = None
    manager.wait_started = None
    manager.cfg = {"action_wait_timeout": 15.0}
    manager.client = NS(server_is_ready=lambda: False)
    manager.advance(0)
    manager.advance(15)
    assert manager.round.state == "failed"
    assert manager.round.active is None


@pytest.mark.parametrize("interrupt,context_ok", [
    (False, True), (True, True), (True, False)])
def test_main_lifecycle_without_ros(monkeypatch, interrupt, context_ok):
    from e3_sorting_system import task_manager
    assert callable(task_manager.main)
    calls = []
    node = NS(destroy_node=lambda: calls.append("destroy"))

    def init(args=None):
        calls.append(("init", args))

    def create():
        calls.append("create")
        return node

    def spin(actual):
        assert actual is node
        calls.append("spin")
        if interrupt:
            raise KeyboardInterrupt()

    monkeypatch.setattr(task_manager, "TaskManager", create)
    monkeypatch.setattr(task_manager.rclpy, "init", init)
    monkeypatch.setattr(task_manager.rclpy, "spin", spin)
    monkeypatch.setattr(task_manager.rclpy, "ok", lambda: context_ok)
    monkeypatch.setattr(
        task_manager.rclpy, "shutdown", lambda: calls.append("shutdown"))
    task_manager.main(args=["test"])
    expected = [("init", ["test"]), "create", "spin", "destroy"]
    if context_ok:
        expected.append("shutdown")
    assert calls == expected


@pytest.mark.parametrize("success", [False, True])
def test_observation_gates_detection_and_dispatch(success):
    manager = TaskManager.__new__(TaskManager)
    manager.round = make_round()
    manager.round.state = "waiting"
    manager.startup_phase = "waiting"
    manager.startup_deadline = 120.0
    manager.cfg = {"detections_topic": "/e3/detections",
                   "detection_wait_timeout": 5.0}
    future = Future()
    calls = []
    manager.client = NS(server_is_ready=lambda: True)
    manager.count_publishers = lambda topic: 1
    manager.prepare_client = NS(
        service_is_ready=lambda: True,
        call_async=lambda request: calls.append(request) or future)
    assert not manager.startup(0)
    stable(manager.round)
    assert manager.round.take() is None
    assert not manager.startup(1)
    assert len(calls) == 1
    future.set_result(NS(success=success, message="preparation result"))
    assert manager.startup(2) == success
    assert manager.round.state == ("observing" if success else "failed")
    assert manager.round.count == 0
    assert manager.round.take() is None
    if success:
        assert manager.round.deadline == 7.0


@pytest.mark.parametrize("missing", ["action", "detector", "prepare"])
def test_startup_waits_for_all_endpoints(missing):
    manager = TaskManager.__new__(TaskManager)
    manager.round = make_round()
    manager.round.state = "waiting"
    manager.startup_phase = "waiting"
    manager.startup_deadline = 120.0
    manager.cfg = {"detections_topic": "/e3/detections"}
    manager.client = NS(server_is_ready=lambda: missing != "action")
    manager.count_publishers = lambda topic: int(missing != "detector")
    manager.prepare_client = NS(
        service_is_ready=lambda: missing != "prepare")
    assert not manager.startup(0)
    assert manager.round.state == "waiting"
    assert not manager.startup(120)
    assert manager.round.state == "failed"
