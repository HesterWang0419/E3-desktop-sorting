"""Offline regression tests using fake robot endpoints."""

from pathlib import Path
from types import SimpleNamespace as NS
import math
import threading

import pytest
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from e3_sorting_interfaces.action import PickAndSort
from gazebo_model_attachment_plugin_msgs.srv import Attach, Detach
from gazebo_msgs.srv import GetEntityState
from moveit_msgs.srv import (
    GetMotionPlan, GetPositionIK, GetStateValidity)
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.action import GoalResponse
from rclpy.task import Future
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from e3_sorting_system.pick_action_server import (
    CalibrationRequired, PickActionServer, TaskCancelled)


def ready(value):
    future = Future()
    future.set_result(value)
    return future


class Client:
    def __init__(self, responder):
        self.responder = responder
        self.requests = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.requests.append(request)
        return ready(self.responder(request))


class Goal:
    def __init__(self, object_class="comb"):
        self.request = PickAndSort.Goal(
            grid_id=1, object_class=object_class, object_model="pick_object_1")
        self.is_cancel_requested = False
        self.stages = []
        self.progress = []
        self.state = None

    def publish_feedback(self, feedback):
        self.stages.append(feedback.stage)
        self.progress.append(feedback.progress)

    def succeed(self):
        self.state = "succeeded"

    def abort(self):
        self.state = "aborted"

    def canceled(self):
        self.state = "canceled"


@pytest.fixture
def node():
    rclpy.init(args=["--ros-args", "-p", "dry_run:=false",
                     "-p", "feedback_delay:=0.0", "-p", "gripper_wait:=0.0",
                     "-p", "grid_1.pick_joints:="
                     "[0.724953, 0.819108, -0.589521, "
                     "-0.000043, 1.337840, 0.734401]"])
    server = PickActionServer()
    assert server.cfg["attach_enabled"] is False
    server.cfg["attach_enabled"] = True
    server.get_service_names_and_types = lambda: []
    assert server.attach_client.srv_name == "/gazebo/attach"
    assert server.detach_client.srv_name == "/gazebo/detach"
    assert server.attach_client.srv_type is Attach
    assert server.detach_client.srv_type is Detach
    server.joint_state_callback(seed_message(server))
    # Destroy real action transport before replacing it with a fake.
    server.arm_client.destroy()
    original = server.arm_client
    commands = []

    def publish_gripper(message):
        commands.extend(message.data)
        current = (server._joint_sample[2]
                   if server._joint_sample is not None else [0.1] * 6)
        sample = seed_message(server, current)
        sample.position[-1] = message.data[0]
        server.joint_state_callback(sample)

    server.gripper = NS(
        get_subscription_count=lambda: 1, publish=publish_gripper)

    def entity(request):
        result = GetEntityState.Response()
        result.success = True
        pos = result.state.pose.position
        if request.name in (
                "pick_object_1", "mecharm_270::grasp_attach_link"):
            pos.x, pos.y, pos.z = 0.1275, 0.1216, 0.4285
        else:
            pos.x, pos.y, pos.z = (
                0.31, 0.15 if request.name == "red_bin" else -0.15, 0.406)
        return result

    def ik(request):
        result = GetPositionIK.Response()
        result.error_code.val = 1
        # Reverse order ensures extraction uses joint names, not indices.
        result.solution.joint_state.name = list(
            reversed(server.cfg["joint_names"]))
        seed = request.ik_request.robot_state.joint_state.position
        result.solution.joint_state.position = [
            value + 0.01 for value in reversed(seed)]
        return result

    def plan(request):
        result = GetMotionPlan.Response()
        result.motion_plan_response.error_code.val = 1
        trajectory = result.motion_plan_response.trajectory.joint_trajectory
        trajectory.joint_names = server.cfg["joint_names"]
        point = JointTrajectoryPoint()
        constraints = request.motion_plan_request.goal_constraints[0]
        point.positions = [j.position for j in constraints.joint_constraints]
        point.time_from_start.sec = 1
        trajectory.points = [point]
        return result

    server.entity_client = Client(entity)
    server.ik_client = Client(ik)
    server.plan_client = Client(plan)
    server.attach_client = Client(
        lambda _: Attach.Response(success=True, message="attached"))
    server.detach_client = Client(
        lambda _: Detach.Response(success=True, message="detached"))
    trajectories = []
    result = FollowJointTrajectory.Result(error_code=0)
    timers = []
    measured = []

    def send_trajectory(goal):
        trajectories.append(goal.trajectory)

        def controller_result():
            actual = [value + 0.005
                      for value in goal.trajectory.points[-1].positions]

            def feedback():
                measured.append(actual)
                server.joint_state_callback(seed_message(server, actual))

            timer = threading.Timer(0.03, feedback)
            timers.append(timer)
            timer.start()
            return ready(NS(status=4, result=result))
        return ready(NS(accepted=True, get_result_async=controller_result))

    server.arm_client = NS(
        server_is_ready=lambda: True,
        send_goal_async=send_trajectory,
        destroy=lambda: None)
    server.test_measured = measured
    server.test_commands = commands
    server.test_trajectories = trajectories
    yield server
    for timer in timers:
        timer.join()
    server.destroy_node()
    del original
    rclpy.shutdown()


@pytest.mark.parametrize("object_class,bin_name,y", [
    ("comb", "red_bin", 0.15), ("mouse", "blue_bin", -0.15)])
def test_full_planned_route(node, object_class, bin_name, y):
    goal = Goal(object_class)
    assert node.goal_callback(goal.request) == GoalResponse.REJECT
    result = node.execute_callback(goal)
    assert result.success and goal.state == "succeeded"
    assert len(goal.stages) == 14
    assert goal.stages[-2:] == ["returning_home", "completed"]
    assert [r.name for r in node.entity_client.requests
            if "::" not in r.name] == [
        "pick_object_1", "pick_object_1", bin_name]
    poses = [r.ik_request.pose_stamped for r in node.ik_client.requests]
    assert [p.pose.position.z for p in poses] == pytest.approx(
        [0.480, 0.451, 0.480, 0.530])
    assert poses[0].pose.position.x == pytest.approx(0.145)
    assert poses[0].pose.position.y == pytest.approx(0.138)
    assert poses[-1].pose.position.y == pytest.approx(y)
    assert poses[-1].pose.orientation.y == pytest.approx(-0.702836, abs=1e-5)
    for request in node.ik_client.requests:
        ik = request.ik_request
        assert ik.robot_state.joint_state.name == node.cfg["joint_names"]
        assert len(ik.robot_state.joint_state.position) == 6
        assert ik.robot_state.is_diff
        assert ik.avoid_collisions is False
        assert ik.ik_link_name == "grasp_attach_link"
        assert ik.group_name == "mecharm_arm"
        assert ik.pose_stamped.header.frame_id == "world"
        assert ik.timeout.sec == 5
    for request in node.plan_client.requests:
        plan = request.motion_plan_request
        assert plan.start_state.is_diff
        assert plan.pipeline_id == "ompl"
        assert plan.planner_id == "RRTConnect"
        assert 0 < plan.max_velocity_scaling_factor <= 1
        constraints = plan.goal_constraints[0].joint_constraints
        assert [j.joint_name for j in constraints] == node.cfg["joint_names"]
    assert len(node.test_trajectories) == 5
    assert node.test_trajectories[0].points[0].positions == pytest.approx(
        [v + 0.01 for v in node.cfg["grid_1.pick_joints"]])
    assert list(node.test_trajectories[-1].points[0].positions) == [0.0] * 6
    assert node.test_commands == [0.15, -0.31, 0.15]
    attached = node.attach_client.requests[0]
    detached = node.detach_client.requests[0]
    assert attached.joint_name == detached.joint_name
    assert attached.model_name_2 == "pick_object_1"
    assert attached.link_name_1 == "grasp_attach_link"


def test_invalid_and_concurrent_goals(node):
    goal = Goal()
    goal.request.object_model = "pick_object_2"
    assert node.goal_callback(goal.request) == GoalResponse.REJECT
    goal.request.object_model = "pick_object_1"
    node.cfg["direct_pick_calibration"] = True
    node.cfg["attach_enabled"] = False
    assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
    assert node.goal_callback(goal.request) == GoalResponse.REJECT
    node.execute_callback(goal)


def test_ik_failure_aborts_before_execution(node):
    node.ik_client.responder = lambda _: GetPositionIK.Response()
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert result.final_state == "failed_moving_to_pregrasp"
    assert not node.test_trajectories
    assert node.test_commands == [0.15, 0.15]


@pytest.mark.parametrize("error,empty", [(0, False), (1, True)])
def test_bad_plan_never_executes(node, error, empty):
    response = GetMotionPlan.Response()
    response.motion_plan_response.error_code.val = error
    node.plan_client.responder = lambda _: response
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert not node.test_trajectories


def test_cancel_after_attachment_recovers(node):
    goal = Goal()
    original = goal.publish_feedback

    def feedback(value):
        original(value)
        if value.stage == "lifting_object":
            goal.is_cancel_requested = True
    goal.publish_feedback = feedback
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "canceled" and goal.state == "canceled"
    assert len(node.detach_client.requests) == 1
    assert len(node.test_trajectories) == 2
    assert node.test_commands[-1] == 0.15


def test_recovery_failure_latches_server(node):
    goal = Goal()
    original = node.entity_client.responder

    def entity(request):
        if request.name == "red_bin":
            raise RuntimeError("destination unavailable")
        return original(request)
    node.entity_client.responder = entity
    node.detach_client.responder = lambda _: Detach.Response(
        success=False, message="failed")
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "recovery incomplete" in result.message
    assert node.goal_callback(goal.request) == GoalResponse.REJECT


@pytest.mark.parametrize("accepted,status,error", [
    (False, 4, 0), (True, 6, 0), (True, 4, -4)])
def test_controller_rejection_and_errors(node, accepted, status, error):
    handle = NS(
        accepted=accepted,
        get_result_async=lambda: ready(NS(
            status=status,
            result=FollowJointTrajectory.Result(error_code=error))))
    node.arm_client.send_goal_async = lambda _: ready(handle)
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success and goal.state == "aborted"
    assert not node.attach_client.requests


def test_controller_timeout_cancels_before_recovery(node):
    pending = Future()
    canceled = []

    def cancel():
        canceled.append(True)
        pending.set_result(NS(status=GoalStatus.STATUS_CANCELED))
        return ready(NS())
    handle = NS(accepted=True, get_result_async=lambda: pending,
                cancel_goal_async=cancel)
    node.arm_client.send_goal_async = lambda _: ready(handle)
    node.cfg["execution_timeout"] = 0.03
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success and canceled
    assert "recovery completed" in result.message
    assert not node._faulted


def test_waits_timeout_cancel_and_async_completion(node):
    with pytest.raises(TimeoutError):
        node.wait_future(Future(), 0.02, "test")
    goal = Goal()
    goal.is_cancel_requested = True
    with pytest.raises(TaskCancelled):
        node.wait_future(Future(), 1.0, "test", goal)
    future = Future()
    timer = threading.Timer(0.02, lambda: future.set_result(42))
    timer.start()
    assert node.wait_future(future, 1.0, "test") == 42
    timer.join()


def test_dry_run_never_uses_robot_endpoints(node):
    node.dry_run = True
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.success
    assert not node.entity_client.requests
    assert not node.test_trajectories
    assert not node.test_commands
    node.dry_run = False


@pytest.mark.parametrize("cancel", [False, True])
def test_ros_action_transport_dry_run(node, cancel):
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor

    node.dry_run = True
    node.cfg["feedback_delay"] = 0.02
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    worker = threading.Thread(target=executor.spin)
    worker.start()
    client = ActionClient(node, PickAndSort, node.cfg["action_name"])
    try:
        assert client.wait_for_server(timeout_sec=3.0)
        handle = node.wait_future(
            client.send_goal_async(Goal().request), 3.0, "test acceptance")
        assert handle.accepted
        if cancel:
            response = node.wait_future(
                handle.cancel_goal_async(), 3.0, "test cancel")
            assert response.goals_canceling
        response = node.wait_future(
            handle.get_result_async(), 5.0, "test result")
        assert response.status == (
            GoalStatus.STATUS_CANCELED if cancel
            else GoalStatus.STATUS_SUCCEEDED)
        assert response.result.success is not cancel
        assert not node.test_commands
        assert not node.test_trajectories
    finally:
        client.destroy()
        executor.shutdown()
        worker.join(timeout=3.0)
        node.dry_run = False


def test_ik_collision_default(node):
    assert node.get_parameter("ik_avoid_collisions").value is False
    assert node.cfg["ik_avoid_collisions"] is False


@pytest.mark.parametrize("avoid_collisions", [False, True])
def test_ik_collision_yaml_loading(tmp_path, avoid_collisions):
    config = (Path(__file__).resolve().parents[1]
              / "config" / "sorting_config.yaml")
    text = config.read_text()
    assert "    ik_avoid_collisions: false" in text
    if avoid_collisions:
        # A true variant proves YAML overrides reach cfg, beyond the default.
        config = tmp_path / "sorting_config.yaml"
        config.write_text(text.replace(
            "ik_avoid_collisions: false", "ik_avoid_collisions: true"))
    rclpy.init(args=["--ros-args", "--params-file", str(config)])
    server = None
    try:
        server = PickActionServer()
        assert server.dry_run is True
        assert server.cfg["attach_enabled"] is False
        assert server.cfg["red_bin.approach_joints"] == [
            1.5780840510, 0.8013771349, -0.9467428297,
            -0.0332285864, 1.7100726856, 1.6302872993]
        assert server.cfg["blue_bin.approach_joints"] == [
            -1.5779212067, 0.8827475163, -1.0850132822,
            0.0335590192, 1.7794357656, -1.5141581248]
        assert server.cfg["red_bin.slot_offsets_x"] == [-0.040, 0.0, 0.040]
        assert server.cfg["blue_bin.slot_offsets_x"] == [-0.040, 0.0, 0.040]
        assert server.cfg["red_bin.center_world_xy"] == [0.0, 0.18]
        assert server.cfg["blue_bin.center_world_xy"] == [0.0, -0.18]
        assert server.cfg["direct_pick_calibration"] is False
        assert server.cfg["local_center_calibration"] is False
        assert server.cfg["local_center_tolerance"] == 0.0035
        assert server.cfg["local_center_max_iterations"] == 3
        assert server.cfg["local_ik_seed_max_distance"] == 0.40
        assert server.cfg["ik_seed_max_distance"] == 1.0
        assert server.cfg["grid_1.pick_joints"] == [
            0.724953, 0.819108, -0.589521, -0.000043, 1.337840, 0.734401]
        assert server.cfg["planning_pipeline_id"] == "ompl"
        assert server.cfg["planner_id"] == "RRTConnect"
        parameter = server.get_parameter("ik_avoid_collisions")
        assert parameter.value is avoid_collisions
        assert server.cfg["ik_avoid_collisions"] is avoid_collisions
    finally:
        if server is not None:
            server.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("avoid_collisions", [False, True])
def test_ik_request_and_target_log(node, monkeypatch, avoid_collisions):
    node.cfg["ik_avoid_collisions"] = avoid_collisions
    messages = []
    monkeypatch.setattr(node, "get_logger", lambda: NS(info=messages.append))
    position = NS(x=0.1275, y=0.1216, z=0.4285)
    pose = node.target_pose(
        position, node.cfg["grasp_offset"],
        node.cfg["grasp_orientation"], node.cfg["pregrasp_height"])
    responder = node.ik_client.responder

    def check_log_before_request(request):
        assert messages, "IK target must be logged before the service call"
        return responder(request)

    node.ik_client.responder = check_log_before_request
    node.solve_ik(pose, Goal(), purpose="moving_to_pregrasp")
    request = node.ik_client.requests[0].ik_request
    assert request.avoid_collisions is avoid_collisions
    assert "purpose=moving_to_pregrasp" in messages[0]
    assert "position=(0.1450000000, 0.1380000000, 0.4800000000)" in messages[0]
    q = pose.pose.orientation
    assert (f"orientation=({q.x:.10f}, {q.y:.10f}, "
            f"{q.z:.10f}, {q.w:.10f})") in messages[0]
    assert f"avoid_collisions={avoid_collisions}" in messages[0]


@pytest.mark.parametrize("pipeline,planner", [
    ("ompl", "RRTConnect"), ("test_pipeline", "test_planner")])
def test_planning_request_parameters_and_log(node, monkeypatch,
                                             pipeline, planner):
    assert node.get_parameter("planning_pipeline_id").value == "ompl"
    assert node.get_parameter("planner_id").value == "RRTConnect"
    node.cfg["planning_pipeline_id"] = pipeline
    node.cfg["planner_id"] = planner
    messages = []
    monkeypatch.setattr(node, "get_logger", lambda: NS(info=messages.append))
    responder = node.plan_client.responder

    def check_log_before_request(request):
        assert messages, "Planning targets must be logged before the request"
        return responder(request)

    node.plan_client.responder = check_log_before_request
    values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    node.plan_joints(values, Goal())
    plan = node.plan_client.requests[0].motion_plan_request
    assert plan.pipeline_id == pipeline
    assert plan.planner_id == planner
    assert "group_name=mecharm_arm" in messages[0]
    assert f"pipeline_id={pipeline}, planner_id={planner}" in messages[0]
    for name, value in zip(node.cfg["joint_names"], values):
        assert f"{name}={value:.6f}" in messages[0]
    assert "planning_time=5.00s, planning_attempts=5" in messages[0]


def seed_message(node, values=None):
    message = JointState()
    message.name = list(reversed(node.cfg["joint_names"]))
    message.position = list(reversed(
        values if values is not None else [0.1] * 6))
    message.name.append("gripper_controller")
    message.position.append(-0.31)
    message.header.stamp = node.get_clock().now().to_msg()
    return message


def test_ik_seed_uses_latest_named_arm_state(node):
    for values in ([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [0.2] * 6):
        message = seed_message(node, values)
        node.joint_state_callback(message)
        pose = node.target_pose(
            NS(x=0.1, y=0.1, z=0.4),
            node.cfg["grasp_offset"], node.cfg["grasp_orientation"])
        node.solve_ik(pose, Goal())
        state = node.ik_client.requests[-1].ik_request.robot_state.joint_state
        assert state.name == node.cfg["joint_names"]
        assert list(state.position) == values
        assert state.header.stamp == message.header.stamp


@pytest.mark.parametrize("problem", [
    "missing", "stale", "incomplete", "nonfinite", "duplicate"])
def test_ik_seed_rejects_invalid_samples(node, problem):
    node.cfg["joint_state_timeout"] = 0.03
    message = seed_message(node)
    if problem == "incomplete":
        message.name.pop(0)
        message.position.pop(0)
    elif problem == "nonfinite":
        message.position[0] = float("nan")
    elif problem == "duplicate":
        message.name[0] = message.name[1]
    node.joint_state_callback(message)
    if problem == "missing":
        node._joint_sample = None
    elif problem == "stale":
        _, stamp, positions = node._joint_sample
        node._joint_sample = (0.0, stamp, positions)
    pose = node.target_pose(
        NS(x=0.1, y=0.1, z=0.4),
        node.cfg["grasp_offset"], node.cfg["grasp_orientation"])
    with pytest.raises(TimeoutError, match="six-joint IK seed"):
        node.solve_ik(pose, Goal())
    assert not node.ik_client.requests


def test_seed_wait_accepts_async_sample_and_cancel(node):
    node._joint_sample = None
    timer = threading.Timer(
        0.03, lambda: node.joint_state_callback(seed_message(node)))
    timer.start()
    try:
        assert len(node.current_ik_seed(Goal()).position) == 6
    finally:
        timer.join()
    goal = Goal()
    goal.is_cancel_requested = True
    with pytest.raises(TaskCancelled):
        node.current_ik_seed(goal)


@pytest.mark.parametrize("endpoint", ["attach_client", "detach_client"])
def test_missing_attachment_endpoint_never_attaches(node, endpoint):
    node.cfg["service_timeout"] = 0.03
    getattr(node, endpoint).service_is_ready = lambda: False
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success and goal.state == "aborted"
    assert "endpoint unavailable" in result.message
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    assert node._attached_model is None
    assert not node.test_trajectories


def test_attach_rejection_does_not_recover_detach(node):
    node.attach_client.responder = lambda _: Attach.Response(
        success=False, message="constraint rejected")
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "failed_attaching_object"
    assert "constraint rejected" in result.message
    assert node._attached_model is None
    assert not node.detach_client.requests
    assert len(node.test_trajectories) == 2


def test_attach_timeout_unknown_is_not_confirmed(node):
    node.cfg["service_timeout"] = 0.03
    pending = Future()
    node.attach_client.call_async = lambda _: pending
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success and node._faulted
    assert "attach outcome unknown" in result.message
    assert node._attached_model is None
    assert not node.detach_client.requests
    assert len(node.test_trajectories) == 2


def test_pending_attach_confirmed_during_recovery(node):
    node._joint_name = "test_joint"
    node._pending_attach_model = "pick_object_1"
    node._pending_attach = ready(Attach.Response(
        success=True, message="late success"))
    assert node._attached_model is None
    errors = node.recover("pick_object_1")
    assert not errors
    assert len(node.detach_client.requests) == 1
    assert node._attached_model is None


def test_pending_attach_rejected_during_recovery(node):
    node._pending_attach_model = "pick_object_1"
    node._pending_attach = ready(Attach.Response(
        success=False, message="late rejection"))
    assert node.recover("pick_object_1") == []
    assert node._attached_model is None
    assert not node.detach_client.requests


def test_detach_temporary_unavailability_and_confirmed_model(node):
    node._joint_name = "test_joint"
    node.attach_object("pick_object_1", Goal())
    assert node._attached_model == "pick_object_1"
    available = threading.Event()
    node.detach_client.service_is_ready = available.is_set
    timer = threading.Timer(0.03, available.set)
    timer.start()
    try:
        # Recover the recorded object even if a caller supplies a wrong model.
        assert node.recover("pick_object_2") == []
    finally:
        timer.join()
    assert node.detach_client.requests[0].model_name_2 == "pick_object_1"
    assert node._attached_model is None


def test_missing_detach_keeps_confirmed_state(node):
    node._joint_name = "test_joint"
    node.attach_object("pick_object_1", Goal())
    node.cfg["service_timeout"] = 0.03
    node.detach_client.service_is_ready = lambda: False
    errors = node.recover("pick_object_1")
    assert any("/gazebo/detach: endpoint unavailable" in e for e in errors)
    assert node._attached_model == "pick_object_1"
    assert node._faulted


def test_pending_detach_is_not_resent(node):
    node._joint_name = "test_joint"
    node.attach_object("pick_object_1", Goal())
    node.cfg["service_timeout"] = 0.03
    pending = Future()
    sends = []
    node.detach_client.call_async = lambda r: (sends.append(r) or pending)
    with pytest.raises(TimeoutError):
        node.detach_object("pick_object_1")
    assert node._attached_model == "pick_object_1"
    pending.set_result(Detach.Response(success=True, message="detached"))
    assert node.recover("pick_object_1") == []
    assert len(sends) == 1
    assert node._attached_model is None


@pytest.mark.parametrize("failure", ["distance", "open", "missing", "stale"])
def test_grasp_guards_never_attach(node, failure):
    node._joint_name = "test_joint"
    if failure == "distance":
        original = node.entity_client.responder

        def entity(request):
            response = original(request)
            if "::" in request.name:
                response.state.pose.position.x += 0.1
            return response
        node.entity_client.responder = entity
    elif failure == "open":
        message = seed_message(node)
        message.position[-1] = 0.15
        node.joint_state_callback(message)
    elif failure == "missing":
        node._gripper_sample = None
    else:
        node._gripper_sample = (0.0, -0.31)
    with pytest.raises(RuntimeError, match="Attach forbidden"):
        node.attach_object("pick_object_1", Goal())
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    assert not node.attached


def test_default_calibration_stops_after_closing(node):
    node.cfg["attach_enabled"] = False
    node.attach_client.service_is_ready = lambda: pytest.fail("attach probe")
    node.detach_client.service_is_ready = lambda: pytest.fail("detach probe")
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert len(node.test_trajectories) == 2
    assert node.test_commands == [0.15, -0.31]
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    assert not node.attached
    assert "lifting_object" not in goal.stages


def test_advertised_attachment_services_not_false_unavailable(node):
    node._joint_name = "test_joint"
    node.attach_client.service_is_ready = lambda: False
    node.detach_client.service_is_ready = lambda: False
    node.get_service_names_and_types = lambda: [
        (name, ["gazebo_model_attachment_plugin_msgs/srv/"
                + ("Detach" if name.endswith("detach") else "Attach")])
        for name in ("/gazebo/attach", "/gazebo/detach")]
    node.attach_object("pick_object_1", Goal())
    assert node.attached
    assert node._attached_model == "pick_object_1"
    assert node.recover("pick_object_1") == []
    assert not node.attached
    assert len(node.detach_client.requests) == 1


def test_advertised_service_without_reply_does_not_confirm_attach(node):
    node._joint_name = "test_joint"
    node.cfg["service_timeout"] = 0.03
    node.attach_client.service_is_ready = lambda: False
    node.get_service_names_and_types = lambda: [
        ("/gazebo/attach", ["gazebo_model_attachment_plugin_msgs/srv/Attach"])]
    node.attach_client.call_async = lambda _: Future()
    with pytest.raises(TimeoutError, match="timed out"):
        node.attach_object("pick_object_1", Goal())
    assert not node.attached
    assert not node.detach_client.requests


def test_wrong_service_type_is_diagnosed(node):
    node.cfg["service_timeout"] = 0.03
    node.attach_client.service_is_ready = lambda: False
    node.get_service_names_and_types = lambda: [
        ("/gazebo/attach", ["wrong_package/srv/Wrong"])]
    with pytest.raises(TimeoutError, match="wrong_package/srv/Wrong"):
        node.attach_object("pick_object_1", Goal())
    assert not node.attach_client.requests


def test_initial_pregrasp_uses_calibrated_grid_reference(node):
    node.joint_state_callback(seed_message(node, [0.0] * 6))
    node.cfg["attach_enabled"] = False
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_required"
    seed = node.ik_client.requests[0].ik_request.robot_state.joint_state
    assert seed.name == node.cfg["joint_names"]
    assert list(seed.position) == node.cfg["grid_1.pick_joints"]


def test_stage_seeds_use_executed_feedback_not_ik_targets(node):
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.success
    seeds = [list(r.ik_request.robot_state.joint_state.position)
             for r in node.ik_client.requests]
    assert seeds[0] == node.cfg["grid_1.pick_joints"]
    assert seeds[1] == node.test_measured[0]
    assert seeds[2] == node.test_measured[1]
    for index in (0, 1):
        planned = node.test_trajectories[index].points[-1].positions
        assert node.test_measured[index] != list(planned)


def test_far_ik_branch_never_reaches_planning(node):
    original = node.ik_client.responder

    def far_branch(request):
        response = original(request)
        response.solution.joint_state.position[0] += 2.0
        return response
    node.ik_client.responder = far_branch
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert "IK branch rejected" in result.message
    assert not node.plan_client.requests
    assert not node.test_trajectories


def test_ik_distance_log_and_configurable_threshold(node, monkeypatch):
    logs = []
    monkeypatch.setattr(node, "get_logger", lambda: NS(info=logs.append))
    pose = node.target_pose(NS(x=0.1, y=0.1, z=0.4),
                            node.cfg["grasp_offset"],
                            node.cfg["grasp_orientation"])
    node.cfg["ik_seed_max_distance"] = 0.01
    with pytest.raises(RuntimeError, match="IK branch rejected"):
        node.solve_ik(pose, Goal())
    assert "seed_distance=0.0244948974rad" in logs[-1]
    for name in node.cfg["joint_names"]:
        assert f"{name}=+0.010000" in logs[-1]


@pytest.mark.parametrize("grid", range(2, 7))
def test_uncalibrated_grid_stops_without_motion(node, grid):
    node.cfg["attach_enabled"] = False
    goal = Goal()
    goal.request.grid_id = grid
    goal.request.object_model = f"pick_object_{grid}"
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_required"
    assert f"grid {grid} is uncalibrated" in result.message
    assert not node.test_commands
    assert not node.ik_client.requests
    assert not node.test_trajectories


def test_post_execution_seed_requires_new_feedback(node):
    node.cfg["joint_state_timeout"] = 0.03
    handle = NS(accepted=True, get_result_async=lambda: ready(NS(
        status=4, result=FollowJointTrajectory.Result(error_code=0))))
    node.arm_client.send_goal_async = lambda _: ready(handle)
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert "six-joint IK seed" in result.message
    assert len(node.ik_client.requests) == 1


def test_direct_calibration_exact_joints_without_ik_or_grasp(
        node, monkeypatch):
    assert node.get_parameter("direct_pick_calibration").value is False
    node.cfg["attach_enabled"] = False
    node.cfg["direct_pick_calibration"] = True

    def forbidden(*args, **kwargs):
        pytest.fail("Direct calibration must not call IK or attachment")

    monkeypatch.setattr(node, "solve_ik", forbidden)
    monkeypatch.setattr(node, "attach_object", forbidden)
    monkeypatch.setattr(node, "check_attachment_services", forbidden)
    logs = []
    monkeypatch.setattr(node, "get_logger", lambda: NS(
        info=logs.append, warning=logs.append, error=logs.append))
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert "holding current pose with gripper open" in result.message
    assert len(node.plan_client.requests) == 1
    constraints = node.plan_client.requests[0].motion_plan_request
    targets = constraints.goal_constraints[0].joint_constraints
    expected = [0.724953, 0.819108, -0.589521, -0.000043, 1.337840, 0.734401]
    assert [j.joint_name for j in targets] == node.cfg["joint_names"]
    assert [j.position for j in targets] == expected
    assert len(node.test_trajectories) == 1
    assert list(node.test_trajectories[0].points[-1].positions) == expected
    assert node.test_commands == [0.15]
    assert not node.ik_client.requests
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    assert goal.stages == [
        "validating_goal", "opening_gripper", "querying_object_pose",
        "moving_to_pick_joints", "recording_calibration"]
    assert [r.name for r in node.entity_client.requests] == [
        "pick_object_1", "pick_object_1", "mecharm_270::grasp_attach_link"]
    assert any(f"actual_joints={node.test_measured[0]}" in s for s in logs)
    assert any("object_world=" in s and "link_world=" in s
               and "distance=" in s for s in logs)


@pytest.mark.parametrize("grid,configured", [
    (1, False), (2, False), (2, True)])
def test_direct_calibration_rejects_before_motion(node, grid, configured):
    node.cfg["attach_enabled"] = False
    node.cfg["direct_pick_calibration"] = True
    node.cfg[f"grid_{grid}.pick_joints"] = [0.1] * 6 if configured else []
    goal = Goal()
    goal.request.grid_id = grid
    goal.request.object_model = f"pick_object_{grid}"
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_required"
    assert not node.test_commands
    assert not node.entity_client.requests
    assert not node.plan_client.requests
    assert not node.test_trajectories


def test_direct_calibration_with_attachment_enabled_rejects(node):
    node.cfg["direct_pick_calibration"] = True
    node.cfg["attach_enabled"] = True
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_required"
    assert "requires attach_enabled=false" in result.message
    assert not node.test_commands
    assert not node.plan_client.requests


def test_direct_calibration_planner_rejection_never_executes(node):
    node.cfg["direct_pick_calibration"] = True
    node.cfg["attach_enabled"] = False
    response = GetMotionPlan.Response()
    response.motion_plan_response.error_code.val = -2
    node.plan_client.responder = lambda _: response
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "failed_moving_to_pick_joints"
    assert "Planning error_code=-2" in result.message
    assert not node.test_trajectories
    assert not node.ik_client.requests
    assert -0.31 not in node.test_commands
    assert not node.attach_client.requests


def test_direct_calibration_dry_run_never_moves(node):
    node.cfg["direct_pick_calibration"] = True
    node.cfg["attach_enabled"] = False
    node.dry_run = True
    try:
        goal = Goal()
        node.goal_callback(goal.request)
        result = node.execute_callback(goal)
        assert result.success
        assert not node.test_commands
        assert not node.plan_client.requests
        assert not node.test_trajectories
    finally:
        node.dry_run = False


@pytest.fixture
def local_node(node):
    assert node.cfg["local_center_calibration"] is False
    assert node.cfg["local_ik_seed_max_distance"] == 0.40
    node.cfg["direct_pick_calibration"] = True
    node.cfg["local_center_calibration"] = True
    node.cfg["local_center_tolerance"] = 0.001
    node.cfg["attach_enabled"] = False
    node.cfg["grasp_offset"] = [99.0, -99.0, 99.0]

    def entity(request):
        result = GetEntityState.Response()
        result.success = True
        pose = result.state.pose
        point = (0.1300002, 0.1200037, 0.4247174)
        if "::" in request.name and len(node.test_trajectories) < 2:
            point = (0.1274, 0.1167, 0.4226)
        pose.position.x, pose.position.y, pose.position.z = point
        pose.orientation.z, pose.orientation.w = 0.6, 0.8
        return result

    node.entity_client.responder = entity
    return node


def test_local_center_target_orientation_seed_and_hold(local_node, capfd):
    node = local_node
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_completed"
    assert result.success
    assert "tolerance reached" in result.message
    assert len(node.test_trajectories) == 2
    assert len(node.ik_client.requests) == 1
    request = node.ik_client.requests[0].ik_request
    pose = request.pose_stamped
    assert pose.header.frame_id == "world"
    assert [pose.pose.position.x, pose.pose.position.y,
            pose.pose.position.z] == [0.1300002, 0.1200037, 0.4247174]
    q = pose.pose.orientation
    assert [q.x, q.y, q.z, q.w] == [0.0, 0.0, 0.6, 0.8]
    assert list(request.robot_state.joint_state.position) == (
        node.test_measured[0])
    assert list(node.test_trajectories[0].points[-1].positions) == (
        node.cfg["grid_1.pick_joints"])
    assert node.test_commands == [0.15]
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    assert goal.stages[-2:] == [
        "local_center_correction", "calibration_completed"]
    assert [r.name for r in node.entity_client.requests][-2:] == [
        "pick_object_1", "mecharm_270::grasp_attach_link"]
    output = capfd.readouterr().err
    assert "distance=0.0000000000m" in output
    assert "actual_joints=" in output


@pytest.mark.parametrize("distance,accepted", [
    (0.01, True), (0.0100561316, True), (0.012, True),
    (0.012 + 2e-9, False), (0.0121, False)])
def test_local_translation_limit_before_ik(local_node, distance, accepted):
    node = local_node
    original = node.entity_client.responder

    def entity(request):
        result = original(request)
        if "::" in request.name:
            pos = result.state.pose.position
            pos.x, pos.y, pos.z = (
                0.1300002 + distance, 0.1200037, 0.4247174)
        return result

    node.entity_client.responder = entity
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    if accepted:
        assert len(node.ik_client.requests) == 3
        assert result.final_state == "calibration_required"
    else:
        assert "exceeds 0.0120000000m" in result.message
        assert not node.ik_client.requests
        assert len(node.plan_client.requests) == 1
        assert len(node.test_trajectories) == 1


@pytest.mark.parametrize("limit,delta", [(0.30, 0.31), (0.01, 0.02)])
def test_local_ik_threshold_blocks_second_plan(local_node, limit, delta):
    node = local_node
    node.cfg["local_ik_seed_max_distance"] = limit
    original = node.ik_client.responder

    def ik(request):
        result = original(request)
        result.solution.joint_state.position[0] += delta
        return result

    node.ik_client.responder = ik
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "IK branch rejected" in result.message
    assert len(node.plan_client.requests) == 1
    assert len(node.test_trajectories) == 1
    assert not node.attach_client.requests


def test_local_invalid_orientation_rejected_before_ik(local_node):
    node = local_node
    original = node.entity_client.responder

    def entity(request):
        result = original(request)
        result.state.pose.orientation.w = float("nan")
        return result

    node.entity_client.responder = entity
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "invalid link orientation" in result.message
    assert not node.ik_client.requests
    assert len(node.test_trajectories) == 1


@pytest.mark.parametrize("direct,attach", [(False, False), (True, True)])
def test_local_invalid_combination_has_no_motion(node, direct, attach):
    node.cfg["local_center_calibration"] = True
    node.cfg["local_center_tolerance"] = 0.001
    node.cfg["direct_pick_calibration"] = direct
    node.cfg["attach_enabled"] = attach
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.final_state == "calibration_required"
    assert "local_center_calibration requires" in result.message
    assert not node.test_commands
    assert not node.plan_client.requests


def test_local_rejected_plan_is_not_executed(local_node):
    node = local_node
    original = node.plan_client.responder

    def plan(request):
        result = original(request)
        if len(node.plan_client.requests) == 2:
            result.motion_plan_response.error_code.val = -2
        return result

    node.plan_client.responder = plan
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "Planning error_code=-2" in result.message
    assert len(node.test_trajectories) == 1


def residual_sequence(node, distances):
    original = node.entity_client.responder

    def entity(request):
        result = original(request)
        iteration = max(0, len(node.test_trajectories) - 1)
        centre = 0.1300002 + iteration * 0.0001
        result.state.pose.position.x = centre
        result.state.pose.position.y = 0.1200037
        result.state.pose.position.z = 0.4247174
        if "::" in request.name:
            result.state.pose.position.x += distances[
                min(iteration, len(distances) - 1)]
        q = result.state.pose.orientation
        q.z, q.w = (0.6, 0.8) if iteration == 0 else (0.0, 1.0)
        return result
    node.entity_client.responder = entity


def test_center_already_within_tolerance_never_calls_ik(local_node):
    node = local_node
    residual_sequence(node, [0.0009])
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.success
    assert result.final_state == "calibration_completed"
    assert "corrections=0" in result.message
    assert len(node.test_trajectories) == 1
    assert not node.ik_client.requests
    assert node.test_commands == [0.15]


@pytest.mark.parametrize("residuals,count", [
    ([0.004, 0.0023, 0.0008], 2),
    ([0.004, 0.0023, 0.0015, 0.0008], 3)])
def test_center_iterates_with_fresh_pose_seed_and_orientation(
        local_node, capfd, residuals, count):
    node = local_node
    residual_sequence(node, residuals)
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.success
    assert result.final_state == "calibration_completed"
    assert len(node.ik_client.requests) == count
    assert len(node.test_trajectories) == count + 1
    for i, request in enumerate(node.ik_client.requests):
        ik = request.ik_request
        assert list(ik.robot_state.joint_state.position) == (
            node.test_measured[i])
        assert ik.pose_stamped.pose.position.x == pytest.approx(
            0.1300002 + i * 0.0001)
        q = ik.pose_stamped.pose.orientation
        assert (q.z, q.w) == ((0.6, 0.8) if i == 0 else (0.0, 1.0))
    assert node.test_commands == [0.15]
    assert not node.attach_client.requests
    assert not node.detach_client.requests
    log = capfd.readouterr().err
    assert f"iteration={count}/3" in log
    assert "dx=" in log and "dy=" in log and "dz=" in log
    assert "distance=0.0008000000m" in log


@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_center_nonconvergence_stops_at_configured_limit(local_node, maximum):
    node = local_node
    node.cfg["local_center_max_iterations"] = maximum
    residual_sequence(node, [0.0023])
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert "iteration limit reached without convergence" in result.message
    assert "distance=0.0023000m" in result.message
    assert len(node.ik_client.requests) == maximum
    assert len(node.test_trajectories) == maximum + 1
    assert node.test_commands == [0.15]
    assert not node.attach_client.requests


def test_center_displacement_limit_rechecked_each_iteration(local_node):
    node = local_node
    residual_sequence(node, [0.004, 0.013])
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "exceeds 0.0120000000m" in result.message
    assert len(node.ik_client.requests) == 1
    assert len(node.test_trajectories) == 2


def test_center_ik_limit_rechecked_each_iteration(local_node):
    node = local_node
    residual_sequence(node, [0.004, 0.0023])
    original = node.ik_client.responder

    def ik(request):
        response = original(request)
        if len(node.ik_client.requests) == 2:
            response.solution.joint_state.position[0] += 0.3
        return response
    node.ik_client.responder = ik
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert "IK branch rejected" in result.message
    assert len(node.ik_client.requests) == 2
    assert len(node.plan_client.requests) == 2
    assert len(node.test_trajectories) == 2


def test_custom_center_tolerance(local_node):
    node = local_node
    node.cfg["local_center_tolerance"] = 0.0005
    residual_sequence(node, [0.0008, 0.0004])
    goal = Goal()
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert result.success and len(node.ik_client.requests) == 1


@pytest.mark.parametrize("parameter,value", [
    ("local_center_max_iterations", "0"),
    ("local_center_max_iterations", "4"),
    ("local_center_tolerance", "0.0"),
    ("local_center_tolerance", "0.02"),
    ("local_ik_seed_max_distance", "0.41")])
def test_invalid_local_iteration_limits(parameter, value):
    rclpy.init(args=["--ros-args", "-p", f"{parameter}:={value}"])
    try:
        with pytest.raises(ValueError, match=parameter):
            PickActionServer()
    finally:
        rclpy.shutdown()


@pytest.fixture
def lift_node(node):
    assert node.cfg["lift_calibration"] is False
    assert node.cfg["lift_calibration_height"] == 0.05
    assert node.cfg["return_height_tolerance"] == 0.005
    assert node.cfg["close_gripper_position"] == -0.31
    assert node.cfg["open_gripper_position"] == 0.15
    assert node.cfg["attach_distance_limit"] == 0.02
    assert node.cfg["local_center_tolerance"] == 0.0035
    assert node.cfg["local_ik_seed_max_distance"] == 0.40
    node.cfg["lift_calibration"] = True
    node.cfg["direct_pick_calibration"] = True
    node.cfg["local_center_calibration"] = True
    node.cfg["object_settle_time"] = 0.0
    node.cfg["grasp_offset"] = [10.0, 20.0, 30.0]
    events = []

    def entity(request):
        events.append("query:" + request.name)
        response = GetEntityState.Response(success=True)
        pose = response.state.pose
        pose.position.x, pose.position.y = 0.1275, 0.1216
        pose.position.z = (
            0.4785 if len(node.test_trajectories) == 2 else 0.4285)
        pose.orientation.z = 0.6
        pose.orientation.w = 0.8
        return response

    node.entity_client.responder = entity
    for name in ("command_gripper", "verify_closed_gripper", "plan_joints",
                 "execute_trajectory", "attach_object", "detach_object"):
        original = getattr(node, name)

        def wrap(*args, _name=name, _original=original, **kwargs):
            events.append(_name + (
                ":" + str(args[0]) if _name == "command_gripper" else ""))
            return _original(*args, **kwargs)
        setattr(node, name, wrap)
    node.test_events = events
    return node


def run_cycle(node):
    goal = Goal()
    assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
    return node.execute_callback(goal), goal


def test_calibration_cycle_order_target_seed_and_release(lift_node):
    node = lift_node
    result, goal = run_cycle(node)
    assert result.success
    assert result.final_state == "calibration_cycle_completed"
    assert not node.attached
    assert node.test_commands == [0.15, -0.31, 0.15]
    assert len(node.test_trajectories) == 3
    assert list(node.test_trajectories[0].points[-1].positions) == (
        node.cfg["grid_1.pick_joints"])
    assert list(node.test_trajectories[2].points[-1].positions) == (
        node.test_measured[0])
    request, = node.ik_client.requests
    target = request.ik_request.pose_stamped
    assert target.header.frame_id == "world"
    assert target.pose.position.x == 0.1275
    assert target.pose.position.y == 0.1216
    assert target.pose.position.z == pytest.approx(0.4785)
    assert target.pose.orientation.z == 0.6
    assert target.pose.orientation.w == 0.8
    assert list(request.ik_request.robot_state.joint_state.position) == (
        node.test_measured[0])
    attach, = node.attach_client.requests
    detach, = node.detach_client.requests
    assert isinstance(detach, Detach.Request)
    assert set(detach.get_fields_and_field_types()) == {
        "joint_name", "model_name_1", "model_name_2"}
    assert attach.joint_name == detach.joint_name
    assert detach.model_name_1 == "mecharm_270"
    assert detach.model_name_2 == "pick_object_1"
    events = node.test_events
    assert events.index("command_gripper:False") < events.index(
        "verify_closed_gripper") < events.index(
            "query:pick_object_1::object_link")
    assert events.index("detach_object") > max(
        i for i, event in enumerate(events) if event == "execute_trajectory")
    assert events.index("detach_object") < len(events) - 3
    assert events[-3:] == [
        "command_gripper:True", "query:pick_object_1::object_link",
        "query:mecharm_270::grasp_attach_link"]
    for request in node.plan_client.requests:
        plan = request.motion_plan_request
        assert plan.max_velocity_scaling_factor <= 0.1
        assert plan.max_acceleration_scaling_factor <= 0.1
    assert "returning_home" not in goal.stages
    assert "querying_destination" not in goal.stages


@pytest.mark.parametrize("flag,value", [
    ("dry_run", True), ("direct_pick_calibration", False),
    ("local_center_calibration", False), ("attach_enabled", False),
    ("grid_id", 2), ("lift_calibration_height", 0.03),
    ("gripper_closed", -0.30)])
def test_cycle_invalid_configuration_no_commands(lift_node, flag, value):
    node = lift_node
    goal = Goal()
    if flag == "grid_id":
        goal.request.grid_id = value
        goal.request.object_model = "pick_object_2"
    else:
        node.cfg[flag] = value
    if flag == "dry_run":
        node.dry_run = value
    node.goal_callback(goal.request)
    result = node.execute_callback(goal)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.attach_client.requests == []
    node.dry_run = False


@pytest.mark.parametrize("failure", [
    "center", "gripper", "distance", "attach", "ik", "plan_before",
    "send_unknown", "lift_execution", "lift_height", "object_height",
    "lower_plan", "lower_execution", "return_height", "detach"])
def test_cycle_failures_hold_or_release_only_when_safe(lift_node, failure):
    node = lift_node

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    if failure == "center":
        node.local_center_correction = lambda *a: (False, 3, 0.004)
    elif failure == "gripper":
        node.verify_closed_gripper = fail
    elif failure in (
            "distance", "lift_height", "object_height", "return_height"):
        original = node.entity_client.responder

        def entity(request):
            response = original(request)
            count = len(node.test_trajectories)
            if failure == "distance" and request.name.endswith("object_link"):
                response.state.pose.position.x += 0.1
            elif failure == "lift_height" and count == 2:
                response.state.pose.position.z += 0.02
            elif (failure == "object_height" and count == 2
                  and request.name.endswith("object_link")):
                response.state.pose.position.z -= 0.02
            elif failure == "return_height" and count == 3:
                response.state.pose.position.z += 0.01
            return response
        node.entity_client.responder = entity
    elif failure == "attach":
        node.attach_client.responder = lambda _: Attach.Response(success=False)
    elif failure == "ik":
        original = node.ik_client.responder

        def distant(request):
            response = original(request)
            response.solution.joint_state.position[0] += 0.3
            return response
        node.ik_client.responder = distant
    elif failure in ("plan_before", "lower_plan"):
        original = node.plan_joints

        def plan(*args, **kwargs):
            count = len(node.test_trajectories)
            if count == (1 if failure == "plan_before" else 2):
                return fail()
            return original(*args, **kwargs)
        node.plan_joints = plan
    elif failure == "detach":
        node.detach_client.responder = lambda _: Detach.Response(success=False)
    else:
        original = node.arm_client.send_goal_async

        def send(goal):
            count = len(node.test_trajectories)
            target = 2 if failure == "lower_execution" else 1
            if count == target:
                if failure == "send_unknown":
                    return fail()
                node.test_trajectories.append(goal.trajectory)
                return ready(NS(
                    accepted=True, get_result_async=lambda: ready(NS(
                        status=6, result=FollowJointTrajectory.Result(
                            error_code=-1)))))
            return original(goal)
        node.arm_client.send_goal_async = send
    result, goal = run_cycle(node)
    assert not result.success
    unsafe = failure in (
        "send_unknown", "lift_execution", "lift_height", "object_height",
        "lower_plan", "lower_execution", "return_height", "detach")
    assert node.attached == unsafe
    if unsafe:
        assert result.final_state == "recovery_required"
        assert node.test_commands == [0.15, -0.31]
        assert len(node.detach_client.requests) == int(failure == "detach")
        assert node.goal_callback(goal.request) == GoalResponse.REJECT
    else:
        assert "recovery completed" in result.message
        assert node.test_commands[-1] == 0.15
        assert len(node.test_trajectories) == 1
        assert len(node.detach_client.requests) == int(
            failure in ("ik", "plan_before"))
    assert "returning_home" not in goal.stages


def test_cycle_disabled_does_not_run_attached_direct_mode(lift_node):
    node = lift_node
    node.cfg["lift_calibration"] = False
    result, _ = run_cycle(node)
    assert not result.success
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.attach_client.requests == []


@pytest.mark.parametrize("endpoint", ["attach_client", "detach_client"])
def test_cycle_unknown_service_result_never_opens_gripper(lift_node, endpoint):
    node = lift_node
    node.cfg["service_timeout"] = 0.02
    requests = []
    pending = Future()
    client = getattr(node, endpoint)
    client.call_async = lambda request: (requests.append(request) or pending)
    result, _ = run_cycle(node)
    assert result.final_state == "recovery_required"
    assert node.test_commands == [0.15, -0.31]
    assert len(requests) == 1
    if endpoint == "attach_client":
        assert not node.attached
        assert node._pending_attach is pending
        assert len(node.test_trajectories) == 1
    else:
        assert node.attached
        assert node._pending_detach is pending
        assert len(node.test_trajectories) == 3


def test_cycle_latest_feedback_saved_for_ik_and_descent(lift_node):
    node = lift_node
    original = node.cycle_measurement
    latest = [0.72, 0.82, -0.59, 0.0, 1.34, 0.73]

    def measure(model, goal, label):
        result = original(model, goal, label)
        if label == "before_lift":
            node.joint_state_callback(seed_message(node, latest))
        return result
    node.cycle_measurement = measure
    result, _ = run_cycle(node)
    assert result.success
    request, = node.ik_client.requests
    assert list(request.ik_request.robot_state.joint_state.position) == latest
    assert list(node.test_trajectories[-1].points[-1].positions) == latest


def test_cycle_cancellation_in_air_never_releases(lift_node):
    node = lift_node
    original = node.cycle_measurement

    def measure(model, goal, label):
        result = original(model, goal, label)
        if label == "after_lift":
            goal.is_cancel_requested = True
        return result
    node.cycle_measurement = measure
    result, goal = run_cycle(node)
    assert result.final_state == "recovery_required"
    assert goal.state == "canceled"
    assert node.attached
    assert node.detach_client.requests == []
    assert node.test_commands == [0.15, -0.31]
    assert len(node.test_trajectories) == 2


@pytest.mark.parametrize("height", [0.045, 0.055])
def test_cycle_lift_measurement_bounds(lift_node, height):
    node = lift_node
    original = node.entity_client.responder

    def entity(request):
        response = original(request)
        if len(node.test_trajectories) == 2:
            response.state.pose.position.z = 0.4285 + height
        return response
    node.entity_client.responder = entity
    result, _ = run_cycle(node)
    assert result.success


def test_cycle_feedback_is_monotonic(lift_node):
    result, goal = run_cycle(lift_node)
    assert result.success
    assert all(a < b for a, b in zip(goal.progress, goal.progress[1:]))
    assert goal.progress[-1] == 1.0
    lowering = goal.stages.index("lowering_calibration")
    assert goal.progress[lowering] == pytest.approx(0.95)


def run_legacy_cycle_internally(node):
    goal = Goal()
    assert node._task_lock.acquire(blocking=False)
    return node.execute_callback(goal), goal


def test_regular_feedback_is_monotonic(node):
    result, goal = run_legacy_cycle_internally(node)
    assert result.success
    assert all(a < b for a, b in zip(goal.progress, goal.progress[1:]))
    assert goal.progress[-1] == 1.0


@pytest.mark.parametrize("values", [
    [0.0] * 5, [0.0] * 7, [float("nan")] * 6, [float("inf")] * 6,
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -3.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.8, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 2.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -3.2],
])
@pytest.mark.parametrize("model", ["red_bin", "blue_bin"])
@pytest.mark.parametrize("suffix", [
    "approach_joints", "left_release_joints",
    "center_release_joints", "right_release_joints"])
def test_calibration_joint_data_rejects_invalid_values(
        node, model, suffix, values):
    node.cfg[f"{model}.{suffix}"] = values
    with pytest.raises(ValueError, match=model):
        node._validate_parameters()


def test_stored_approaches_do_not_change_ordinary_motion(node):
    red = [1.5780840510, 0.8013771349, -0.9467428297,
           -0.0332285864, 1.7100726856, 1.6302872993]
    blue = [-1.5779212067, 0.8827475163, -1.0850132822,
            0.0335590192, 1.7794357656, -1.5141581248]
    node.cfg["red_bin.approach_joints"] = red
    node.cfg["blue_bin.approach_joints"] = blue
    node._validate_parameters()
    result, _ = run_legacy_cycle_internally(node)
    assert result.success
    for trajectory in node.test_trajectories:
        assert list(trajectory.points[-1].positions) not in (red, blue)


def test_motion_never_reads_slot_calibration(node):
    class GuardedConfig(dict):
        def __getitem__(self, key):
            if ("slot_offsets_x" in key or "center_world_xy" in key
                    or key.endswith("_release_joints")):
                pytest.fail("Motion must not read slot calibration")
            return super().__getitem__(key)

    node.cfg = GuardedConfig(node.cfg)
    result, _ = run_legacy_cycle_internally(node)
    assert result.success


@pytest.fixture
def bin_slot_node(node):
    server = node
    server.cfg.update({
        "bin_slot_calibration": True,
        "calibration_bin": "red",
        "calibration_slot": "left",
        "calibration_execute": True,
        "attach_enabled": False,
        "lift_calibration": False,
        "direct_pick_calibration": False,
        "local_center_calibration": False,
        "red_bin.approach_joints": [
            1.5780840510, 0.8013771349, -0.9467428297,
            -0.0332285864, 1.7100726856, 1.6302872993],
        "blue_bin.approach_joints": [
            -1.5779212067, 0.8827475163, -1.0850132822,
            0.0335590192, 1.7794357656, -1.5141581248],
    })
    parameter = ParameterValue(
        type=ParameterType.PARAMETER_BOOL, bool_value=True)
    server.move_group_parameters_client = Client(
        lambda _: GetParameters.Response(values=[parameter]))
    server.state_validity_client = Client(
        lambda _: GetStateValidity.Response(valid=True))
    link = [0.0, 0.18, 0.478]
    orientation = [0.1, -0.2, 0.3, 0.9273618495]

    def entity(request):
        response = GetEntityState.Response(success=True)
        response.state.pose.position.x = link[0]
        response.state.pose.position.y = link[1]
        response.state.pose.position.z = link[2]
        q = response.state.pose.orientation
        q.x, q.y, q.z, q.w = orientation
        return response

    server.entity_client.responder = entity
    original_send = server.arm_client.send_goal_async

    def send(goal):
        if server.ik_client.requests:
            target = server.ik_client.requests[-1].ik_request.pose_stamped.pose
            link[:] = [
                target.position.x, target.position.y, target.position.z]
        return original_send(goal)

    server.arm_client.send_goal_async = send
    server.test_bin_link = link
    server.test_bin_orientation = orientation
    return server


def run_bin_slot(node):
    goal = Goal()
    assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
    return node.execute_callback(goal), goal


def test_bin_slot_defaults_and_recorded_centers():
    config = (Path(__file__).resolve().parents[1]
              / "config" / "sorting_config.yaml")
    rclpy.init(args=["--ros-args", "--params-file", str(config)])
    server = PickActionServer()
    try:
        assert server.cfg["bin_slot_calibration"] is False
        assert server.cfg["calibration_bin"] == ""
        assert server.cfg["calibration_slot"] == ""
        assert server.cfg["calibration_execute"] is False
        assert server.cfg["bin_horizontal_tolerance"] == 0.004
        assert server.cfg["attach_enabled"] is False
        assert server.cfg["lift_calibration"] is False
        assert server.cfg["red_bin.center_release_joints"] == [
            1.5860181203, 0.8380505628, -0.7422356759,
            -0.0370827207, 1.4941800225, 1.6558392442]
        assert server.cfg["blue_bin.center_release_joints"] == [
            -1.5786027935, 0.8032457336, -0.6907313796,
            0.0336656462, 1.4543002999, -1.5214733686]
    finally:
        server.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("name,value", [
    ("attach_enabled", True),
    ("lift_calibration", True),
    ("direct_pick_calibration", True),
    ("local_center_calibration", True),
    ("calibration_bin", ""),
    ("calibration_bin", "green"),
    ("calibration_slot", ""),
    ("calibration_slot", "center"),
])
def test_bin_slot_parameter_guard_before_motion(bin_slot_node, name, value):
    node = bin_slot_node
    node.cfg[name] = value
    result, _ = run_bin_slot(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert "rejected before motion" in result.message
    assert node.test_trajectories == []
    assert node.plan_client.requests == []
    assert node.ik_client.requests == []


@pytest.mark.parametrize("color,slot,x,y", [
    ("red", "left", -0.040, 0.18),
    ("red", "right", 0.040, 0.18),
    ("blue", "left", -0.040, -0.18),
    ("blue", "right", 0.040, -0.18),
])
def test_bin_slot_selection_and_success(
        bin_slot_node, color, slot, x, y):
    node = bin_slot_node
    node.cfg["calibration_bin"] = color
    node.cfg["calibration_slot"] = slot
    node.test_bin_link[:] = [0.0, y, 0.478]
    before = dict(node.cfg)
    result, goal = run_bin_slot(node)
    assert result.success
    assert result.final_state == "bin_slot_calibration_completed"
    assert f"bin={color}" in result.message
    assert f"slot={slot}" in result.message
    assert f"{color}_bin.{slot}_release_joints:" in result.message
    assert node.test_bin_link == pytest.approx([x, y, 0.438])
    assert node.cfg == before
    assert len(node.test_trajectories) >= 3
    assert "closing_gripper" not in goal.stages
    assert "returning_home" not in goal.stages
    assert node.test_commands == []
    assert node.attach_client.requests == []
    assert node.detach_client.requests == []
    assert len(node.state_validity_client.requests) == (
        len(node.test_trajectories))


def test_bin_slot_segments_orientation_seed_and_limits(bin_slot_node):
    node = bin_slot_node
    result, _ = run_bin_slot(node)
    assert result.success
    assert node.ik_client.requests
    targets = [
        request.ik_request.pose_stamped.pose
        for request in node.ik_client.requests]
    previous = [0.0, 0.18, 0.478]
    for index, (request, target) in enumerate(zip(
            node.ik_client.requests, targets)):
        current = [target.position.x, target.position.y, target.position.z]
        horizontal = target.position.z == pytest.approx(0.478)
        limit = 0.02 if horizontal else 0.01
        assert math.dist(previous, current) <= limit + 1e-9
        q = target.orientation
        assert [q.x, q.y, q.z, q.w] == node.test_bin_orientation
        assert all(abs(value) < 1.0 for value in current)
        seed = request.ik_request.robot_state.joint_state
        assert list(seed.position) == node.test_measured[index]
        previous = current
    assert node.cfg["grasp_offset"] == [0.0175, 0.0164, 0.0225]
    for request in node.plan_client.requests:
        plan = request.motion_plan_request
        assert plan.max_velocity_scaling_factor <= 0.1
        assert plan.max_acceleration_scaling_factor <= 0.1


def test_bin_slot_plan_only_never_executes(bin_slot_node):
    node = bin_slot_node
    node.cfg["calibration_execute"] = False
    result, _ = run_bin_slot(node)
    assert result.success
    assert result.final_state == "calibration_plan_validated"
    assert len(node.plan_client.requests) == 1
    assert node.test_trajectories == []
    assert node.ik_client.requests == []
    assert node.entity_client.requests == []
    assert node.test_commands == []


def test_bin_slot_ik_distance_limit_stops(bin_slot_node):
    node = bin_slot_node
    original = node.ik_client.responder

    def distant(request):
        response = original(request)
        response.solution.joint_state.position[0] += 0.3
        return response

    node.ik_client.responder = distant
    result, _ = run_bin_slot(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert "0.3000000000rad" in result.message
    assert len(node.test_trajectories) == 1


def test_bin_slot_invalid_state_stops(bin_slot_node):
    node = bin_slot_node
    node.state_validity_client.responder = (
        lambda _: GetStateValidity.Response(valid=False))
    result, _ = run_bin_slot(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert len(node.test_trajectories) == 1
    assert node.ik_client.requests == []


def test_bin_slot_final_tolerance_stops(bin_slot_node):
    node = bin_slot_node
    original = node.entity_client.responder

    def inaccurate(request):
        response = original(request)
        position = response.state.pose.position
        if position.x == pytest.approx(-0.040) and position.z == pytest.approx(
                0.438):
            position.x += 0.003
        return response

    node.entity_client.responder = inaccurate
    result, _ = run_bin_slot(node)
    assert not result.success
    assert "0.0025000000m" in result.message
    assert node.test_commands == []


def test_bin_slot_execution_uncertainty_stops(bin_slot_node):
    node = bin_slot_node
    original = node.arm_client.send_goal_async

    def fail_second(goal):
        if len(node.test_trajectories) == 1:
            raise RuntimeError("unknown execution")
        return original(goal)

    node.arm_client.send_goal_async = fail_second
    result, _ = run_bin_slot(node)
    assert not result.success
    assert result.final_state == "recovery_required"
    assert len(node.test_trajectories) == 1
    assert node.test_commands == []
    assert node.attach_client.requests == []
    assert node.detach_client.requests == []


@pytest.mark.parametrize("step", [0.02, 0.02 - 1e-10])
def test_horizontal_step_at_or_below_limit_is_accepted(step):
    position = NS(x=0.0, y=0.0, z=0.5)
    target = PickActionServer.bin_segment_target(
        position, [step, 0.0, 0.5], 0.02, True)
    assert math.dist(target, [0.0, 0.0, 0.5]) == pytest.approx(step)
    assert not PickActionServer.exceeds_limit(step, 0.02)


def test_horizontal_step_beyond_numeric_epsilon_is_rejected():
    assert PickActionServer.exceeds_limit(0.02 + 2e-9, 0.02)
    assert not PickActionServer.exceeds_limit(0.02 + 0.5e-9, 0.02)


def test_vertical_step_at_limit_is_accepted_and_true_excess_is_rejected():
    position = NS(x=0.0, y=0.0, z=0.5)
    target = PickActionServer.bin_segment_target(
        position, [0.0, 0.0, 0.51], 0.01, False)
    assert math.dist(target, [0.0, 0.0, 0.5]) == pytest.approx(0.01)
    assert not PickActionServer.exceeds_limit(0.01, 0.01)
    assert PickActionServer.exceeds_limit(0.01 + 2e-9, 0.01)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_horizontal_055_is_recomputed_as_three_bounded_segments(direction):
    current = NS(x=0.0, y=0.18, z=0.478)
    final = [direction * 0.055, 0.18, 0.438]
    steps = []
    for _ in range(3):
        target = PickActionServer.bin_segment_target(
            current, final, 0.02, True)
        steps.append(math.dist(
            target, [current.x, current.y, current.z]))
        current = NS(x=target[0], y=target[1], z=target[2])
    assert current.x == pytest.approx(final[0])
    assert current.y == pytest.approx(final[1])
    assert len(steps) == 3
    assert all(
        not PickActionServer.exceeds_limit(step, 0.02)
        for step in steps)
    assert steps == pytest.approx([0.055 / 3] * 3)


@pytest.mark.parametrize(
    "residual", [0.0014774533, 0.0027258765, 0.004])
def test_horizontal_tolerance_enters_vertical_without_another_segment(
        bin_slot_node, residual):
    node = bin_slot_node
    pose = NS(position=NS(
        x=-0.040 + residual, y=0.18, z=0.478 + 0.004))
    node.query_entity_pose = lambda *args, **kwargs: pose
    calls = []
    node.execute_bin_segment = lambda *args: calls.append(args)
    node.run_bin_axis(
        [-0.040, 0.18, 0.438], NS(), 0.02, True, Goal(),
        "bin_horizontal")
    assert calls == []


def test_horizontal_error_ignores_high_position_z_drift():
    position = NS(x=-0.040, y=0.18, z=123.0)
    assert PickActionServer.bin_axis_error(
        position, [-0.040, 0.18, 0.438], True) == 0.0


def test_horizontal_post_measurement_in_tolerance_prevents_fourth_plan(
        bin_slot_node):
    node = bin_slot_node
    before = NS(position=NS(x=-0.035, y=0.18, z=0.478))
    after = NS(position=NS(
        x=-0.040 + 0.0014774533, y=0.18, z=0.479))
    node.query_entity_pose = lambda *args, **kwargs: before
    calls = []

    def execute(*args):
        calls.append(args)
        return None, after

    node.execute_bin_segment = execute
    node.run_bin_axis(
        [-0.040, 0.18, 0.438], NS(), 0.02, True, Goal(),
        "bin_horizontal")
    assert len(calls) == 1


def test_no_progress_above_axis_tolerance_is_rejected(bin_slot_node):
    node = bin_slot_node
    pose = NS(position=NS(x=-0.035, y=0.18, z=0.478))
    node.query_entity_pose = lambda *args, **kwargs: pose
    node.execute_bin_segment = lambda *args: (None, pose)
    with pytest.raises(CalibrationRequired, match="no measured progress"):
        node.run_bin_axis(
            [-0.040, 0.18, 0.438], NS(), 0.02, True, Goal(),
            "bin_horizontal")


def test_vertical_target_uses_exact_slot_xy_with_bounded_3d_step():
    position = NS(x=-0.040 + 0.0027258765, y=0.18, z=0.478)
    target = PickActionServer.bin_segment_target(
        position, [-0.040, 0.18, 0.438], 0.01, False)
    assert target[:2] == [-0.040, 0.18]
    assert math.dist(
        target, [position.x, position.y, position.z]) <= 0.01 + 1e-9


def test_vertical_tolerance_stops_without_noise_correction(bin_slot_node):
    node = bin_slot_node
    pose = NS(position=NS(x=-0.040, y=0.18, z=0.4405))
    node.query_entity_pose = lambda *args, **kwargs: pose
    calls = []
    node.execute_bin_segment = lambda *args: calls.append(args)
    node.run_bin_axis(
        [-0.040, 0.18, 0.438], NS(), 0.01, False, Goal(),
        "bin_vertical")
    assert calls == []


def test_bin_slot_requires_move_group_sim_time(bin_slot_node):
    node = bin_slot_node
    parameter = ParameterValue(
        type=ParameterType.PARAMETER_BOOL, bool_value=False)
    node.move_group_parameters_client.responder = (
        lambda _: GetParameters.Response(values=[parameter]))
    result, _ = run_bin_slot(node)
    assert not result.success
    assert "use_sim_time must be true" in result.message
    assert node.plan_client.requests == []
    assert node.test_trajectories == []


def test_bin_slot_planning_failure_stops(bin_slot_node):
    node = bin_slot_node
    original = node.plan_client.responder
    calls = 0

    def fail_second(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            response = GetMotionPlan.Response()
            response.motion_plan_response.error_code.val = -1
            return response
        return original(request)

    node.plan_client.responder = fail_second
    result, _ = run_bin_slot(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert len(node.test_trajectories) == 1
    assert len(node.ik_client.requests) == 1
    assert node.test_commands == []


def test_bin_slot_execute_rejected_in_dry_run(bin_slot_node):
    node = bin_slot_node
    node.dry_run = True
    result, _ = run_bin_slot(node)
    assert not result.success
    assert "requires dry_run=false" in result.message
    assert node.plan_client.requests == []
    assert node.test_trajectories == []
    node.dry_run = False


def test_bin_slot_rejects_low_approach_before_cartesian_motion(bin_slot_node):
    node = bin_slot_node
    node.test_bin_link[2] = 0.445
    result, _ = run_bin_slot(node)
    assert not result.success
    assert "safe height" in result.message
    assert len(node.test_trajectories) == 1
    assert node.ik_client.requests == []


@pytest.fixture
def transport_node(node):
    server = node
    config = yaml.safe_load((
        Path(__file__).resolve().parents[1] /
        "config/sorting_config.yaml").read_text())
    recorded = config["e3_pick_action_server"]["ros__parameters"]
    path_keys = [
        *(f"grid_{grid}.pick_joints" for grid in range(1, 7)),
        *(f"grid_{grid}.lift_joints" for grid in range(1, 7)),
        *(f"{bin_name}.{slot}_release_joints"
          for bin_name in ("red_bin", "blue_bin")
          for slot in ("left", "right")),
        "grid_1.lift_stage_1_joints", "grid_1.clearance_joints",
        "red_bin.approach_joints",
        "red_bin.center_intermediate_joints",
        "red_bin.center_release_joints",
        "blue_bin.approach_joints",
        "blue_bin.center_release_joints"]
    for key in path_keys:
        server.cfg[key] = list(recorded[key])
    server.cfg.update({
        "classification_transport_enabled": True,
        "lift_calibration_height": recorded["lift_calibration_height"],
        "attach_enabled": True,
        "direct_pick_calibration": False,
        "bin_slot_calibration": False,
        "lift_calibration": False,
        "local_center_calibration": False,
        "object_settle_time": 0.0,
    })
    sim_time = ParameterValue(
        type=ParameterType.PARAMETER_BOOL, bool_value=True)
    server.move_group_parameters_client = Client(
        lambda _: GetParameters.Response(values=[sim_time]))
    server.state_validity_client = Client(
        lambda _: GetStateValidity.Response(valid=True))

    link = [0.1275, 0.1216, 0.4285]
    object_position = [0.1275, 0.1216, 0.4285]

    def entity(request):
        response = GetEntityState.Response(success=True)
        position = response.state.pose.position
        if request.name == "mecharm_270::grasp_attach_link":
            position.x, position.y, position.z = link
            response.state.pose.orientation.w = 1.0
        elif request.name.startswith("pick_object_"):
            coordinates = link if server.attached else object_position
            position.x, position.y, position.z = coordinates
            response.state.pose.orientation.w = 1.0
        return response

    server.entity_client.responder = entity
    original_send = server.arm_client.send_goal_async
    executed_ik_count = 0

    def send(goal):
        nonlocal executed_ik_count
        if len(server.ik_client.requests) > executed_ik_count:
            target = server.ik_client.requests[-1].ik_request.pose_stamped.pose
            link[:] = [
                target.position.x, target.position.y, target.position.z]
            executed_ik_count = len(server.ik_client.requests)
        return original_send(goal)

    server.arm_client.send_goal_async = send
    server.test_transport_link = link
    server.test_transport_object = object_position
    server.test_transport_path_keys = path_keys
    return server


def run_transport(node, goal=None):
    goal = Goal() if goal is None else goal
    assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
    return node.execute_callback(goal), goal


def test_classification_switch_defaults_false_and_blocks_motion(node):
    assert node.cfg["classification_transport_enabled"] is False
    goal = Goal()
    assert node.goal_callback(goal.request) == GoalResponse.REJECT
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.attach_client.requests == []


@pytest.mark.parametrize("grid_id,object_class,object_model", [
    (0, "comb", "pick_object_0"),
    (7, "comb", "pick_object_7"),
    (2, "comb", "pick_object_2"),
    (5, "mouse", "pick_object_5"),
    (3, "comb", "pick_object_4"),
    (4, "mouse", "pick_object_3"),
])
def test_transport_invalid_combinations_rejected_before_motion(
        transport_node, grid_id, object_class, object_model):
    node = transport_node
    goal = Goal(object_class)
    goal.request.grid_id = grid_id
    goal.request.object_model = object_model
    assert node.goal_callback(goal.request) == GoalResponse.REJECT
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.attach_client.requests == []


@pytest.mark.parametrize("grid_id,object_class,destination", [
    (1, "comb", "red_bin"),
    (2, "mouse", "blue_bin"),
    (3, "comb", "red_bin"),
    (4, "mouse", "blue_bin"),
    (5, "comb", "red_bin"),
    (6, "mouse", "blue_bin"),
])
def test_transport_accepts_six_calibrated_goals_and_uses_yaml(
        transport_node, grid_id, object_class, destination):
    node = transport_node
    goal = Goal(object_class)
    goal.request.grid_id = grid_id
    goal.request.object_model = f"pick_object_{grid_id}"
    result, _ = run_transport(node, goal)
    assert result.success
    assert result.final_state == "sorting_completed"
    first_plan = node.plan_client.requests[0].motion_plan_request
    actual_pick = [
        item.position
        for item in first_plan.goal_constraints[0].joint_constraints]
    assert actual_pick == node.cfg[f"grid_{grid_id}.pick_joints"]
    assert node.attach_client.requests[0].model_name_2 == (
        goal.request.object_model)
    assert node.detach_client.requests[0].model_name_2 == (
        goal.request.object_model)
    assert destination in result.message
    queried = [request.name for request in node.entity_client.requests]
    assert goal.request.object_model in queried
    assert f"{goal.request.object_model}::object_link" in queried
    lift_index = goal.stages.index("moving_to_lift_joints")
    follow_index = goal.stages.index(
        "verifying_object_follow_after_lift")
    clearance_index = goal.stages.index("moving_to_clearance")
    assert lift_index < follow_index < clearance_index
    assert "moving_to_lift_stage_1" not in goal.stages


def test_transport_path_order_comes_from_yaml(transport_node):
    node = transport_node
    goal = Goal("comb")
    goal.request.grid_id = 3
    goal.request.object_model = "pick_object_3"
    result, goal = run_transport(node, goal)
    assert result.success
    assert result.final_state == "sorting_completed"
    expected_keys = [
        "grid_3.pick_joints", "grid_3.lift_joints",
        "grid_1.clearance_joints", "red_bin.approach_joints",
        "red_bin.center_intermediate_joints",
        "red_bin.center_release_joints",
        "red_bin.center_intermediate_joints",
        "red_bin.approach_joints"]
    all_targets = [
        [constraint.position for constraint in
         request.motion_plan_request.goal_constraints[0].joint_constraints]
        for request in node.plan_client.requests]
    stored = [node.cfg[key] for key in expected_keys]
    actual = [target for target in all_targets if target in stored]
    assert actual == stored
    assert node.cfg["grid_1.lift_stage_1_joints"] not in all_targets
    assert node.test_commands == [0.15, -0.31, 0.15]
    assert goal.stages[-1] == "sorting_completed"
    assert "returning_home" not in goal.stages
    assert node.attach_client.requests[0].joint_name == (
        node.detach_client.requests[0].joint_name)


def test_transport_local_center_uses_live_pose_without_offset(transport_node):
    node = transport_node
    original = node.entity_client.responder
    calls = 0

    def entity(request):
        nonlocal calls
        response = original(request)
        if request.name == "mecharm_270::grasp_attach_link" and calls == 0:
            response.state.pose.position.x -= 0.004
            calls += 1
        return response

    node.entity_client.responder = entity
    result, _ = run_transport(node)
    assert result.success
    local_request = next(
        request.ik_request for request in node.ik_client.requests
        if request.ik_request.pose_stamped.pose.position.x == pytest.approx(
            0.1275))
    assert local_request.pose_stamped.pose.position.x == pytest.approx(0.1275)
    assert node.cfg["grasp_offset"] == [0.0175, 0.0164, 0.0225]


def test_transport_attach_failure_never_detaches_or_moves_onward(
        transport_node):
    node = transport_node
    node.attach_client.responder = lambda _: Attach.Response(
        success=False, message="attach failed")
    goal = Goal("comb")
    goal.request.grid_id = 5
    goal.request.object_model = "pick_object_5"
    result, _ = run_transport(node, goal)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert node.attach_client.requests[0].model_name_2 == "pick_object_5"
    assert node.detach_client.requests == []
    assert len(node.test_trajectories) == 1


def test_transport_post_attach_plan_failure_keeps_grip_and_attachment(
        transport_node):
    node = transport_node
    original = node.plan_client.responder

    def fail_lift(request):
        if len(node.plan_client.requests) == 2:
            return GetMotionPlan.Response()
        return original(request)

    node.plan_client.responder = fail_lift
    result, _ = run_transport(node)
    assert not result.success
    assert result.final_state == "recovery_required"
    assert node.attached
    assert "attached=True" in result.message
    assert node.detach_client.requests == []
    assert node.test_commands == [0.15, -0.31]


def test_transport_detach_failure_never_opens_gripper(transport_node):
    node = transport_node
    node.detach_client.responder = lambda _: Detach.Response(
        success=False, message="detach failed")
    result, _ = run_transport(node)
    assert not result.success
    assert result.final_state == "recovery_required"
    assert node.attached
    assert node.test_commands == [0.15, -0.31]


def test_transport_detach_success_precedes_open_and_retreat(transport_node):
    node = transport_node
    events = []
    original_detach = node.detach_client.responder
    original_publish = node.gripper.publish
    original_plan = node.plan_client.responder

    def detach(request):
        events.append("detach")
        return original_detach(request)

    def publish(message):
        events.append(f"gripper:{message.data[0]}")
        original_publish(message)

    def plan(request):
        target = [
            item.position for item in
            request.motion_plan_request.goal_constraints[0].joint_constraints]
        events.append("plan:" + str(target))
        return original_plan(request)

    node.detach_client.responder = detach
    node.gripper.publish = publish
    node.plan_client.responder = plan
    result, _ = run_transport(node)
    assert result.success
    detach_index = events.index("detach")
    open_index = events.index("gripper:0.15", 1)
    assert detach_index < open_index
    assert events[open_index + 1].startswith("plan:")


@pytest.fixture
def grid_pick_node(bin_slot_node):
    server = bin_slot_node
    server.cfg.update({
        "grid_pick_calibration": True,
        "classification_transport_enabled": False,
        "direct_pick_calibration": False,
        "bin_slot_calibration": False,
        "attach_enabled": False,
        "lift_calibration": False,
        "local_center_calibration": False,
    })
    config = yaml.safe_load((
        Path(__file__).resolve().parents[1] /
        "config/sorting_config.yaml").read_text())
    recorded = config["e3_pick_action_server"]["ros__parameters"]
    server.cfg["grid_1.clearance_joints"] = list(
        recorded["grid_1.clearance_joints"])
    link = [0.0, 0.0, 0.478]
    object_xyz = [0.040, 0.180, 0.438]
    orientation = [0.1, -0.2, 0.3, 0.9273618495]

    def entity(request):
        response = GetEntityState.Response(success=True)
        xyz = (
            link if request.name == "mecharm_270::grasp_attach_link"
            else object_xyz)
        position = response.state.pose.position
        position.x, position.y, position.z = xyz
        q = response.state.pose.orientation
        q.x, q.y, q.z, q.w = orientation
        return response

    server.entity_client.responder = entity
    original_send = server.arm_client.send_goal_async

    def send(goal):
        if server.ik_client.requests:
            target = server.ik_client.requests[-1].ik_request.pose_stamped.pose
            link[:] = [
                target.position.x, target.position.y, target.position.z]
        return original_send(goal)

    server.arm_client.send_goal_async = send
    server.test_grid_link = link
    server.test_grid_object = object_xyz
    server.test_grid_orientation = orientation
    return server


def grid_goal(grid_id=2, model=None):
    goal = Goal()
    goal.request.grid_id = grid_id
    goal.request.object_model = model or f"pick_object_{grid_id}"
    return goal


def run_grid_pick(node, goal=None):
    goal = grid_goal() if goal is None else goal
    assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
    return node.execute_callback(goal), goal


def test_grid_pick_calibration_defaults_false():
    config = (Path(__file__).resolve().parents[1]
              / "config" / "sorting_config.yaml")
    data = yaml.safe_load(config.read_text())
    params = data["e3_pick_action_server"]["ros__parameters"]
    assert params["grid_pick_calibration"] is False


@pytest.mark.parametrize("name,value", [
    ("classification_transport_enabled", True),
    ("direct_pick_calibration", True),
    ("bin_slot_calibration", True),
    ("attach_enabled", True),
    ("lift_calibration", True),
])
def test_grid_pick_guard_rejects_before_motion(
        grid_pick_node, name, value):
    node = grid_pick_node
    node.cfg[name] = value
    result, _ = run_grid_pick(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.plan_client.requests == []
    assert node.ik_client.requests == []


@pytest.mark.parametrize("grid_id,model,accepted", [
    (1, "pick_object_1", False),
    (2, "pick_object_2", True),
    (3, "pick_object_3", True),
    (4, "pick_object_4", True),
    (5, "pick_object_5", True),
    (6, "pick_object_6", True),
    (2, "pick_object_3", False),
    (6, "pick_object_5", False),
])
def test_grid_pick_range_mapping_and_precedes_transport_check(
        grid_pick_node, grid_id, model, accepted):
    node = grid_pick_node
    goal = grid_goal(grid_id, model)
    response = node.goal_callback(goal.request)
    if model != f"pick_object_{grid_id}":
        assert response == GoalResponse.REJECT
        assert node.test_commands == []
        assert node.test_trajectories == []
        return
    assert response == GoalResponse.ACCEPT
    result = node.execute_callback(goal)
    assert result.success is accepted
    if accepted:
        assert result.final_state == "grid_pick_calibration_completed"
    else:
        assert result.final_state == "calibration_required"
        assert node.test_commands == []
        assert node.test_trajectories == []


def test_grid_pick_path_limits_seeds_orientation_and_yaml(grid_pick_node):
    node = grid_pick_node
    result, goal = run_grid_pick(node)
    assert result.success
    assert result.final_state == "grid_pick_calibration_completed"
    assert "grid_2.pick_joints: [" in result.message
    first = node.plan_client.requests[0].motion_plan_request
    first_values = [
        item.position
        for item in first.goal_constraints[0].joint_constraints]
    assert first_values == node.cfg["grid_1.clearance_joints"]
    ik_requests = [request.ik_request for request in node.ik_client.requests]
    assert ik_requests
    fixed_q = node.test_grid_orientation
    for request in ik_requests:
        pose = request.pose_stamped.pose
        assert [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w] == pytest.approx(fixed_q)
        assert request.robot_state.joint_state.name == node.cfg["joint_names"]
    for previous, current in zip(ik_requests, ik_requests[1:]):
        expected = [
            value + 0.015
            for value in previous.robot_state.joint_state.position]
        assert current.robot_state.joint_state.position == pytest.approx(
            expected)
    poses = [request.pose_stamped.pose.position for request in ik_requests]
    # Segment stages are reflected in order: horizontal preserves safe Z,
    # vertical then reaches the un-offset object coordinates.
    assert all(p.z == pytest.approx(0.478) for p in poses[:-5])
    for before, after in zip(
            [[0.0, 0.0, 0.478]] + [
                [p.x, p.y, p.z] for p in poses[:-1]],
            [[p.x, p.y, p.z] for p in poses]):
        limit = 0.02 if after[2] == pytest.approx(0.478) else 0.01
        assert math.dist(before, after) <= limit + 1e-9
    final = poses[-1]
    assert [final.x, final.y, final.z] == pytest.approx(
        node.test_grid_object)
    assert node.cfg["grasp_offset"] == [0.0175, 0.0164, 0.0225]
    assert node.test_commands == [0.15]
    assert node.attach_client.requests == []
    assert node.detach_client.requests == []
    assert node.cfg["home_joints"] not in [
        list(trajectory.points[-1].positions)
        for trajectory in node.test_trajectories]
    assert goal.stages[-1] == "grid_pick_calibration_completed"


def test_grid_pick_invalid_state_stops_open_and_unattached(grid_pick_node):
    node = grid_pick_node
    node.state_validity_client.responder = (
        lambda _: GetStateValidity.Response(valid=False))
    result, _ = run_grid_pick(node)
    assert not result.success
    assert result.final_state == "calibration_required"
    assert node.test_commands == [0.15]
    assert len(node.test_trajectories) == 1
    assert node.attach_client.requests == []
    assert node.detach_client.requests == []


def adaptive_ik_response(node, request, error=1, joint_delta=0.01):
    response = GetPositionIK.Response()
    response.error_code.val = error
    if error == 1:
        response.solution.joint_state.name = list(
            reversed(node.cfg["joint_names"]))
        seed = request.ik_request.robot_state.joint_state.position
        response.solution.joint_state.position = [
            value + joint_delta for value in reversed(seed)]
    return response


def run_adaptive_horizontal(node, distance):
    node.test_grid_link[:] = [0.0, 0.0, 0.478]
    return node.run_grid_pick_horizontal(
        [distance, 0.0, 0.478], None)


def requested_horizontal_x(node):
    return [
        request.ik_request.pose_stamped.pose.position.x
        for request in node.ik_client.requests]


def test_grid_horizontal_minus_31_halves_then_executes(grid_pick_node):
    node = grid_pick_node
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return adaptive_ik_response(node, request, error=-31)
        return adaptive_ik_response(node, request)

    node.ik_client.responder = responder
    run_adaptive_horizontal(node, 0.01844)
    assert requested_horizontal_x(node)[:2] == pytest.approx(
        [0.01844, 0.00922])
    assert len(node.plan_client.requests) == len(node.test_trajectories)
    assert len(node.plan_client.requests) == 2


def test_grid_horizontal_seed_distance_rejection_is_never_planned(
        grid_pick_node):
    node = grid_pick_node
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        delta = 0.18 if calls < 3 else 0.01
        return adaptive_ik_response(node, request, joint_delta=delta)

    node.ik_client.responder = responder
    run_adaptive_horizontal(node, 0.01844)
    assert requested_horizontal_x(node)[:3] == pytest.approx(
        [0.01844, 0.00922, 0.00461])
    # Both excessive branches were rejected before planning.
    assert len(node.plan_client.requests) == len(node.test_trajectories)
    assert len(node.plan_client.requests) == 2
    assert node.test_grid_link[0] == pytest.approx(0.01844)


def test_grid_horizontal_final_tolerance_is_not_a_step_floor(
        grid_pick_node):
    node = grid_pick_node
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return adaptive_ik_response(node, request, joint_delta=0.18)
        return adaptive_ik_response(node, request)

    node.ik_client.responder = responder
    run_adaptive_horizontal(node, 0.00436)
    assert requested_horizontal_x(node) == pytest.approx([0.00436, 0.00218])
    assert len(node.plan_client.requests) == 1
    assert len(node.test_trajectories) == 1


def test_grid_horizontal_non_branch_ik_error_does_not_retry(grid_pick_node):
    node = grid_pick_node
    node.ik_client.responder = (
        lambda request: adaptive_ik_response(node, request, error=-1))
    with pytest.raises(RuntimeError, match="error_code=-1"):
        run_adaptive_horizontal(node, 0.01844)
    assert requested_horizontal_x(node) == pytest.approx([0.01844])
    assert node.plan_client.requests == []
    assert node.test_trajectories == []


def test_grid_horizontal_next_segment_resets_candidate(grid_pick_node):
    node = grid_pick_node
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return adaptive_ik_response(node, request, error=-31)
        return adaptive_ik_response(node, request)

    node.ik_client.responder = responder
    run_adaptive_horizontal(node, 0.040)
    assert requested_horizontal_x(node) == pytest.approx(
        [0.020, 0.010, 0.025, 0.040])
    assert len(node.plan_client.requests) == 3
    assert len(node.test_trajectories) == 3


def test_grid_horizontal_00461_compliant_solution_executes(grid_pick_node):
    node = grid_pick_node
    run_adaptive_horizontal(node, 0.00461)
    assert requested_horizontal_x(node) == pytest.approx([0.00461])
    assert len(node.plan_client.requests) == 1
    assert len(node.test_trajectories) == 1


def test_grid_horizontal_fifth_rejection_stops_without_planning(
        grid_pick_node):
    node = grid_pick_node
    node.ik_client.responder = (
        lambda request: adaptive_ik_response(node, request, error=-31))
    with pytest.raises(CalibrationRequired, match="5 candidate attempts"):
        run_adaptive_horizontal(node, 0.01844)
    assert requested_horizontal_x(node) == pytest.approx([
        0.01844, 0.00922, 0.00461, 0.002305, 0.0011525])
    assert node.plan_client.requests == []
    assert node.test_trajectories == []


def test_grid_horizontal_retries_hold_then_next_segment_refreshes_orientation(
        grid_pick_node):
    node = grid_pick_node
    first = [0.1, -0.2, 0.3, 0.9273618495]
    second = [-0.1, 0.2, -0.3, 0.9273618495]
    node.test_grid_orientation[:] = first
    original_send = node.arm_client.send_goal_async
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return adaptive_ik_response(node, request, error=-31)
        return adaptive_ik_response(node, request)

    def send(goal):
        result = original_send(goal)
        if node.test_grid_link[0] >= 0.009:
            node.test_grid_orientation[:] = second
        return result

    node.ik_client.responder = responder
    node.arm_client.send_goal_async = send
    run_adaptive_horizontal(node, 0.040)
    orientations = [[
        request.ik_request.pose_stamped.pose.orientation.x,
        request.ik_request.pose_stamped.pose.orientation.y,
        request.ik_request.pose_stamped.pose.orientation.z,
        request.ik_request.pose_stamped.pose.orientation.w]
        for request in node.ik_client.requests]
    assert orientations[0] == pytest.approx(first)
    assert orientations[1] == pytest.approx(first)
    assert all(value == pytest.approx(second)
               for value in orientations[2:])


def test_grid_local_center_uses_post_descent_measured_orientation(
        grid_pick_node):
    node = grid_pick_node
    initial = [0.1, -0.2, 0.3, 0.9273618495]
    final = [-0.1, 0.2, -0.3, 0.9273618495]
    node.test_grid_orientation[:] = initial
    original_send = node.arm_client.send_goal_async
    captured = []

    def send(goal):
        result = original_send(goal)
        if node.test_grid_link[2] < 0.478:
            node.test_grid_orientation[:] = final
        return result

    def local_center(model, seed, goal_handle, fixed_orientation=None):
        captured.append([
            fixed_orientation.x, fixed_orientation.y,
            fixed_orientation.z, fixed_orientation.w])
        node.test_grid_link[:] = node.test_grid_object
        return True, 0, 0.0

    node.arm_client.send_goal_async = send
    node.local_center_correction = local_center
    result, _ = run_grid_pick(node)
    assert result.success
    assert captured == [pytest.approx(final)]


def test_grid_horizontal_tiny_accepted_step_is_not_inherited(grid_pick_node):
    node = grid_pick_node
    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return adaptive_ik_response(node, request, error=-31)
        return adaptive_ik_response(node, request)

    node.ik_client.responder = responder
    run_adaptive_horizontal(node, 0.018)
    assert requested_horizontal_x(node) == pytest.approx([
        0.018, 0.009, 0.0045, 0.00225, 0.018])
    assert len(node.plan_client.requests) == 2
    assert len(node.test_trajectories) == 2


def test_grid_horizontal_no_progress_still_stops(grid_pick_node):
    node = grid_pick_node
    original_send = node.arm_client.send_goal_async

    def send_without_cartesian_progress(goal):
        result = original_send(goal)
        node.test_grid_link[:] = [0.0, 0.0, 0.478]
        return result

    node.arm_client.send_goal_async = send_without_cartesian_progress
    with pytest.raises(CalibrationRequired, match="no measured progress"):
        run_adaptive_horizontal(node, 0.018)
    assert len(node.plan_client.requests) == 1
    assert len(node.test_trajectories) == 1


@pytest.mark.parametrize("grid", range(1, 7))
def test_transport_uses_matching_fixed_lift(transport_node, grid):
    node = transport_node
    goal = Goal("comb" if grid % 2 else "mouse")
    goal.request.grid_id = grid
    goal.request.object_model = f"pick_object_{grid}"
    result, goal = run_transport(node, goal)
    assert result.success
    targets = [
        [item.position for item in request.motion_plan_request
         .goal_constraints[0].joint_constraints]
        for request in node.plan_client.requests]
    assert targets[:3] == [
        node.cfg[f"grid_{grid}.pick_joints"],
        node.cfg[f"grid_{grid}.lift_joints"],
        node.cfg["grid_1.clearance_joints"]]
    assert node.ik_client.requests == []
    assert goal.stages.index("moving_to_lift_joints") < (
        goal.stages.index("verifying_object_follow_after_lift"))


@pytest.mark.parametrize("failure", ["planning", "execution", "state"])
def test_fixed_lift_failure_holds_attachment(transport_node, failure):
    node = transport_node
    if failure == "planning":
        original = node.plan_client.responder

        def plan(request):
            if len(node.plan_client.requests) == 2:
                return GetMotionPlan.Response()
            return original(request)
        node.plan_client.responder = plan
    elif failure == "execution":
        original = node.execute_trajectory

        def execute(trajectory, goal):
            if node.attached:
                raise TimeoutError("lift execution uncertain")
            return original(trajectory, goal)
        node.execute_trajectory = execute
    else:
        node.state_validity_client.responder = (
            lambda _: GetStateValidity.Response(valid=False))
    result, goal = run_transport(node)
    assert result.final_state == "recovery_required"
    assert node.attached
    assert not node.detach_client.requests
    assert node.test_commands == [0.15, -0.31]
    assert "moving_to_clearance" not in goal.stages
    assert len(node.test_trajectories) == (2 if failure == "state" else 1)


@pytest.mark.parametrize("invalid", [[], [0.0], [float("nan")] * 6,
                                     [100.0] * 6])
def test_invalid_fixed_lift_rejected_before_motion(transport_node, invalid):
    node = transport_node
    node.cfg["grid_1.lift_joints"] = invalid
    result, _ = run_transport(node)
    assert not result.success
    assert node.test_trajectories == []
    assert node.test_commands == []


@pytest.mark.parametrize("grid, expected", [
    (1, [0.8028159892, 0.6055746569, -0.670333503,
         0.0003421158, 1.6267768699, 0.8106297914]),
    (2, [0.6607262926, 0.962752097, -1.2705171831,
         0.0147027684, 1.8008709198, 0.6831425012]),
    (3, [0.0349721737, 0.1044865427, 0.0144229312,
         -0.0408897965, 1.4933222094, 0.0735970607]),
    (4, [-0.0144077556, 0.5373394226, -0.5901479584,
         0.0028369126, 1.5736648222, 0.0480247021]),
    (5, [-0.8475973797, 0.6088910297, -0.6532995754,
         0.0418098551, 1.6059558881, -0.7894696733]),
    (6, [-0.6566222784, 0.959148394, -1.251972088,
         -0.0188379429, 1.7936285878, -0.6882079693]),
])
def test_fixed_lift_yaml_values_and_limits(transport_node, grid, expected):
    values = transport_node.cfg[f"grid_{grid}.lift_joints"]
    assert values == expected
    assert len(values) == 6
    assert all(math.isfinite(value) for value in values)
    transport_node.validate_joint_values("lift", values)


@pytest.mark.parametrize("distance", [0.26, 0.30])
def test_local_ik_updated_limit_accepts_valid_branch(node, distance):
    from geometry_msgs.msg import PoseStamped
    seed = node.current_ik_seed(None)

    def ik(request):
        response = GetPositionIK.Response()
        response.error_code.val = 1
        response.solution.joint_state.name = list(seed.name)
        values = list(seed.position)
        values[0] += distance
        response.solution.joint_state.position = values
        return response

    node.ik_client.responder = ik
    pose = PoseStamped()
    pose.pose.orientation.w = 1.0
    result = node.solve_ik(
        pose, None, reference_seed=seed,
        max_distance=node.cfg["local_ik_seed_max_distance"])
    assert result[0] - seed.position[0] == pytest.approx(distance)


@pytest.mark.parametrize("value", [0.0, -0.001, float("nan"),
                                   float("inf"), 0.0121])
def test_local_displacement_parameter_validation(node, value):
    node.cfg["local_center_max_displacement"] = value
    with pytest.raises(ValueError, match="local_center_max_displacement"):
        node._validate_parameters()


def test_local_displacement_default_and_yaml(node):
    assert node.cfg["local_center_max_displacement"] == 0.012
    config = yaml.safe_load((
        Path(__file__).resolve().parents[1] /
        "config/sorting_config.yaml").read_text())
    assert config["e3_pick_action_server"]["ros__parameters"][
        "local_center_max_displacement"] == 0.012


@pytest.mark.parametrize("grid,destination,slot", [
    (1, "red_bin", "left"), (2, "blue_bin", "left"),
    (3, "red_bin", "center"), (4, "blue_bin", "center"),
    (5, "red_bin", "right"), (6, "blue_bin", "right")])
def test_transport_exact_slot_path_and_reverse(transport_node,
                                               grid, destination, slot):
    node = transport_node
    goal = Goal("comb" if grid % 2 else "mouse")
    goal.request.grid_id = grid
    goal.request.object_model = f"pick_object_{grid}"
    result, goal = run_transport(node, goal)
    assert result.success
    keys = [
        f"grid_{grid}.pick_joints", f"grid_{grid}.lift_joints",
        "grid_1.clearance_joints", f"{destination}.approach_joints"]
    intermediate = grid == 3
    if intermediate:
        keys.append("red_bin.center_intermediate_joints")
    keys.append(f"{destination}.{slot}_release_joints")
    if intermediate:
        keys.append("red_bin.center_intermediate_joints")
    keys.append(f"{destination}.approach_joints")
    actual = [
        [c.position for c in request.motion_plan_request
         .goal_constraints[0].joint_constraints]
        for request in node.plan_client.requests]
    assert actual == [node.cfg[key] for key in keys]
    assert ("moving_to_destination_intermediate" in goal.stages) == (
        intermediate)
    assert ("retreating_to_destination_intermediate" in goal.stages) == (
        intermediate)
    assert f"{destination} {slot}" in result.message


@pytest.mark.parametrize("grid", range(1, 7))
def test_missing_selected_release_rejected_before_motion(transport_node, grid):
    node = transport_node
    goal = Goal("comb" if grid % 2 else "mouse")
    goal.request.grid_id = grid
    goal.request.object_model = f"pick_object_{grid}"
    destination = "red_bin" if grid % 2 else "blue_bin"
    slot = ("left", "center", "right")[(grid - 1) // 2]
    node.cfg[f"{destination}.{slot}_release_joints"] = []
    result, _ = run_transport(node, goal)
    assert not result.success
    assert node.test_commands == []
    assert node.test_trajectories == []
    assert node.attach_client.requests == []


def test_local_parameter_defaults_match_formal_yaml(node):
    config = yaml.safe_load((
        Path(__file__).resolve().parents[1] /
        "config/sorting_config.yaml").read_text())
    parameters = config["e3_pick_action_server"]["ros__parameters"]
    for name in ("local_center_tolerance", "local_center_max_displacement",
                 "local_ik_seed_max_distance"):
        assert node.cfg[name] == parameters[name]
    node._validate_parameters()


def test_prepare_observation_success(transport_node):
    from std_srvs.srv import Trigger
    node = transport_node
    response = node.prepare_observation(Trigger.Request(), Trigger.Response())
    assert response.success
    assert node.test_commands == [0.15]
    assert len(node.test_trajectories) == 1
    target = node.plan_client.requests[0].motion_plan_request
    actual = [c.position for c in
              target.goal_constraints[0].joint_constraints]
    assert actual == (
        node.cfg["red_bin.approach_joints"])
    assert len(node.state_validity_client.requests) == 2
    assert not node.attach_client.requests
    assert not node._task_lock.locked()


@pytest.mark.parametrize("condition", [
    "busy", "attached", "pending_attach", "pending_detach",
    "stale", "planning", "invalid_target", "execution"])
def test_prepare_observation_safe_failure(transport_node, condition):
    from std_srvs.srv import Trigger
    node = transport_node
    if condition == "busy":
        node._task_lock.acquire()
    elif condition == "attached":
        node._attached_model = "pick_object_1"
    elif condition == "pending_attach":
        node._pending_attach = object()
    elif condition == "pending_detach":
        node._pending_detach = object()
    elif condition == "stale":
        node._joint_sample = None
        node.cfg["joint_state_timeout"] = 0.01
    elif condition == "planning":
        node.plan_client.responder = lambda _: GetMotionPlan.Response()
    elif condition == "invalid_target":
        node.state_validity_client.responder = (
            lambda _: GetStateValidity.Response(valid=False))
    else:
        def fail(*args):
            raise TimeoutError("uncertain execution")
        node.execute_trajectory = fail
    response = node.prepare_observation(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert not node.test_trajectories
    assert not node.attach_client.requests
    if condition in ("busy", "attached", "pending_attach",
                     "pending_detach", "stale"):
        assert not node.test_commands
    if condition == "busy":
        assert node._task_lock.locked()
        node._task_lock.release()
    else:
        assert not node._task_lock.locked()
