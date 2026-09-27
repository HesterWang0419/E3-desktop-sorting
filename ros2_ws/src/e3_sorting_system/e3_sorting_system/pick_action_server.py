#!/usr/bin/env python3
"""MoveIt-backed pick and sort, with an actuator-free dry-run mode."""

import copy
import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from e3_sorting_interfaces.action import PickAndSort
from gazebo_model_attachment_plugin_msgs.srv import Attach, Detach
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan, GetPositionIK, GetStateValidity
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import (
    ActionClient, ActionServer, CancelResponse, GoalResponse)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


# Position bounds in the shared mecharm_270_m5_sim.urdf.xacro.
# These bounds validate stored calibration data, not trajectory safety.
NUMERIC_EPSILON = 1e-9
BIN_FINAL_POSITION_TOLERANCE = 0.0025


APPROACH_JOINT_LIMITS = {
    "joint1_to_base": (-2.792527, 2.792527),
    "joint2_to_joint1": (-1.3089, 2.0943),
    "joint3_to_joint2": (-3.0543, 1.1344),
    "joint4_to_joint3": (-2.7052, 2.7052),
    "joint5_to_joint4": (-2.0071, 2.0071),
    "joint6_to_joint5": (-3.14, 3.14),
}


def calibration_slots(cfg):
    """Validate data only; return independent world XY slots in fixed order.

    The platforms are 0.18 by 0.10 m. The object footprint is 0.028 m
    square. Slot offsets run along local X at the platform Y center.
    """
    slots = {}
    offsets_seen = []
    centers_seen = []
    for model in ("red_bin", "blue_bin"):
        offsets = cfg[f"{model}.slot_offsets_x"]
        center = cfg[f"{model}.center_world_xy"]
        release_z = cfg[f"{model}.slot_release_z"]
        if not math.isfinite(release_z):
            raise ValueError(f"{model}: slot release z must be finite")
        if any(offsets is other for other in offsets_seen):
            raise ValueError("Red/blue slot offsets must be independent")
        if any(center is other for other in centers_seen):
            raise ValueError("Red/blue platform centers must be independent")
        offsets_seen.append(offsets)
        centers_seen.append(center)
        if len(offsets) != 3 or not all(math.isfinite(x) for x in offsets):
            raise ValueError(f"{model}: exactly three finite slot offsets")
        if len(center) != 2 or not all(math.isfinite(v) for v in center):
            raise ValueError(f"{model}: center needs two finite coordinates")
        if any(b - a <= 0.028 for a, b in zip(offsets, offsets[1:])):
            raise ValueError(
                f"{model}: ordered slot spacing must exceed 0.028")
        for x in offsets:
            for coordinate, half_size in zip((x, 0.0), (0.09, 0.05)):
                if half_size - abs(coordinate) - 0.014 <= 0:
                    raise ValueError(
                        f"{model}: footprint exceeds platform boundary")
        slots[model] = [[center[0] + x, center[1]] for x in offsets]
    return slots


class TaskCancelled(RuntimeError):
    """The client requested cancellation."""


class CalibrationRequired(RuntimeError):
    """Stop at the grasp pose without creating an attachment."""


class UnsupportedGoal(RuntimeError):
    """The restricted transport mode does not support this goal."""


class NoIKSolution(RuntimeError):
    """MoveIt explicitly reported NO_IK_SOLUTION."""


class IKSeedDistanceExceeded(RuntimeError):
    """A valid IK response selected a branch too far from its seed."""

    def __init__(self, distance, limit):
        self.distance = distance
        self.limit = limit
        super().__init__(
            f"IK branch rejected: distance={distance:.10f}rad exceeds "
            f"{limit:.10f}rad")


class PickActionServer(Node):
    """Serialise robot tasks while allowing concurrent ROS callbacks."""

    def __init__(self):
        super().__init__("e3_pick_action_server")
        defaults = {
            "action_name": "/e3/pick_and_sort", "dry_run": True,
            "feedback_delay": 0.4,
            "ik_avoid_collisions": False,
            "ik_seed_max_distance": 1.0,
            "attach_enabled": False,
            "direct_pick_calibration": False,
            "lift_calibration": False,
            "lift_calibration_height": 0.05,
            "return_height_tolerance": 0.005,
            "object_settle_time": 1.0,
            "local_center_calibration": False,
            "classification_transport_enabled": False,
            "grid_pick_calibration": False,
            "bin_slot_calibration": False,
            "calibration_bin": "",
            "calibration_slot": "",
            "calibration_execute": False,
            "bin_horizontal_tolerance": 0.004,
            "local_center_tolerance": 0.0035,
            "local_center_max_iterations": 3,
            "local_center_max_displacement": 0.012,
            "local_ik_seed_max_distance": 0.40,
            "attach_max_distance": 0.02,
            "gripper_position_tolerance": 0.03,
            "joint_states_topic": "/joint_states",
            "joint_state_timeout": 5.0,
            "joint_state_max_age": 1.0,
            "group_name": "mecharm_arm", "ik_link_name": "grasp_attach_link",
            "frame_id": "world", "ik_service": "/compute_ik",
            "planning_service": "/plan_kinematic_path",
            "planning_pipeline_id": "ompl",
            "planner_id": "RRTConnect",
            "entity_service": "/gazebo/get_entity_state",
            "attach_service": "/gazebo/attach",
            "detach_service": "/gazebo/detach",
            "arm_action": "/arm_controller/follow_joint_trajectory",
            "gripper_topic": "/gripper_position_controller/commands",
            "joint_names": [
                "joint1_to_base", "joint2_to_joint1", "joint3_to_joint2",
                "joint4_to_joint3", "joint5_to_joint4", "joint6_to_joint5"],
            "grasp_offset": [0.0175, 0.0164, 0.0225],
            "pregrasp_height": 0.029,
            "grasp_orientation": [0.042, -0.046, -0.001, 0.998],
            "release_offset": [0.0, 0.0, 0.124],
            "release_orientation": [0.000682, -0.702836, -0.000499, 0.711351],
            "home_joints": [0.0] * 6,
            "service_timeout": 10.0, "ik_timeout": 5.0,
            "planning_time": 5.0, "planning_attempts": 5,
            "joint_tolerance": 0.01, "velocity_scaling": 0.2,
            "acceleration_scaling": 0.2,
            "execution_timeout": 60.0, "cancel_timeout": 5.0,
            "gripper_open": 0.15, "gripper_closed": -0.31,
            "gripper_wait": 2.0, "robot_model": "mecharm_270",
            "robot_link": "grasp_attach_link", "object_link": "object_link",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.cfg = {name: self.get_parameter(name).value for name in defaults}
        for name, legacy in (
                ("open_gripper_position", "gripper_open"),
                ("close_gripper_position", "gripper_closed"),
                ("attach_distance_limit", "attach_max_distance")):
            self.declare_parameter(name, self.cfg[legacy])
            self.cfg[name] = self.get_parameter(name).value
            self.cfg[legacy] = self.cfg[name]
        for grid in range(1, 7):
            for suffix in ("pick_joints", "lift_joints"):
                name = f"grid_{grid}.{suffix}"
                self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.cfg[name] = self.get_parameter_or(
                    name, rclpy.Parameter(name, value=[])).value
        for name in (
                "grid_1.lift_stage_1_joints",
                "grid_1.clearance_joints"):
            self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.cfg[name] = self.get_parameter_or(
                name, rclpy.Parameter(name, value=[])).value
        for model in ("red_bin", "blue_bin"):
            for suffix in (
                    "approach_joints", "left_release_joints",
                    "center_intermediate_joints",
                    "center_release_joints", "right_release_joints"):
                name = f"{model}.{suffix}"
                self.declare_parameter(
                    name, rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.cfg[name] = self.get_parameter_or(
                    name, rclpy.Parameter(name, value=[])).value
        for model, y in (("red_bin", 0.18), ("blue_bin", -0.18)):
            for suffix, value in (
                    ("center_world_xy", [0.0, y]),
                    ("slot_offsets_x", [-0.040, 0.0, 0.040]),
                    ("slot_release_z", 0.438)):
                name = f"{model}.{suffix}"
                self.declare_parameter(name, value)
                configured = self.get_parameter(name).value
                self.cfg[name] = (list(configured)
                                  if isinstance(configured, list)
                                  else configured)
        self._validate_parameters()
        self.dry_run = self.cfg["dry_run"]
        self._task_lock = threading.Lock()
        self._faulted = False
        self._arm_goal = None
        self._arm_result = None
        self._pending_arm = None
        self._cycle_airborne = False
        self._cycle_detach_failed = False
        self._cycle_motion = False
        self._bin_execution_uncertain = False
        self._pending_attach = None
        self._attached_model = None
        self._pending_attach_model = None
        self._pending_detach = None
        self._joint_state_lock = threading.Lock()
        self._joint_sample = None
        self._gripper_sample = None
        self._gripper_close_sent_at = None
        self.callback_group = ReentrantCallbackGroup()
        if (not self.dry_run or self.cfg["bin_slot_calibration"]
                or self.cfg["grid_pick_calibration"]):
            self.joint_subscription = self.create_subscription(
                JointState, self.cfg["joint_states_topic"],
                self.joint_state_callback, qos_profile_sensor_data,
                callback_group=self.callback_group)
            self.entity_client = self.create_client(
                GetEntityState, self.cfg["entity_service"],
                callback_group=self.callback_group)
            self.ik_client = self.create_client(
                GetPositionIK, self.cfg["ik_service"],
                callback_group=self.callback_group)
            self.plan_client = self.create_client(
                GetMotionPlan, self.cfg["planning_service"],
                callback_group=self.callback_group)
            self.state_validity_client = self.create_client(
                GetStateValidity, "/check_state_validity",
                callback_group=self.callback_group)
            self.move_group_parameters_client = self.create_client(
                GetParameters, "/move_group/get_parameters",
                callback_group=self.callback_group)
            self.attach_client = self.create_client(
                Attach, self.cfg["attach_service"],
                callback_group=self.callback_group)
            self.detach_client = self.create_client(
                Detach, self.cfg["detach_service"],
                callback_group=self.callback_group)
            self.arm_client = ActionClient(
                self, FollowJointTrajectory, self.cfg["arm_action"],
                callback_group=self.callback_group)
            self.gripper = self.create_publisher(
                Float64MultiArray, self.cfg["gripper_topic"], 10)
        self.observation_service = self.create_service(
            Trigger, "/e3/prepare_observation", self.prepare_observation,
            callback_group=self.callback_group)
        self.server = ActionServer(
            self, PickAndSort, self.cfg["action_name"],
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group)
        self.get_logger().info(
            f"Pick-and-sort server {self.cfg['action_name']}; "
            f"dry_run={self.dry_run}")

    def prepare_observation(self, request, response):
        if not self._task_lock.acquire(blocking=False):
            response.success = False
            response.message = "Observation rejected: robot busy"
            return response
        motion_sent = False
        try:
            if (self.dry_run or self._faulted or self.attached
                    or self._pending_attach is not None
                    or self._pending_attach_model is not None
                    or self._pending_detach is not None
                    or self._pending_arm is not None
                    or self._arm_goal is not None):
                raise RuntimeError(
                    "Observation rejected: dry-run, fault or pending motion/"
                    "attachment; inspect robot before recovery")
            self.get_logger().info(
                "Observation: waiting for fresh joints, MoveIt, Gazebo "
                "and controllers")
            self.check_bin_calibration_runtime(None)
            self.wait_ready(
                lambda: self.gripper.get_subscription_count() > 0,
                "observation gripper controller")
            self.current_ik_seed(None)
            values = list(self.cfg["red_bin.approach_joints"])
            if len(values) != 6:
                raise ValueError("Observation requires calibrated approach")
            self.validate_joint_values("red_bin.approach_joints", values)
            self.command_gripper(True, None)
            trajectory = self.plan_joints(values, None)
            target = JointState()
            target.name = list(self.cfg["joint_names"])
            target.position = values
            self.check_bin_state_validity(target, None, "observation_target")
            marker = time.monotonic()
            motion_sent = True
            self.execute_trajectory(trajectory, None)
            actual = self.current_ik_seed(None, after=marker)
            self.check_bin_state_validity(actual, None, "observation_actual")
            response.success = True
            response.message = "observation_ready: red_bin.approach_joints"
        except Exception as exc:
            if motion_sent:
                self._faulted = True
            response.success = False
            response.message = f"Observation preparation failed: {exc}"
            self.get_logger().error(response.message)
        finally:
            self._task_lock.release()
        return response

    def _validate_parameters(self):
        calibration_slots(self.cfg)
        self.validate_approach_joints()
        for name, size in (
                ("grasp_offset", 3), ("release_offset", 3),
                ("grasp_orientation", 4), ("release_orientation", 4),
                ("home_joints", 6)):
            values = self.cfg[name]
            if (len(values) != size
                    or not all(math.isfinite(x) for x in values)):
                raise ValueError(f"{name} needs {size} finite values")
        if (len(self.cfg["joint_names"]) != 6
                or len(set(self.cfg["joint_names"])) != 6):
            raise ValueError("joint_names must contain six distinct joints")
        for name in ("grasp_orientation", "release_orientation"):
            norm = math.sqrt(sum(x * x for x in self.cfg[name]))
            if norm < 1e-8:
                raise ValueError(f"{name} has zero norm")
            self.cfg[name] = [x / norm for x in self.cfg[name]]
        for name in (
                "service_timeout", "ik_timeout", "planning_time",
                "execution_timeout", "cancel_timeout", "joint_tolerance",
                "pregrasp_height", "joint_state_timeout",
                "joint_state_max_age", "attach_max_distance",
                "gripper_position_tolerance", "ik_seed_max_distance",
                "local_ik_seed_max_distance", "local_center_tolerance",
                "bin_horizontal_tolerance", "lift_calibration_height",
                "return_height_tolerance", "local_center_max_displacement"):
            if not math.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("feedback_delay", "gripper_wait", "object_settle_time"):
            if (not math.isfinite(self.cfg[name])
                    or self.cfg[name] < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("velocity_scaling", "acceleration_scaling"):
            if not 0 < self.cfg[name] <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.cfg["planning_attempts"] < 1:
            raise ValueError("planning_attempts must be positive")
        for name in ("gripper_open", "gripper_closed"):
            if not math.isfinite(self.cfg[name]):
                raise ValueError(f"{name} must be finite")

        if not 1 <= self.cfg["local_center_max_iterations"] <= 3:
            raise ValueError("local_center_max_iterations must be 1 to 3")
        if self.cfg["local_center_max_displacement"] > 0.012:
            raise ValueError(
                "local_center_max_displacement must not exceed 0.012 m")
        if self.cfg["local_center_tolerance"] > 0.01:
            raise ValueError("local_center_tolerance must not exceed 0.01 m")
        if self.cfg["local_ik_seed_max_distance"] > 0.40:
            raise ValueError("local_ik_seed_max_distance must not exceed 0.40")
        if self.cfg["bin_horizontal_tolerance"] > 0.004:
            raise ValueError(
                "bin_horizontal_tolerance must not exceed 0.004 m")
        for grid in range(1, 7):
            name = f"grid_{grid}.pick_joints"
            self.validate_joint_values(name, self.cfg[name])
            name = f"grid_{grid}.lift_joints"
            self.validate_joint_values(name, self.cfg[name])
        for name in (
                "grid_1.lift_stage_1_joints",
                "grid_1.clearance_joints"):
            self.validate_joint_values(name, self.cfg[name])
        for name, expected in (
                ("attach_service", "/gazebo/attach"),
                ("detach_service", "/gazebo/detach")):
            if self.cfg[name] != expected:
                raise ValueError(f"{name} must be {expected}")

    def validate_joint_values(self, name, values):
        if not values:
            return
        if (len(values) != 6
                or not all(math.isfinite(v) for v in values)):
            raise ValueError(f"{name} needs six finite joint values")
        for joint, value in zip(self.cfg["joint_names"], values):
            if joint not in APPROACH_JOINT_LIMITS:
                raise ValueError(f"{name}: unknown joint {joint}")
            lower, upper = APPROACH_JOINT_LIMITS[joint]
            if not lower <= value <= upper:
                raise ValueError(
                    f"{name}: {joint}={value} outside [{lower}, {upper}]")

    def validate_approach_joints(self):
        for model in ("red_bin", "blue_bin"):
            for suffix in (
                    "approach_joints", "left_release_joints",
                    "center_intermediate_joints",
                    "center_release_joints", "right_release_joints"):
                name = f"{model}.{suffix}"
                self.validate_joint_values(name, self.cfg[name])

    def validate_bin_slot_mode(self):
        if not self.cfg["bin_slot_calibration"]:
            return
        invalid = []
        for name in (
                "attach_enabled", "lift_calibration",
                "direct_pick_calibration", "local_center_calibration",
                "classification_transport_enabled",
                "grid_pick_calibration"):
            if self.cfg[name]:
                invalid.append(f"{name}=true")
        if self.cfg["calibration_bin"] not in ("red", "blue"):
            invalid.append("calibration_bin must be red or blue")
        if self.cfg["calibration_slot"] not in ("left", "right"):
            invalid.append("calibration_slot must be left or right")
        if self.cfg["calibration_execute"] and self.dry_run:
            invalid.append("calibration_execute requires dry_run=false")
        if invalid:
            raise CalibrationRequired(
                "bin_slot_calibration rejected before motion: "
                + "; ".join(invalid))

    @property
    def attached(self):
        return self._attached_model is not None

    def goal_callback(self, request):
        valid = (
            request.grid_id in range(1, 7)
            and request.object_class in {"comb", "mouse"}
            and request.object_model == f"pick_object_{request.grid_id}")
        calibration_mode = (
            self.cfg["direct_pick_calibration"]
            or self.cfg["lift_calibration"]
            or self.cfg["local_center_calibration"]
            or self.cfg["bin_slot_calibration"]
            or self.cfg["grid_pick_calibration"])
        transport_disabled = (
            not self.dry_run and not calibration_mode
            and not self.cfg["classification_transport_enabled"])
        if (self.cfg["classification_transport_enabled"]
                and not self.cfg["grid_pick_calibration"]
                and not self.valid_classification_goal(request)):
            valid = False
        if not valid or self._faulted or self.attached or transport_disabled:
            self.get_logger().warning(
                "Rejecting invalid, disabled, or recovery-faulted goal")
            return GoalResponse.REJECT
        if not self._task_lock.acquire(blocking=False):
            self.get_logger().warning("Rejecting goal: robot already busy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warning("Cancellation requested")
        return CancelResponse.ACCEPT

    def check_cancel(self, goal_handle):
        if goal_handle is not None and goal_handle.is_cancel_requested:
            raise TaskCancelled("Client requested cancellation")

    def wait_future(self, future, timeout, label, goal_handle=None):
        # Wall-clock waits leave other executor threads free to handle replies.
        deadline = time.monotonic() + timeout
        while not future.done():
            self.check_cancel(goal_handle)
            if not rclpy.ok():
                raise RuntimeError(f"{label}: ROS context shut down")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{label}: timed out after {timeout:.1f}s")
            time.sleep(0.02)
        if future.cancelled():
            raise RuntimeError(f"{label}: future cancelled")
        value = future.result()
        if value is None:
            raise RuntimeError(f"{label}: empty response")
        return value

    def pause(self, duration, goal_handle=None):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.check_cancel(goal_handle)
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        self.check_cancel(goal_handle)

    def wait_ready(self, ready, label, goal_handle=None):
        deadline = time.monotonic() + self.cfg["service_timeout"]
        while not ready():
            self.check_cancel(goal_handle)
            if not rclpy.ok() or time.monotonic() >= deadline:
                raise TimeoutError(f"{label}: endpoint unavailable")
            time.sleep(0.02)
        self.check_cancel(goal_handle)

    def call_service(self, client, request, label, goal_handle=None):
        self.wait_ready(client.service_is_ready, label, goal_handle)
        future = client.call_async(request)
        try:
            return self.wait_future(
                future, self.cfg["service_timeout"], label, goal_handle)
        finally:
            if not future.done():
                client.remove_pending_request(future)

    def query_entity_pose(self, model, goal_handle, frame=None):
        request = GetEntityState.Request()
        request.name = model
        request.reference_frame = (
            frame if frame is not None else self.cfg["frame_id"])
        response = self.call_service(
            self.entity_client, request, f"query {model}", goal_handle)
        if not response.success:
            raise RuntimeError(f"Gazebo entity query failed: {model}")
        position = response.state.pose.position
        if not all(math.isfinite(x) for x in (
                position.x, position.y, position.z)):
            raise RuntimeError(f"Nonfinite entity position: {model}")
        return response.state.pose

    def query_entity(self, model, goal_handle):
        return self.query_entity_pose(model, goal_handle).position

    def query_grasp_link(self, goal_handle):
        name = f"{self.cfg['robot_model']}::{self.cfg['robot_link']}"
        return self.query_entity(name, goal_handle)

    def target_pose(self, position, offset, orientation, lift=0.0):
        # Offset is world-axis object-to-IK-link calibration, applied once.
        # The IK model itself resolves link6 -> grasp_attach_link.
        pose = PoseStamped()
        pose.header.frame_id = self.cfg["frame_id"]
        pose.pose.position.x = position.x + offset[0]
        pose.pose.position.y = position.y + offset[1]
        pose.pose.position.z = position.z + offset[2] + lift
        q = pose.pose.orientation
        q.x, q.y, q.z, q.w = orientation
        return pose

    def joint_state_callback(self, message):
        # Store one complete sample; never combine joints from different times.
        sample = None
        gripper_sample = None
        if (len(message.name) == len(message.position)
                and len(set(message.name)) == len(message.name)):
            positions = dict(zip(message.name, message.position))
            if ("gripper_controller" in positions
                    and math.isfinite(positions["gripper_controller"])):
                gripper_sample = (
                    time.monotonic(),
                    float(positions["gripper_controller"]))
            names = self.cfg["joint_names"]
            if all(name in positions and math.isfinite(positions[name])
                   for name in names):
                sample = (
                    time.monotonic(), message.header.stamp,
                    [float(positions[name]) for name in names])
        with self._joint_state_lock:
            self._joint_sample = sample
            self._gripper_sample = gripper_sample

    def grid_reference_seed(self, grid_id):
        values = self.cfg[f"grid_{grid_id}.pick_joints"]
        if not values:
            raise CalibrationRequired(
                f"grid {grid_id} is uncalibrated: configure "
                f"grid_{grid_id}.pick_joints before non-dry-run motion")
        seed = JointState()
        seed.name = list(self.cfg["joint_names"])
        seed.position = list(values)
        return seed

    def current_ik_seed(self, goal_handle, after=None):
        deadline = time.monotonic() + self.cfg["joint_state_timeout"]
        while True:
            self.check_cancel(goal_handle)
            now = time.monotonic()
            with self._joint_state_lock:
                sample = self._joint_sample
            if (sample is not None
                    and (after is None or sample[0] > after)
                    and now - sample[0] <= self.cfg["joint_state_max_age"]):
                seed = JointState()
                seed.header.stamp = sample[1]
                seed.name = list(self.cfg["joint_names"])
                seed.position = list(sample[2])
                return seed
            if not rclpy.ok() or now >= deadline:
                raise TimeoutError(
                    f"{self.cfg['joint_states_topic']}: no fresh complete "
                    "six-joint IK seed")
            time.sleep(0.02)

    def solve_ik(self, pose, goal_handle, purpose="pose_target",
                 reference_seed=None, max_distance=None):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self.cfg["group_name"]
        ik.ik_link_name = self.cfg["ik_link_name"]
        ik.pose_stamped = pose
        # Explicit arm seed; retain other joints from the MoveIt scene.
        ik.robot_state.is_diff = True
        seed = (reference_seed if reference_seed is not None
                else self.current_ik_seed(goal_handle))
        if (seed.name != self.cfg["joint_names"] or len(seed.position) != 6
                or not all(math.isfinite(v) for v in seed.position)):
            raise RuntimeError("Invalid six-joint IK reference seed")
        ik.robot_state.joint_state = seed
        ik.avoid_collisions = bool(self.cfg["ik_avoid_collisions"])
        seconds = self.cfg["ik_timeout"]
        ik.timeout.sec = int(seconds)
        ik.timeout.nanosec = int((seconds - int(seconds)) * 1e9)
        position = pose.pose.position
        orientation = pose.pose.orientation
        self.get_logger().info(
            f"IK target purpose={purpose}, frame={pose.header.frame_id}, "
            f"position=({position.x:.10f}, {position.y:.10f}, "
            f"{position.z:.10f}), "
            f"orientation=({orientation.x:.10f}, {orientation.y:.10f}, "
            f"{orientation.z:.10f}, {orientation.w:.10f}), "
            f"avoid_collisions={ik.avoid_collisions}, "
            f"seed={list(ik.robot_state.joint_state.position)}")
        response = self.call_service(
            self.ik_client, request, "IK", goal_handle)
        if response.error_code.val == MoveItErrorCodes.NO_IK_SOLUTION:
            raise NoIKSolution(
                f"IK error_code={response.error_code.val} "
                "(NO_IK_SOLUTION)")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"IK error_code={response.error_code.val}")
        state = response.solution.joint_state
        values = dict(zip(state.name, state.position))
        if any(name not in values for name in self.cfg["joint_names"]):
            raise RuntimeError("IK solution missing active joints")
        solution = [values[name] for name in self.cfg["joint_names"]]
        if not all(math.isfinite(v) for v in solution):
            raise RuntimeError("IK solution contains nonfinite positions")
        # These bounded arm joints use raw radians, not wrapped differences.
        deltas = [value - reference
                  for value, reference in zip(solution, seed.position)]
        distance = math.sqrt(sum(delta * delta for delta in deltas))
        differences = ", ".join(
            f"{name}={delta:+.10f}"
            for name, delta in zip(self.cfg["joint_names"], deltas))
        limit = self.cfg["ik_seed_max_distance"]
        if max_distance is not None:
            limit = min(limit, max_distance)
        self.get_logger().info(
            f"IK branch purpose={purpose}, delta_rad=[{differences}], "
            f"seed_distance={distance:.10f}rad, "
            f"limit={limit:.10f}rad")
        if self.exceeds_limit(distance, limit):
            raise IKSeedDistanceExceeded(distance, limit)
        return solution

    def plan_joints(self, values, goal_handle):
        if len(values) != 6 or not all(math.isfinite(v) for v in values):
            raise RuntimeError("Invalid joint target")
        request = GetMotionPlan.Request()
        plan = request.motion_plan_request
        plan.group_name = self.cfg["group_name"]
        plan.pipeline_id = self.cfg["planning_pipeline_id"]
        plan.planner_id = self.cfg["planner_id"]
        plan.start_state.is_diff = True
        plan.allowed_planning_time = self.cfg["planning_time"]
        plan.num_planning_attempts = self.cfg["planning_attempts"]
        plan.max_velocity_scaling_factor = self.cfg["velocity_scaling"]
        plan.max_acceleration_scaling_factor = self.cfg["acceleration_scaling"]
        constraints = Constraints()
        for name, value in zip(self.cfg["joint_names"], values):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = self.cfg["joint_tolerance"]
            joint.tolerance_below = self.cfg["joint_tolerance"]
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        if (self.cfg["lift_calibration"]
                or self.cfg["bin_slot_calibration"]):
            plan.max_velocity_scaling_factor = min(
                0.1, plan.max_velocity_scaling_factor)
            plan.max_acceleration_scaling_factor = min(
                0.1, plan.max_acceleration_scaling_factor)
        plan.goal_constraints = [constraints]
        targets = ", ".join(
            f"{name}={value:.6f}"
            for name, value in zip(self.cfg["joint_names"], values))
        self.get_logger().info(
            f"Planning group_name={plan.group_name}, "
            f"pipeline_id={plan.pipeline_id}, planner_id={plan.planner_id}, "
            f"target_joints=[{targets}], "
            f"planning_time={plan.allowed_planning_time:.2f}s, "
            f"planning_attempts={plan.num_planning_attempts}")
        response = self.call_service(
            self.plan_client, request, "motion planning", goal_handle)
        result = response.motion_plan_response
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"Planning error_code={result.error_code.val}")
        trajectory = result.trajectory.joint_trajectory
        if (not trajectory.points or
                set(trajectory.joint_names) != set(self.cfg["joint_names"])):
            raise RuntimeError(
                "Planner returned empty or unexpected trajectory")
        for point in trajectory.points:
            if (len(point.positions) != 6
                    or not all(math.isfinite(x) for x in point.positions)):
                raise RuntimeError("Planner returned invalid joint positions")
        return trajectory

    def execute_trajectory(self, trajectory, goal_handle):
        self.wait_ready(
            self.arm_client.server_is_ready, "arm controller", goal_handle)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        if self._cycle_motion:
            # A send exception may leave the remote execution outcome unknown.
            self._cycle_airborne = True
        self._pending_arm = self.arm_client.send_goal_async(goal)
        self._arm_goal = self.wait_future(
            self._pending_arm, self.cfg["service_timeout"],
            "arm goal acceptance", goal_handle)
        self._pending_arm = None
        if not self._arm_goal.accepted:
            self._arm_goal = None
            raise RuntimeError("Arm controller rejected trajectory")
        self._arm_result = self._arm_goal.get_result_async()
        response = self.wait_future(
            self._arm_result, self.cfg["execution_timeout"],
            "arm execution", goal_handle)
        self._arm_goal = None
        self._arm_result = None
        if (response.status != GoalStatus.STATUS_SUCCEEDED
                or response.result.error_code !=
                FollowJointTrajectory.Result.SUCCESSFUL):
            raise RuntimeError(
                f"Arm status={response.status}, "
                f"error_code={response.result.error_code}: "
                f"{response.result.error_string}")
        self.check_cancel(goal_handle)

    def move_pose(self, pose, goal_handle, purpose="pose_target",
                  reference_seed=None):
        values = self.solve_ik(
            pose, goal_handle, purpose, reference_seed=reference_seed)
        trajectory = self.plan_joints(values, goal_handle)
        self.execute_trajectory(trajectory, goal_handle)
        # Require feedback received after successful trajectory completion.
        measured_seed = self.current_ik_seed(
            goal_handle, after=time.monotonic())
        actual = self.query_grasp_link(goal_handle)
        target = pose.pose.position
        self.get_logger().info(
            f"Reached purpose={purpose}, ik_link={self.cfg['ik_link_name']}, "
            f"target=({target.x:.4f}, {target.y:.4f}, {target.z:.4f}), "
            f"link_world=({actual.x:.4f}, {actual.y:.4f}, {actual.z:.4f}), "
            f"actual_joints={list(measured_seed.position)}")
        return measured_seed

    def command_gripper(self, opened, goal_handle=None):
        self.wait_ready(
            lambda: self.gripper.get_subscription_count() > 0,
            "gripper subscriber", goal_handle)
        message = Float64MultiArray()
        message.data = [float(self.cfg[
            "gripper_open" if opened else "gripper_closed"])]
        if not opened:
            self._gripper_close_sent_at = time.monotonic()
        self.gripper.publish(message)
        self.pause(self.cfg["gripper_wait"], goal_handle)

    def attachment_request(self, model):
        request = Attach.Request()
        request.joint_name = self._joint_name
        request.model_name_1 = self.cfg["robot_model"]
        request.link_name_1 = self.cfg["robot_link"]
        request.model_name_2 = model
        request.link_name_2 = self.cfg["object_link"]
        return request

    def wait_attachment_service(self, client, name, goal_handle=None):
        deadline = time.monotonic() + self.cfg["service_timeout"]
        service_type = (
            "Detach" if name == self.cfg["detach_service"] else "Attach")
        expected = f"gazebo_model_attachment_plugin_msgs/srv/{service_type}"
        while True:
            self.check_cancel(goal_handle)
            if client.service_is_ready():
                return
            graph = dict(self.get_service_names_and_types())
            advertised = graph.get(name, [])
            if expected in advertised:
                # Graph discovery can precede client readiness.
                # Only a successful RPC reply confirms attachment effects.
                self.get_logger().warning(
                    f"{name}: advertised with correct type; client readiness "
                    "not yet confirmed, attempting bounded service request")
                return
            if not rclpy.ok() or time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{name}: endpoint unavailable to this client; "
                    f"expected={expected}, advertised_types={advertised}")
            time.sleep(0.05)

    def check_grasp_conditions(self, model, goal_handle):
        self.verify_closed_gripper()
        if self.cfg["lift_calibration"]:
            object_pose, link_pose = self.cycle_measurement(
                model, goal_handle, "attach_check")
            object_position, link_position = (
                object_pose.position, link_pose.position)
        else:
            object_position = self.query_entity(model, goal_handle)
            link_position = self.query_grasp_link(goal_handle)
        distance = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(
                (object_position.x, object_position.y, object_position.z),
                (link_position.x, link_position.y, link_position.z))))
        self.get_logger().info(
            f"Grasp check object={model}, "
            f"object_world=({object_position.x:.4f}, "
            f"{object_position.y:.4f}, {object_position.z:.4f}), "
            f"link_world=({link_position.x:.4f}, {link_position.y:.4f}, "
            f"{link_position.z:.4f}), distance={distance:.4f}m, "
            f"limit={self.cfg['attach_max_distance']:.4f}m")
        if distance > self.cfg["attach_max_distance"]:
            raise RuntimeError(
                f"Attach forbidden: distance {distance:.4f}m exceeds "
                f"{self.cfg['attach_max_distance']:.4f}m")

    def verify_closed_gripper(self):
        with self._joint_state_lock:
            sample = self._gripper_sample
        if (sample is None
                or time.monotonic() - sample[0]
                > self.cfg["joint_state_max_age"]
                or (self._gripper_close_sent_at is not None
                    and sample[0] < self._gripper_close_sent_at)):
            raise RuntimeError("Attach forbidden: no fresh gripper state "
                               "after the close command")
        error = abs(sample[1] - self.cfg["gripper_closed"])
        self.get_logger().info(
            f"Gripper measured={sample[1]:.4f}, "
            f"close_target={self.cfg['gripper_closed']:.4f}, "
            f"error={error:.4f}, "
            f"tolerance={self.cfg['gripper_position_tolerance']:.4f}")
        if error > self.cfg["gripper_position_tolerance"]:
            raise RuntimeError(
                f"Attach forbidden: gripper has not reached close target "
                f"(error={error:.4f})")

    def check_attachment_services(self, goal_handle):
        for client, name in (
                (self.attach_client, self.cfg["attach_service"]),
                (self.detach_client, self.cfg["detach_service"])):
            self.wait_attachment_service(client, name, goal_handle)

    def finish_attach(self, goal_handle=None):
        response = self.wait_future(
            self._pending_attach, self.cfg["service_timeout"],
            self.cfg["attach_service"], goal_handle)
        model = self._pending_attach_model
        self._pending_attach = None
        self._pending_attach_model = None
        if response.success:
            self._attached_model = model
            self.get_logger().info(
                f"Attached model={model}, joint={self._joint_name}")
        return response

    def attach_object(self, model, goal_handle):
        if not self.cfg["attach_enabled"]:
            diagnostics = "distance and gripper checks passed"
            try:
                self.check_grasp_conditions(model, goal_handle)
            except TaskCancelled:
                raise
            except Exception as exc:
                diagnostics = str(exc)
            raise CalibrationRequired(
                f"attach_enabled=false; {diagnostics}; stopped after closure "
                "for calibration without attach, lift or automatic retreat")

        if (self._attached_model is not None
                or self._pending_attach is not None):
            raise RuntimeError("Cannot attach: existing or pending attachment")
        # Check recovery availability before creating a physical constraint.
        self.check_attachment_services(goal_handle)
        self.check_grasp_conditions(model, goal_handle)
        self.check_cancel(goal_handle)
        self._pending_attach = self.attach_client.call_async(
            self.attachment_request(model))
        self._pending_attach_model = model
        response = self.finish_attach(goal_handle)
        if not response.success:
            raise RuntimeError(
                f"{self.cfg['attach_service']} failed: {response.message}")

    def detachment_request(self, model):
        request = Detach.Request()
        request.joint_name = self._joint_name
        request.model_name_1 = self.cfg["robot_model"]
        request.model_name_2 = model
        return request

    def detach_object(self, model, goal_handle=None):
        if self._attached_model != model:
            raise RuntimeError(f"No confirmed attachment for {model}")
        if self._pending_detach is None:
            self.wait_attachment_service(
                self.detach_client,
                self.cfg["detach_service"], goal_handle)
            self._pending_detach = self.detach_client.call_async(
                self.detachment_request(model))
        # Retain pending mutating calls across timeout/cancel; do not resend.
        response = self.wait_future(
            self._pending_detach, self.cfg["service_timeout"],
            self.cfg["detach_service"], goal_handle)
        self._pending_detach = None
        if not response.success:
            raise RuntimeError(
                f"{self.cfg['detach_service']} failed: {response.message}")
        self._attached_model = None
        self.get_logger().info(
            f"Detached model={model}, joint={self._joint_name}")

    def stop_arm(self):
        # Resolve late acceptance before cancellation so no goal is orphaned.
        if self._pending_arm is not None:
            self._arm_goal = self.wait_future(
                self._pending_arm, self.cfg["cancel_timeout"],
                "pending arm acceptance during recovery")
            self._pending_arm = None
        if self._arm_goal is None or not self._arm_goal.accepted:
            self._arm_goal = None
            return
        if self._arm_result is None:
            self._arm_result = self._arm_goal.get_result_async()
        if not self._arm_result.done():
            self.wait_future(
                self._arm_goal.cancel_goal_async(), self.cfg["cancel_timeout"],
                "arm cancellation acknowledgement")
        result = self.wait_future(
            self._arm_result, self.cfg["cancel_timeout"], "arm stop result")
        if result.status not in (
                GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED):
            raise RuntimeError(f"Arm stop unconfirmed: status={result.status}")
        self._arm_goal = None
        self._arm_result = None

    def recover(self, model):
        if self.dry_run:
            return []
        errors = []
        try:
            self.stop_arm()
        except Exception as exc:
            errors.append(f"arm stop: {exc}")
            # A late accepted goal must still receive a cancellation request.
            if self._pending_arm is not None:
                self._pending_arm.add_done_callback(self.cancel_late_arm)
        if self._pending_attach is not None:
            try:
                response = self.finish_attach()
                if not response.success:
                    self.get_logger().warning(
                        f"Attach was rejected: {response.message}")
            except Exception as exc:
                errors.append(f"attach outcome unknown: {exc}")
        if self._attached_model is not None:
            try:
                self.detach_object(self._attached_model)
            except Exception as exc:
                errors.append(f"detach: {exc}")
        try:
            self.command_gripper(True)
        except Exception as exc:
            errors.append(f"open gripper: {exc}")
        # Reject further goals if the remote state could not be confirmed.
        if errors:
            self._faulted = True
            self.get_logger().error(
                "Recovery incomplete: " + "; ".join(errors))
        return errors

    def cancel_late_arm(self, future):
        try:
            handle = future.result()
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
                self.get_logger().error(
                    "Late arm goal cancelled; inspect robot "
                    "before restarting server")
        except Exception as exc:
            self.get_logger().error(f"Late arm cancellation failed: {exc}")

    def publish_feedback(self, goal_handle, stage, progress):
        feedback = PickAndSort.Feedback()
        feedback.stage = stage
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"Stage: {stage}, progress={progress:.0%}")
        self.pause(self.cfg["feedback_delay"], goal_handle)

    def local_center_correction(self, model, measured_seed, goal_handle,
                                fixed_orientation=None):
        maximum = self.cfg["local_center_max_iterations"]
        tolerance = self.cfg["local_center_tolerance"]
        displacement_limit = self.cfg["local_center_max_displacement"]
        link_name = f"{self.cfg['robot_model']}::{self.cfg['robot_link']}"
        # Include the final measurement after the last permitted correction.
        for completed in range(maximum + 1):
            self.check_cancel(goal_handle)
            object_pose = self.query_entity_pose(
                model, goal_handle, frame="world")
            link_pose = self.query_entity_pose(
                link_name, goal_handle, frame="world")
            p, current = object_pose.position, link_pose.position
            q = (fixed_orientation if fixed_orientation is not None
                 else link_pose.orientation)
            delta = [p.x - current.x, p.y - current.y, p.z - current.z]
            distance = math.sqrt(sum(value * value for value in delta))
            self.get_logger().info(
                f"Local center iteration={completed}/{maximum}, "
                f"object_world=({p.x:.10f}, {p.y:.10f}, {p.z:.10f}), "
                f"link_world=({current.x:.10f}, {current.y:.10f}, "
                f"{current.z:.10f}), dx={delta[0]:+.10f}, "
                f"dy={delta[1]:+.10f}, dz={delta[2]:+.10f}, "
                f"distance={distance:.10f}m, tolerance={tolerance:.10f}m, "
                f"actual_joints={list(measured_seed.position)}")
            if not self.exceeds_limit(distance, tolerance):
                return True, completed, distance
            if completed == maximum:
                return False, completed, distance
            if self.exceeds_limit(distance, displacement_limit):
                raise RuntimeError(
                    f"Local center rejected: displacement={distance:.10f}m "
                    f"exceeds {displacement_limit:.10f}m")
            components = [q.x, q.y, q.z, q.w]
            if (not all(math.isfinite(value) for value in components)
                    or abs(sum(value * value for value in components) - 1.0)
                    > 0.001):
                raise RuntimeError(
                    "Local center rejected: invalid link orientation")
            # Refresh feedback every iteration, never reuse a previous IK goal.
            measured_seed = self.current_ik_seed(goal_handle)
            target = PoseStamped()
            target.header.frame_id = "world"
            target.pose.position = object_pose.position
            target.pose.orientation = copy.deepcopy(q)
            joints = self.solve_ik(
                target, goal_handle,
                purpose=f"local_center_correction_{completed + 1}",
                reference_seed=measured_seed,
                max_distance=min(
                    0.30, self.cfg["local_ik_seed_max_distance"]))
            trajectory = self.plan_joints(joints, goal_handle)
            self.execute_trajectory(trajectory, goal_handle)
            measured_seed = self.current_ik_seed(
                goal_handle, after=time.monotonic())

    def cycle_measurement(self, model, goal_handle, label):
        object_pose = self.query_entity_pose(
            f"{model}::{self.cfg['object_link']}", goal_handle, frame="world")
        link_pose = self.query_entity_pose(
            f"{self.cfg['robot_model']}::{self.cfg['robot_link']}",
            goal_handle, frame="world")
        a, b = object_pose.position, link_pose.position
        distance = math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        self.get_logger().info(
            f"{label}: object_world=({a.x:.7f}, {a.y:.7f}, {a.z:.7f}), "
            f"link_world=({b.x:.7f}, {b.y:.7f}, {b.z:.7f}), "
            f"distance={distance:.7f}m")
        return object_pose, link_pose

    def lift_calibration_motion(self, model, goal_handle):
        before_object, before_link = self.cycle_measurement(
            model, goal_handle, "before_lift")
        seed = self.current_ik_seed(goal_handle)
        self._cycle_before = (
            copy.deepcopy(seed), copy.deepcopy(before_object),
            copy.deepcopy(before_link))
        self.get_logger().info(
            f"Before lift actual_joints={list(seed.position)}")
        q = before_link.orientation
        if (not all(math.isfinite(v) for v in (q.x, q.y, q.z, q.w))
                or abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) > 0.001):
            raise RuntimeError("Lift calibration: invalid link orientation")
        target = PoseStamped()
        target.header.frame_id = "world"
        target.pose = copy.deepcopy(before_link)
        target.pose.position.z += self.cfg["lift_calibration_height"]
        joints = self.solve_ik(
            target, goal_handle, purpose="lift_calibration",
            reference_seed=seed,
            max_distance=min(0.30, self.cfg["local_ik_seed_max_distance"]))
        trajectory = self.plan_joints(joints, goal_handle)
        self._cycle_motion = True
        self.execute_trajectory(trajectory, goal_handle)
        self.current_ik_seed(goal_handle, after=time.monotonic())
        obj, link = self.cycle_measurement(model, goal_handle, "after_lift")
        link_rise = link.position.z - before_link.position.z
        object_rise = obj.position.z - before_object.position.z
        self.get_logger().info(
            f"Actual lift: link={link_rise:.7f}m, object={object_rise:.7f}m")
        expected = self.cfg["lift_calibration_height"]
        lower, upper = expected - 0.005, expected + 0.005
        if not (lower - NUMERIC_EPSILON <= link_rise
                <= upper + NUMERIC_EPSILON
                and lower - NUMERIC_EPSILON <= object_rise
                <= upper + NUMERIC_EPSILON):
            raise RuntimeError(
                f"Actual lift outside {lower:.3f} to {upper:.3f} m")
        self.publish_feedback(goal_handle, "lowering_calibration", 0.95)
        trajectory = self.plan_joints(list(seed.position), goal_handle)
        self.execute_trajectory(trajectory, goal_handle)
        self.current_ik_seed(goal_handle, after=time.monotonic())
        _, returned = self.cycle_measurement(
            model, goal_handle, "after_lowering")
        error = abs(returned.position.z - before_link.position.z)
        self.get_logger().info(f"Return height error={error:.7f}m")
        if self.exceeds_limit(
                error, self.cfg["return_height_tolerance"]):
            raise RuntimeError("Return height not confirmed")
        self._cycle_airborne = False
        self._cycle_motion = False
        self.check_cancel(goal_handle)
        try:
            self.detach_object(model, goal_handle)
        except Exception:
            self._cycle_detach_failed = True
            raise
        self.command_gripper(True, goal_handle)
        self.pause(self.cfg["object_settle_time"], goal_handle)
        self.cycle_measurement(model, goal_handle, "final_settled")

    def recover_cycle(self, model):
        errors = []
        try:
            self.stop_arm()
        except Exception as exc:
            errors.append(f"arm stop: {exc}")
            if self._pending_arm is not None:
                self._pending_arm.add_done_callback(self.cancel_late_arm)
        if self._cycle_airborne or self._cycle_detach_failed or errors:
            errors.append(
                "Release forbidden: motion/return or detach unconfirmed")
        else:
            try:
                if self._pending_attach is not None:
                    self.finish_attach()
                if self.attached:
                    self.detach_object(model)
                self.command_gripper(True)
            except Exception as exc:
                errors.append(f"safe release: {exc}")
        if errors:
            self._faulted = True
        return errors

    def bin_slot_target(self):
        model = self.cfg["calibration_bin"] + "_bin"
        index = 0 if self.cfg["calibration_slot"] == "left" else 2
        xy = calibration_slots(self.cfg)[model][index]
        z = self.cfg[f"{model}.slot_release_z"]
        if not math.isfinite(z):
            raise CalibrationRequired(f"{model}.slot_release_z is not finite")
        return model, [xy[0], xy[1], z]

    def check_bin_calibration_runtime(self, goal_handle):
        endpoints = (
            (self.move_group_parameters_client, "move_group parameters"),
            (self.ik_client, self.cfg["ik_service"]),
            (self.plan_client, self.cfg["planning_service"]),
            (self.state_validity_client, "/check_state_validity"),
            (self.entity_client, self.cfg["entity_service"]))
        for client, label in endpoints:
            self.wait_ready(client.service_is_ready, label, goal_handle)
        self.wait_ready(
            self.arm_client.server_is_ready, self.cfg["arm_action"],
            goal_handle)
        request = GetParameters.Request()
        request.names = ["use_sim_time"]
        response = self.call_service(
            self.move_group_parameters_client, request,
            "move_group use_sim_time", goal_handle)
        if (len(response.values) != 1
                or response.values[0].type != ParameterType.PARAMETER_BOOL
                or not response.values[0].bool_value):
            raise CalibrationRequired(
                "/move_group use_sim_time must be true")

    def check_bin_state_validity(self, seed, goal_handle, stage):
        request = GetStateValidity.Request()
        request.group_name = self.cfg["group_name"]
        request.robot_state.is_diff = True
        request.robot_state.joint_state = seed
        response = self.call_service(
            self.state_validity_client, request,
            f"state validity after {stage}", goal_handle)
        if not response.valid:
            contacts = [
                f"{item.contact_body_1}/{item.contact_body_2}"
                for item in response.contacts]
            raise CalibrationRequired(
                f"{stage}: current state invalid, contacts={contacts}")

    @staticmethod
    def exceeds_limit(actual, configured):
        return actual > configured + NUMERIC_EPSILON

    @staticmethod
    def bin_segment_target(current, final, maximum, horizontal):
        if horizontal:
            delta = [
                final[0] - current.x, final[1] - current.y, 0.0]
            distance = math.hypot(delta[0], delta[1])
            if distance <= NUMERIC_EPSILON:
                return None
            segment_count = max(
                1, math.ceil((distance - NUMERIC_EPSILON) / maximum))
            scale = 1.0 / segment_count
            target = [
                current.x + delta[0] * scale,
                current.y + delta[1] * scale,
                current.z]
        else:
            xy_error = math.hypot(
                final[0] - current.x, final[1] - current.y)
            if PickActionServer.exceeds_limit(xy_error, maximum):
                raise RuntimeError(
                    f"Vertical XY correction {xy_error:.10f}m exceeds "
                    f"{maximum:.10f}m step limit")
            z_delta = final[2] - current.z
            if abs(z_delta) <= NUMERIC_EPSILON:
                return None
            squared_budget = maximum ** 2 - xy_error ** 2
            z_budget = math.sqrt(max(0.0, squared_budget))
            if z_budget <= NUMERIC_EPSILON:
                raise RuntimeError(
                    "No vertical step budget after XY correction")
            segment_count = max(
                1, math.ceil((abs(z_delta) - NUMERIC_EPSILON) / z_budget))
            target = [
                final[0], final[1], current.z + z_delta / segment_count]
        step = math.sqrt(sum(
            (value - actual) ** 2 for value, actual in zip(
                target, (current.x, current.y, current.z))))
        if PickActionServer.exceeds_limit(step, maximum):
            raise RuntimeError(
                f"Segment {step:.10f}m exceeds {maximum:.10f}m")
        return target

    def execute_bin_segment(self, before, target, orientation, maximum,
                            goal_handle, stage):
        step = math.sqrt(sum(
            (value - actual) ** 2 for value, actual in zip(
                target, (before.position.x, before.position.y,
                         before.position.z))))
        if self.exceeds_limit(step, maximum):
            raise CalibrationRequired(
                f"{stage}: requested step={step:.10f}m exceeds "
                f"{maximum:.10f}m")
        seed = self.current_ik_seed(goal_handle)
        pose = PoseStamped()
        pose.header.frame_id = "world"
        position = pose.pose.position
        position.x, position.y, position.z = target
        pose.pose.orientation = copy.deepcopy(orientation)
        try:
            joints = self.solve_ik(
                pose, goal_handle, purpose=stage, reference_seed=seed,
                max_distance=min(
                    0.30, self.cfg["local_ik_seed_max_distance"]))
            trajectory = self.plan_joints(joints, goal_handle)
        except Exception as exc:
            raise CalibrationRequired(f"{stage}: {exc}") from exc
        marker = time.monotonic()
        self._bin_execution_uncertain = True
        try:
            self.execute_trajectory(trajectory, goal_handle)
        except Exception as exc:
            raise RuntimeError(f"{stage}: {exc}") from exc
        self._bin_execution_uncertain = False
        actual_seed = self.current_ik_seed(goal_handle, after=marker)
        self.check_bin_state_validity(actual_seed, goal_handle, stage)
        actual_pose = self.query_entity_pose(
            f"{self.cfg['robot_model']}::{self.cfg['robot_link']}",
            goal_handle, frame="world")
        return actual_seed, actual_pose

    @staticmethod
    def bin_axis_error(position, final, horizontal):
        if horizontal:
            return math.hypot(
                final[0] - position.x, final[1] - position.y)
        return abs(final[2] - position.z)

    def run_bin_axis(self, final, orientation, maximum, horizontal,
                     goal_handle, label):
        tolerance = (
            self.cfg["bin_horizontal_tolerance"] if horizontal
            else BIN_FINAL_POSITION_TOLERANCE)
        link_name = (
            f"{self.cfg['robot_model']}::{self.cfg['robot_link']}")
        for index in range(1, 21):
            pose = self.query_entity_pose(
                link_name, goal_handle, frame="world")
            before_remaining = self.bin_axis_error(
                pose.position, final, horizontal)
            if not self.exceeds_limit(before_remaining, tolerance):
                self.get_logger().info(
                    f"{label}: aligned before next segment, "
                    f"remaining_error={before_remaining:.10f}m, "
                    f"tolerance={tolerance:.10f}m")
                return
            current = [
                pose.position.x, pose.position.y, pose.position.z]
            target = self.bin_segment_target(
                pose.position, final, maximum, horizontal)
            if target is None:
                return
            step = math.dist(target, current)
            stage = f"{label}_{index}"
            self.get_logger().info(
                f"{stage}: target=({target[0]:.10f}, {target[1]:.10f}, "
                f"{target[2]:.10f}), step={step:.10f}m, "
                f"limit={maximum:.10f}m, "
                f"remaining_error={before_remaining:.10f}m")
            _, actual = self.execute_bin_segment(
                pose, target, orientation, maximum, goal_handle, stage)
            after_remaining = self.bin_axis_error(
                actual.position, final, horizontal)
            self.get_logger().info(
                f"{stage}: measured remaining_error="
                f"{after_remaining:.10f}m, tolerance={tolerance:.10f}m")
            if not self.exceeds_limit(after_remaining, tolerance):
                return
            if after_remaining >= before_remaining - 1e-6:
                raise CalibrationRequired(f"{label}: no measured progress")
        raise CalibrationRequired(f"{label}: segment limit reached")

    def run_grid_pick_horizontal(self, final, goal_handle):
        """Align XY with bounded retries for two explicit IK rejections."""
        tolerance = self.cfg["bin_horizontal_tolerance"]
        link_name = (
            f"{self.cfg['robot_model']}::{self.cfg['robot_link']}")
        for index in range(1, 41):
            measured = self.query_entity_pose(
                link_name, goal_handle, frame="world")
            remaining = self.bin_axis_error(
                measured.position, final, horizontal=True)
            if not self.exceeds_limit(remaining, tolerance):
                self.get_logger().info(
                    "grid_pick_horizontal: aligned before next segment, "
                    f"remaining_error={remaining:.10f}m, "
                    f"tolerance={tolerance:.10f}m")
                return
            segment_orientation = copy.deepcopy(measured.orientation)
            if not self.valid_world_orientation(segment_orientation):
                raise CalibrationRequired(
                    "grid pick horizontal measured orientation is invalid")
            initial_target = self.bin_segment_target(
                measured.position, final, 0.02, horizontal=True)
            if initial_target is None:
                return
            start = [
                measured.position.x, measured.position.y,
                measured.position.z]
            vector = [
                initial_target[0] - start[0],
                initial_target[1] - start[1]]
            candidate = math.hypot(vector[0], vector[1])
            direction = [vector[0] / candidate, vector[1] / candidate]
            seed = self.current_ik_seed(goal_handle)
            original = candidate
            attempt = 1
            while True:
                target = [
                    start[0] + direction[0] * candidate,
                    start[1] + direction[1] * candidate,
                    start[2]]
                pose = PoseStamped()
                pose.header.frame_id = "world"
                pose.pose.position.x = target[0]
                pose.pose.position.y = target[1]
                pose.pose.position.z = target[2]
                pose.pose.orientation = copy.deepcopy(
                    segment_orientation)
                stage = f"grid_pick_horizontal_{index}_attempt_{attempt}"
                self.get_logger().info(
                    f"{stage}: original_candidate={original:.10f}m, "
                    f"candidate={candidate:.10f}m")
                try:
                    joints = self.solve_ik(
                        pose, goal_handle, purpose=stage,
                        reference_seed=seed,
                        max_distance=min(
                            0.30,
                            self.cfg["local_ik_seed_max_distance"]))
                except (NoIKSolution, IKSeedDistanceExceeded) as exc:
                    reduced = candidate / 2.0
                    detail = str(exc)
                    if isinstance(exc, IKSeedDistanceExceeded):
                        detail += (
                            f", seed_distance={exc.distance:.10f}rad")
                    self.get_logger().warning(
                        f"{stage}: rejected={detail}; "
                        f"halved_candidate={reduced:.10f}m")
                    if attempt >= 5:
                        raise CalibrationRequired(
                            f"{stage}: no safe IK solution after "
                            "5 candidate attempts") from exc
                    candidate = reduced
                    attempt += 1
                    continue
                self.get_logger().info(
                    f"{stage}: accepted_step={candidate:.10f}m")
                try:
                    trajectory = self.plan_joints(joints, goal_handle)
                except Exception as exc:
                    raise CalibrationRequired(
                        f"{stage}: planning failed: {exc}") from exc
                marker = time.monotonic()
                self._bin_execution_uncertain = True
                try:
                    self.execute_trajectory(trajectory, goal_handle)
                except Exception as exc:
                    raise RuntimeError(f"{stage}: {exc}") from exc
                self._bin_execution_uncertain = False
                actual_seed = self.current_ik_seed(
                    goal_handle, after=marker)
                self.check_bin_state_validity(
                    actual_seed, goal_handle, stage)
                actual = self.query_entity_pose(
                    link_name, goal_handle, frame="world")
                after_remaining = self.bin_axis_error(
                    actual.position, final, horizontal=True)
                self.get_logger().info(
                    f"{stage}: measured remaining_error="
                    f"{after_remaining:.10f}m, "
                    "next_segment_resets_candidate=true")
                if not self.exceeds_limit(after_remaining, tolerance):
                    return
                if after_remaining >= remaining - 1e-6:
                    raise CalibrationRequired(
                        "grid_pick_horizontal: no measured progress")
                break
        raise CalibrationRequired(
            "grid_pick_horizontal: segment limit reached")

    def validate_grid_pick_mode(self, request):
        if not self.cfg["grid_pick_calibration"]:
            raise CalibrationRequired("grid_pick_calibration is disabled")
        invalid = []
        for name in (
                "classification_transport_enabled",
                "direct_pick_calibration", "bin_slot_calibration",
                "attach_enabled", "lift_calibration"):
            if self.cfg[name]:
                invalid.append(f"{name}=true")
        if self.dry_run:
            invalid.append("dry_run must be false")
        if invalid:
            raise CalibrationRequired(
                "grid_pick_calibration rejected before motion: "
                + "; ".join(invalid))
        if request.grid_id not in range(2, 7):
            raise CalibrationRequired(
                "grid_pick_calibration supports grid_id 2 through 6 only")
        expected = f"pick_object_{request.grid_id}"
        if request.object_model != expected:
            raise CalibrationRequired(
                f"grid {request.grid_id} requires object_model={expected}")

    @staticmethod
    def valid_world_orientation(orientation):
        values = [
            orientation.x, orientation.y, orientation.z, orientation.w]
        return (
            all(math.isfinite(value) for value in values)
            and abs(sum(value * value for value in values) - 1.0) <= 0.001)

    def run_grid_pick_calibration(self, goal_handle):
        request = goal_handle.request
        self.validate_grid_pick_mode(request)
        self.check_bin_calibration_runtime(goal_handle)
        # Confirm a complete, fresh six-joint state before any command.
        self.current_ik_seed(goal_handle)
        if self.gripper.get_subscription_count() < 1:
            raise CalibrationRequired(
                f"{self.cfg['gripper_topic']}: no controller subscriber")
        self.publish_feedback(goal_handle, "opening_gripper", 0.10)
        self.command_gripper(True, goal_handle)

        self.publish_feedback(goal_handle, "moving_to_clearance", 0.20)
        clearance = list(self.cfg["grid_1.clearance_joints"])
        self.validate_joint_values("grid_1.clearance_joints", clearance)
        try:
            trajectory = self.plan_joints(clearance, goal_handle)
        except Exception as exc:
            raise CalibrationRequired(f"grid_clearance: {exc}") from exc
        marker = time.monotonic()
        self._bin_execution_uncertain = True
        try:
            self.execute_trajectory(trajectory, goal_handle)
        except Exception as exc:
            raise RuntimeError(f"grid_clearance: {exc}") from exc
        self._bin_execution_uncertain = False
        seed = self.current_ik_seed(goal_handle, after=marker)
        self.check_bin_state_validity(
            seed, goal_handle, "grid_pick_clearance")

        self.publish_feedback(goal_handle, "measuring_safe_start", 0.30)
        model = request.object_model
        link_name = (
            f"{self.cfg['robot_model']}::{self.cfg['robot_link']}")
        object_pose = self.query_entity_pose(
            model, goal_handle, frame="world")
        link_pose = self.query_entity_pose(
            link_name, goal_handle, frame="world")
        orientation = copy.deepcopy(link_pose.orientation)
        if not self.valid_world_orientation(orientation):
            raise CalibrationRequired(
                "grid pick clearance orientation is invalid")

        horizontal_target = [
            object_pose.position.x, object_pose.position.y,
            link_pose.position.z]
        self.publish_feedback(goal_handle, "grid_pick_horizontal", 0.45)
        self.run_grid_pick_horizontal(
            horizontal_target, goal_handle)

        self.publish_feedback(goal_handle, "grid_pick_vertical", 0.65)
        object_pose = self.query_entity_pose(
            model, goal_handle, frame="world")
        vertical_target = [
            object_pose.position.x, object_pose.position.y,
            object_pose.position.z]
        self.run_bin_axis(
            vertical_target, orientation, 0.01, False, goal_handle,
            "grid_pick_vertical")

        self.publish_feedback(goal_handle, "grid_pick_local_center", 0.82)
        seed = self.current_ik_seed(goal_handle)
        latest_link_pose = self.query_entity_pose(
            link_name, goal_handle, frame="world")
        latest_orientation = copy.deepcopy(latest_link_pose.orientation)
        if not self.valid_world_orientation(latest_orientation):
            raise CalibrationRequired(
                "grid pick final local orientation is invalid")
        converged, count, residual = self.local_center_correction(
            model, seed, goal_handle,
            fixed_orientation=latest_orientation)
        if not converged:
            raise CalibrationRequired(
                "grid pick local center did not converge: "
                f"distance={residual:.10f}m; corrections={count}")

        self.publish_feedback(goal_handle, "validating_grid_pick", 0.95)
        seed = self.current_ik_seed(goal_handle)
        object_pose = self.query_entity_pose(
            model, goal_handle, frame="world")
        link_pose = self.query_entity_pose(
            link_name, goal_handle, frame="world")
        object_xyz = [
            object_pose.position.x, object_pose.position.y,
            object_pose.position.z]
        link_xyz = [
            link_pose.position.x, link_pose.position.y,
            link_pose.position.z]
        distance = math.dist(object_xyz, link_xyz)
        if self.exceeds_limit(
                distance, self.cfg["local_center_tolerance"]):
            raise CalibrationRequired(
                f"grid pick final distance={distance:.10f}m exceeds "
                f"{self.cfg['local_center_tolerance']:.10f}m")
        self.check_bin_state_validity(
            seed, goal_handle, "grid_pick_final")
        values = ", ".join(f"{value:.10f}" for value in seed.position)
        snippet = f"grid_{request.grid_id}.pick_joints: [{values}]"
        message = (
            f"grid={request.grid_id}, object_model={model}, "
            f"object={object_xyz}, link={link_xyz}, "
            f"distance={distance:.10f}m, actual_joints={list(seed.position)}; "
            f"YAML: {snippet}")
        self.get_logger().info(message)
        return message

    def execute_grid_pick_callback(self, goal_handle, result):
        self._bin_execution_uncertain = False
        try:
            message = self.run_grid_pick_calibration(goal_handle)
            self.publish_feedback(
                goal_handle, "grid_pick_calibration_completed", 1.0)
            goal_handle.succeed()
            result.success = True
            result.final_state = "grid_pick_calibration_completed"
            result.message = message
        except Exception as exc:
            self.get_logger().error(f"grid_pick_calibration: {exc}")
            cancelled = (
                isinstance(exc, TaskCancelled)
                or goal_handle.is_cancel_requested)
            recovery_errors = []
            if self._bin_execution_uncertain:
                try:
                    self.stop_arm()
                except Exception as stop_error:
                    recovery_errors.append(str(stop_error))
            result.success = False
            if cancelled:
                goal_handle.canceled()
                result.final_state = "canceled"
            else:
                goal_handle.abort()
                result.final_state = (
                    "recovery_required" if self._bin_execution_uncertain
                    else "calibration_required")
            result.message = f"grid_pick_calibration: {exc}"
            if recovery_errors:
                result.message += (
                    "; arm stop unconfirmed: " + "; ".join(recovery_errors))
        return result

    def run_bin_slot_calibration(self, goal_handle):
        self.validate_bin_slot_mode()
        model, target = self.bin_slot_target()
        self.check_bin_calibration_runtime(goal_handle)
        approach_name = f"{model}.approach_joints"
        approach = list(self.cfg[approach_name])
        if len(approach) != 6:
            raise CalibrationRequired(f"{model}.approach_joints unavailable")
        try:
            trajectory = self.plan_joints(approach, goal_handle)
        except Exception as exc:
            raise CalibrationRequired(f"bin_approach: {exc}") from exc
        if not self.cfg["calibration_execute"]:
            message = (
                f"Plan-only validated for bin={self.cfg['calibration_bin']}, "
                f"slot={self.cfg['calibration_slot']}; approach planned; "
                "no trajectory executed")
            self.get_logger().info(message)
            return "calibration_plan_validated", message
        marker = time.monotonic()
        self._bin_execution_uncertain = True
        try:
            self.execute_trajectory(trajectory, goal_handle)
        except Exception as exc:
            raise RuntimeError(f"bin_approach: {exc}") from exc
        self._bin_execution_uncertain = False
        seed = self.current_ik_seed(goal_handle, after=marker)
        self.check_bin_state_validity(
            seed, goal_handle, "bin_approach")
        link_name = f"{self.cfg['robot_model']}::{self.cfg['robot_link']}"
        pose = self.query_entity_pose(link_name, goal_handle, frame="world")
        if pose.position.z <= target[2] + 0.01:
            raise CalibrationRequired(
                f"bin_approach safe height={pose.position.z:.7f}m must "
                f"exceed release z by 0.0100000m")
        orientation = copy.deepcopy(pose.orientation)
        components = [
            orientation.x, orientation.y, orientation.z, orientation.w]
        if (not all(math.isfinite(value) for value in components)
                or abs(sum(value * value for value in components) - 1.0)
                > 0.001):
            raise CalibrationRequired("Bin approach orientation is invalid")
        self.run_bin_axis(
            target, orientation, 0.02, True, goal_handle,
            "bin_horizontal")
        self.run_bin_axis(
            target, orientation, 0.01, False, goal_handle,
            "bin_vertical")
        seed = self.current_ik_seed(goal_handle)
        actual = self.query_entity_pose(link_name, goal_handle, frame="world")
        coordinates = [
            actual.position.x, actual.position.y, actual.position.z]
        error = math.dist(coordinates, target)
        if self.exceeds_limit(error, BIN_FINAL_POSITION_TOLERANCE):
            raise CalibrationRequired(
                f"bin final position error={error:.10f}m "
                "exceeds 0.0025000000m")
        key = (
            f"{model}.{self.cfg['calibration_slot']}_release_joints")
        values = ", ".join(f"{value:.10f}" for value in seed.position)
        snippet = f"{key}: [{values}]"
        message = (
            f"bin={self.cfg['calibration_bin']}, "
            f"slot={self.cfg['calibration_slot']}, "
            f"target={target}, actual={coordinates}, error={error:.10f}m, "
            f"actual_joints={list(seed.position)}; YAML: {snippet}")
        self.get_logger().info(message)
        return "bin_slot_calibration_completed", message

    def execute_bin_slot_callback(self, goal_handle, result):
        self._bin_execution_uncertain = False
        try:
            state, message = self.run_bin_slot_calibration(goal_handle)
            goal_handle.succeed()
            result.success = True
            result.final_state = state
            result.message = message
        except Exception as exc:
            self.get_logger().error(f"bin_slot_calibration: {exc}")
            cancelled = (
                isinstance(exc, TaskCancelled)
                or goal_handle.is_cancel_requested)
            recovery_errors = []
            if self._bin_execution_uncertain:
                try:
                    self.stop_arm()
                except Exception as stop_error:
                    recovery_errors.append(str(stop_error))
            result.success = False
            if cancelled:
                goal_handle.canceled()
                result.final_state = "canceled"
            else:
                goal_handle.abort()
                result.final_state = (
                    "recovery_required" if self._bin_execution_uncertain
                    else "calibration_required")
            result.message = f"bin_slot_calibration: {exc}"
            if recovery_errors:
                result.message += "; arm stop unconfirmed: " + "; ".join(
                    recovery_errors)
        return result

    def record_direct_calibration(self, model, measured_seed, goal_handle):
        object_position = self.query_entity_pose(
            model, goal_handle, frame="world").position
        link_name = f"{self.cfg['robot_model']}::{self.cfg['robot_link']}"
        link_position = self.query_entity_pose(
            link_name, goal_handle, frame="world").position
        distance = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(
                (object_position.x, object_position.y, object_position.z),
                (link_position.x, link_position.y, link_position.z))))
        self.get_logger().info(
            f"Direct pick calibration actual_joints="
            f"{list(measured_seed.position)}, "
            f"object_world=({object_position.x:.4f}, "
            f"{object_position.y:.4f}, {object_position.z:.4f}), "
            f"link_world=({link_position.x:.4f}, {link_position.y:.4f}, "
            f"{link_position.z:.4f}), distance={distance:.4f}m")

    @staticmethod
    def valid_classification_goal(request):
        expected_classes = {
            1: "comb", 2: "mouse", 3: "comb",
            4: "mouse", 5: "comb", 6: "mouse"}
        return (
            request.grid_id in expected_classes
            and request.object_class == expected_classes[request.grid_id]
            and request.object_model == f"pick_object_{request.grid_id}")

    def validate_classification_transport(self, request):
        invalid = []
        if self.dry_run:
            invalid.append("dry_run must be false")
        if self.cfg["direct_pick_calibration"]:
            invalid.append("direct_pick_calibration must be false")
        if self.cfg["bin_slot_calibration"]:
            invalid.append("bin_slot_calibration must be false")
        if self.cfg["lift_calibration"]:
            invalid.append("lift_calibration must be false")
        if self.cfg["local_center_calibration"]:
            invalid.append("local_center_calibration must be false")
        if not self.cfg["attach_enabled"]:
            invalid.append("attach_enabled must be true")
        if (self.cfg["gripper_open"] != 0.15
                or self.cfg["gripper_closed"] != -0.31):
            invalid.append("gripper positions must be open=0.15, close=-0.31")
        if invalid:
            raise CalibrationRequired(
                "classification transport rejected before motion: "
                + "; ".join(invalid))
        if not self.valid_classification_goal(request):
            raise UnsupportedGoal(
                "Supported goals require grid_id 1..6, matching "
                "pick_object_<grid_id>, and alternating classes "
                "comb/mouse from grid 1")
        destination = (
            "red_bin" if request.object_class == "comb" else "blue_bin")
        slot = ("left", "center", "right")[(request.grid_id - 1) // 2]
        release_key = f"{destination}.{slot}_release_joints"
        intermediate_key = f"{destination}.{slot}_intermediate_joints"
        if not self.cfg.get(intermediate_key):
            intermediate_key = None
        path = [
            f"grid_{request.grid_id}.pick_joints",
            f"grid_{request.grid_id}.lift_joints",
            "grid_1.clearance_joints",
            f"{destination}.approach_joints", release_key]
        if intermediate_key:
            path.append(intermediate_key)
        for name in path:
            values = self.cfg[name]
            if len(values) != 6:
                raise CalibrationRequired(
                    f"classification path unavailable: {name}")
            self.validate_joint_values(name, values)
        self.get_logger().info(
            f"Transport destination: release_key={release_key}, "
            f"intermediate_key={intermediate_key}")
        return destination, slot, release_key, intermediate_key

    def check_classification_runtime(self, goal_handle):
        self.check_bin_calibration_runtime(goal_handle)
        self.check_attachment_services(goal_handle)
        self.wait_ready(
            lambda: self.gripper.get_subscription_count() > 0,
            "gripper subscriber", goal_handle)

    def move_classification_joints(self, name, goal_handle):
        values = list(self.cfg[name])
        trajectory = self.plan_joints(values, goal_handle)
        self._transport_execution_uncertain = True
        self.execute_trajectory(trajectory, goal_handle)
        self._transport_execution_uncertain = False
        return self.current_ik_seed(
            goal_handle, after=time.monotonic())

    def classification_measurement(self, model, goal_handle, label):
        object_pose, link_pose = self.cycle_measurement(
            model, goal_handle, label)
        distance = math.dist(
            (object_pose.position.x, object_pose.position.y,
             object_pose.position.z),
            (link_pose.position.x, link_pose.position.y,
             link_pose.position.z))
        return object_pose, link_pose, distance

    def execute_classification_transport(self, goal_handle, result):
        request = goal_handle.request
        stage = "validating_transport"
        attached_once = False
        self._transport_execution_uncertain = False
        try:
            destination, slot, release_key, intermediate_key = (
                self.validate_classification_transport(request))
            destination_stages = [
                "moving_to_destination_approach"]
            if intermediate_key:
                destination_stages.append(
                    "moving_to_destination_intermediate")
            destination_stages.extend([
                "moving_to_destination_release",
                "validating_release_state", "detaching_object",
                "opening_after_detach", "waiting_for_object"])
            if intermediate_key:
                destination_stages.append(
                    "retreating_to_destination_intermediate")
            destination_stages.extend([
                "retreating_to_destination_approach", "sorting_completed"])
            stages = [
                "validating_transport", "opening_gripper",
                "moving_to_pick_joints", "local_center_correction",
                "closing_gripper", "checking_grasp", "attaching_object",
                "moving_to_lift_joints",
                "verifying_object_follow_after_lift",
                "moving_to_clearance", "verifying_object_follow"
            ] + destination_stages
            self.check_classification_runtime(goal_handle)
            for index, stage in enumerate(stages[1:], start=1):
                self.check_cancel(goal_handle)
                self.publish_feedback(
                    goal_handle, stage, (index + 1) / len(stages))
                if stage == "opening_gripper":
                    self.command_gripper(True, goal_handle)
                elif stage == "moving_to_pick_joints":
                    measured = self.move_classification_joints(
                        f"grid_{request.grid_id}.pick_joints", goal_handle)
                elif stage == "local_center_correction":
                    converged, count, residual = self.local_center_correction(
                        request.object_model, measured, goal_handle)
                    if not converged:
                        raise CalibrationRequired(
                            "local center did not converge: "
                            f"corrections={count}, residual={residual:.10f}m")
                    measured = self.current_ik_seed(goal_handle)
                elif stage == "closing_gripper":
                    self.command_gripper(False, goal_handle)
                    self.verify_closed_gripper()
                elif stage == "checking_grasp":
                    _, _, distance = self.classification_measurement(
                        request.object_model, goal_handle, stage)
                    if self.exceeds_limit(
                            distance, self.cfg["attach_max_distance"]):
                        raise CalibrationRequired(
                            f"attach distance={distance:.10f}m exceeds "
                            f"{self.cfg['attach_max_distance']:.10f}m")
                elif stage == "attaching_object":
                    self.attach_object(request.object_model, goal_handle)
                    if not self.attached:
                        raise RuntimeError("Attach success was not confirmed")
                    attached_once = True
                elif stage == "moving_to_lift_joints":
                    name = f"grid_{request.grid_id}.lift_joints"
                    self.validate_joint_values(name, self.cfg[name])
                    measured = self.move_classification_joints(
                        name, goal_handle)
                    self.check_bin_state_validity(
                        measured, goal_handle, stage)
                elif stage == "verifying_object_follow_after_lift":
                    if not self.attached:
                        raise RuntimeError("Attachment state was lost")
                    _, _, distance = self.classification_measurement(
                        request.object_model, goal_handle, stage)
                    if self.exceeds_limit(
                            distance, self.cfg["attach_max_distance"]):
                        raise RuntimeError(
                            "Object did not follow fixed lift: "
                            f"distance={distance:.10f}m")
                elif stage == "moving_to_clearance":
                    self.move_classification_joints(
                        "grid_1.clearance_joints", goal_handle)
                elif stage == "verifying_object_follow":
                    if not self.attached:
                        raise RuntimeError("Attachment state was lost")
                    _, _, distance = self.classification_measurement(
                        request.object_model, goal_handle, stage)
                    if self.exceeds_limit(
                            distance, self.cfg["attach_max_distance"]):
                        raise RuntimeError(
                            f"Object did not follow gripper: "
                            f"distance={distance:.10f}m")
                elif stage == "moving_to_destination_approach":
                    self.move_classification_joints(
                        f"{destination}.approach_joints", goal_handle)
                elif stage == "moving_to_destination_intermediate":
                    self.move_classification_joints(
                        intermediate_key, goal_handle)
                elif stage == "moving_to_destination_release":
                    self.move_classification_joints(
                        release_key, goal_handle)
                elif stage == "validating_release_state":
                    seed = self.current_ik_seed(goal_handle)
                    self.check_bin_state_validity(
                        seed, goal_handle, stage)
                    self.classification_measurement(
                        request.object_model, goal_handle, stage)
                elif stage == "detaching_object":
                    self.detach_object(request.object_model, goal_handle)
                    if self.attached:
                        raise RuntimeError("Detach success was not confirmed")
                elif stage == "opening_after_detach":
                    self.command_gripper(True, goal_handle)
                elif stage == "waiting_for_object":
                    self.pause(
                        self.cfg["object_settle_time"], goal_handle)
                    final_pose = self.query_entity_pose(
                        f"{request.object_model}::{self.cfg['object_link']}",
                        goal_handle, frame="world")
                    position = final_pose.position
                    self.get_logger().info(
                        f"Final object_world=({position.x:.10f}, "
                        f"{position.y:.10f}, {position.z:.10f})")
                elif stage == "retreating_to_destination_intermediate":
                    self.move_classification_joints(
                        intermediate_key, goal_handle)
                elif stage == "retreating_to_destination_approach":
                    self.move_classification_joints(
                        f"{destination}.approach_joints", goal_handle)
                elif stage == "sorting_completed":
                    goal_handle.succeed()
                    result.success = True
                    result.final_state = "sorting_completed"
                    result.message = (
                        f"Grid {request.grid_id} {request.object_class} "
                        f"transported to {destination} {slot}; "
                        "no return-home motion")
                    return result
        except UnsupportedGoal as exc:
            goal_handle.abort()
            result.success = False
            result.final_state = "unsupported_goal"
            result.message = str(exc)
        except Exception as exc:
            self.get_logger().error(f"{stage}: {exc}")
            goal_handle.abort()
            result.success = False
            if attached_once:
                self._faulted = True
                result.final_state = "recovery_required"
            else:
                result.final_state = "calibration_required"
            result.message = (
                f"{stage}: {exc}; attached={self.attached}; "
                "automatic detach/open/continued motion forbidden")
        return result

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        stage = "validating_goal"
        result = PickAndSort.Result()
        self._joint_name = (
            f"e3_{request.object_model}_{time.time_ns()}_grasp_joint")
        if self.cfg["grid_pick_calibration"]:
            try:
                return self.execute_grid_pick_callback(goal_handle, result)
            finally:
                self._task_lock.release()
        if self.cfg["bin_slot_calibration"]:
            try:
                return self.execute_bin_slot_callback(goal_handle, result)
            finally:
                self._task_lock.release()
        if self.cfg["classification_transport_enabled"]:
            try:
                return self.execute_classification_transport(
                    goal_handle, result)
            finally:
                self._task_lock.release()
        lift = bool(self.cfg["lift_calibration"])
        self._cycle_airborne = False
        self._cycle_detach_failed = False
        self._cycle_motion = False
        direct = (not self.dry_run
                  and self.cfg["direct_pick_calibration"]
                  and (not self.cfg["attach_enabled"] or lift))
        stages = [
            "validating_goal", "opening_gripper", "querying_object_pose",
            "moving_to_pregrasp", "moving_to_grid", "closing_gripper",
            "attaching_object", "lifting_object", "querying_destination",
            "moving_to_destination", "releasing_object", "detaching_object",
            "returning_home", "completed"]
        if direct:
            stages = [
                "validating_goal", "opening_gripper",
                "querying_object_pose", "moving_to_pick_joints",
                "recording_calibration"]
            if self.cfg["local_center_calibration"]:
                stages[-1] = "local_center_correction"
        if lift:
            stages = [
                "validating_goal", "opening_gripper", "querying_object_pose",
                "moving_to_pick_joints", "local_center_correction",
                "closing_gripper", "attaching_object",
                "lift_calibration", "calibration_cycle_completed"]
        try:
            if lift and (self.dry_run
                         or not self.cfg["direct_pick_calibration"]
                         or not self.cfg["attach_enabled"]
                         or not self.cfg["local_center_calibration"]
                         or request.grid_id != 1):
                raise CalibrationRequired(
                    "lift_calibration requires dry_run=false, "
                    "direct_pick_calibration=true, "
                    "local_center_calibration=true, "
                    "attach_enabled=true and grid_id=1; "
                    "no motion performed")
            if lift and (self.cfg["lift_calibration_height"] != 0.05
                         or self.cfg["gripper_closed"] != -0.31):
                raise CalibrationRequired(
                    "Restricted cycle requires height=0.05 and closure=-0.31")
            for index, stage in enumerate(stages):
                self.check_cancel(goal_handle)
                self.publish_feedback(
                    goal_handle, stage, (index + 1) / len(stages))
                if self.dry_run:
                    continue
                if stage == "validating_goal":
                    if self.cfg["local_center_calibration"] and not direct:
                        raise CalibrationRequired(
                            "local_center_calibration requires "
                            "direct_pick_calibration=true and "
                            "attach_enabled=false; no motion performed")

                    if (self.cfg["direct_pick_calibration"]
                            and self.cfg["attach_enabled"]
                            and not lift):
                        raise CalibrationRequired(
                            "direct_pick_calibration requires "
                            "attach_enabled=false; no motion performed")
                    if direct and request.grid_id != 1:
                        raise CalibrationRequired(
                            "Direct pick calibration supports grid 1 only; "
                            "no motion performed")
                    reference_seed = self.grid_reference_seed(
                        request.grid_id)
                    if self.cfg["attach_enabled"]:
                        self.check_attachment_services(goal_handle)
                elif stage == "opening_gripper":
                    self.command_gripper(True, goal_handle)
                elif stage == "querying_object_pose" and direct:
                    position = self.query_entity_pose(
                        request.object_model, goal_handle,
                        frame="world").position
                    self.get_logger().info(
                        f"Direct calibration initial object_world="
                        f"({position.x:.4f}, {position.y:.4f}, "
                        f"{position.z:.4f})")
                elif stage == "moving_to_pick_joints":
                    trajectory = self.plan_joints(
                        list(reference_seed.position), goal_handle)
                    self.execute_trajectory(trajectory, goal_handle)
                    measured_seed = self.current_ik_seed(
                        goal_handle, after=time.monotonic())
                elif stage == "local_center_correction":
                    converged, count, residual = self.local_center_correction(
                        request.object_model, measured_seed, goal_handle)
                    self.check_cancel(goal_handle)
                    if lift:
                        if not converged:
                            raise RuntimeError(
                                "Lift calibration center did not converge: "
                                f"distance={residual:.7f}m; "
                                f"corrections={count}")
                        measured_seed = self.current_ik_seed(goal_handle)
                        self.get_logger().info(
                            "Centered actual_joints="
                            f"{list(measured_seed.position)}")
                        continue
                    message = (
                        f"Local center calibration: corrections={count}, "
                        f"distance={residual:.7f}m; "
                        "holding current pose with gripper open")
                    if converged:
                        self.publish_feedback(
                            goal_handle, "calibration_completed", 1.0)
                        goal_handle.succeed()
                        result.success = True
                        result.final_state = "calibration_completed"
                        result.message = message + "; tolerance reached"
                        return result
                    raise CalibrationRequired(
                        message + "; iteration limit reached without "
                        "convergence; no further motion")
                elif stage == "lift_calibration":
                    self.lift_calibration_motion(
                        request.object_model, goal_handle)
                elif stage == "calibration_cycle_completed":
                    self.check_cancel(goal_handle)
                    if self.attached:
                        raise RuntimeError("Cycle attachment not released")
                    goal_handle.succeed()
                    result.success = True
                    result.final_state = "calibration_cycle_completed"
                    result.message = (
                        "Calibration lift, lowering and release completed; "
                        "no transfer or home motion")
                    return result
                elif stage == "recording_calibration":
                    self.record_direct_calibration(
                        request.object_model, measured_seed, goal_handle)
                    raise CalibrationRequired(
                        "Direct grid 1 joint calibration completed; "
                        "holding current pose with gripper open. "
                        "No IK, closure, attach, lift, transfer "
                        "or home motion.")
                elif stage == "querying_object_pose":
                    position = self.query_entity(
                        request.object_model, goal_handle)
                    grasp = self.target_pose(
                        position, self.cfg["grasp_offset"],
                        self.cfg["grasp_orientation"])
                    pregrasp = self.target_pose(
                        position, self.cfg["grasp_offset"],
                        self.cfg["grasp_orientation"],
                        self.cfg["pregrasp_height"])
                elif stage == "moving_to_pregrasp":
                    pregrasp_seed = self.move_pose(
                        pregrasp, goal_handle, purpose=stage,
                        reference_seed=reference_seed)
                elif stage == "moving_to_grid":
                    grasp_seed = self.move_pose(
                        grasp, goal_handle, purpose=stage,
                        reference_seed=pregrasp_seed)
                elif stage == "lifting_object":
                    self.move_pose(
                        pregrasp, goal_handle, purpose=stage,
                        reference_seed=grasp_seed)
                elif stage == "closing_gripper":
                    self.command_gripper(False, goal_handle)
                elif stage == "attaching_object":
                    self.attach_object(request.object_model, goal_handle)
                elif stage == "querying_destination":
                    model = (
                        "red_bin" if request.object_class == "comb"
                        else "blue_bin")
                    destination = self.target_pose(
                        self.query_entity(model, goal_handle),
                        self.cfg["release_offset"],
                        self.cfg["release_orientation"])
                elif stage == "moving_to_destination":
                    self.move_pose(destination, goal_handle, purpose=stage)
                elif stage == "releasing_object":
                    self.command_gripper(True, goal_handle)
                elif stage == "detaching_object":
                    self.detach_object(request.object_model, goal_handle)
                elif stage == "returning_home":
                    self.execute_trajectory(
                        self.plan_joints(self.cfg["home_joints"], goal_handle),
                        goal_handle)
            self.check_cancel(goal_handle)
            goal_handle.succeed()
            result.success = True
            result.final_state = "completed"
            result.message = (
                f"{'Dry-run' if self.dry_run else 'MoveIt pick-and-sort'} "
                "completed: "
                f"grid={request.grid_id}, class={request.object_class}, "
                f"model={request.object_model}")
        except CalibrationRequired as exc:
            self.check_cancel_calibration(goal_handle, result, exc)
        except Exception as exc:
            cancelled = (
                isinstance(exc, TaskCancelled)
                or goal_handle.is_cancel_requested)
            self.get_logger().error(f"{stage}: {exc}")
            errors = (self.recover_cycle(request.object_model) if lift
                      else self.recover(request.object_model))
            result.success = False
            result.final_state = "canceled" if cancelled else f"failed_{stage}"
            if lift and errors:
                result.final_state = "recovery_required"
            result.message = f"{stage}: {exc}"
            result.message += (
                "; recovery incomplete: " + "; ".join(errors)
                if errors else "; recovery completed")
            if cancelled:
                goal_handle.canceled()
            else:
                goal_handle.abort()
        finally:
            if self._task_lock.locked():
                self._task_lock.release()
        return result

    def check_cancel_calibration(self, goal_handle, result, error):
        result.success = False
        if goal_handle.is_cancel_requested:
            errors = self.recover(goal_handle.request.object_model)
            goal_handle.canceled()
            result.final_state = "canceled"
            result.message = "Calibration canceled; " + (
                "; ".join(errors) if errors else "recovery completed")
        else:
            goal_handle.abort()
            result.final_state = "calibration_required"
            result.message = str(error)
            self.get_logger().warning(result.message)

    def destroy_node(self):
        self.server.destroy()
        if (not self.dry_run or self.cfg["bin_slot_calibration"]
                or self.cfg["grid_pick_calibration"]):
            self.arm_client.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PickActionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
