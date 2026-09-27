"""Single-round, serial sorting after a complete stable observation."""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from e3_sorting_interfaces.action import PickAndSort
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray


class SortingRound:
    """Keep detection assignment and dispatch independent of the ROS graph."""

    def __init__(self, centers, frames, distance, confidence, timeout, now):
        self.centers = centers
        self.frames = frames
        self.distance = distance
        self.confidence = confidence
        self.deadline = now + timeout
        self.state = "observing"
        self.message = "Waiting for all configured grids"
        self.count = 0
        self.last = {}
        self.queue = []
        self.done = set()
        self.active = None

    def fail(self, message):
        self.state = "failed"
        self.message = message

    def observe(self, detections, now):
        if self.state != "observing":
            return
        if now >= self.deadline:
            self.tick(now)
            return
        assigned = {}
        duplicates = set()
        for detection in detections:
            if not detection.results:
                continue
            hypothesis = max(
                (r.hypothesis for r in detection.results),
                key=lambda h: h.score)
            label, score = hypothesis.class_id, hypothesis.score
            point = detection.bbox.center.position
            if (label not in ("comb", "mouse")
                    or not all(math.isfinite(v)
                               for v in (point.x, point.y, score))
                    or score < self.confidence):
                continue
            grid = min(self.centers, key=lambda g: math.dist(
                self.centers[g], (point.x, point.y)))
            distance = math.dist(self.centers[grid], (point.x, point.y))
            if distance > self.distance:
                continue
            if label != ("comb" if grid % 2 else "mouse"):
                continue
            if grid in assigned:
                duplicates.add(grid)
            assigned[grid] = label
        for grid in duplicates:
            assigned.pop(grid, None)
        complete = set(assigned) == set(self.centers)
        same = complete and assigned == self.last
        self.count = self.count + 1 if same else (
            1 if complete else 0)
        self.last = assigned
        if self.count >= self.frames:
            self.queue = [(g, assigned[g]) for g in sorted(self.centers)]
            self.state = "ready"
            self.message = "All grids stable; queue frozen for this round"

    def tick(self, now):
        if self.state == "observing" and now >= self.deadline:
            missing = sorted(set(self.centers) - set(self.last))
            self.fail(
                f"Detection timeout; missing grids={missing}; "
                f"stable frames={self.count}/{self.frames}")

    def take(self):
        if self.state != "ready" or self.active is not None:
            return None
        self.active = self.queue.pop(0)
        self.state = "active"
        grid, label = self.active
        return grid, label, f"pick_object_{grid}"

    def finish(self, status, result):
        if self.state != "active":
            return
        if (status != GoalStatus.STATUS_SUCCEEDED or not result.success
                or result.final_state != "sorting_completed"):
            self.fail(
                f"Action stopped at grid={self.active[0]}; status={status}; "
                f"state={result.final_state}; {result.message}")
            return
        self.done.add(self.active[0])
        self.active = None
        self.state = "ready" if self.queue else "sorting_completed"
        self.message = (
            f"Completed grids={sorted(self.done)}; state={self.state}")


class TaskManager(Node):
    """Observe once and send at most one outstanding PickAndSort goal."""

    def __init__(self):
        super().__init__("task_manager")
        defaults = {
            "detections_topic": "/e3/detections",
            "action_name": "/e3/pick_and_sort",
            "detection_wait_timeout": 5.0, "action_wait_timeout": 15.0,
            "startup_wait_timeout": 120.0,
            "maximum_assignment_distance": 45.0, "minimum_confidence": 0.70,
            "stable_detection_frames": 3, "grid_ids": list(range(1, 7)),
            "supported_classes": ["comb", "mouse"],
            "comb.destination_model": "red_bin",
            "mouse.destination_model": "blue_bin",
        }
        self.cfg = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            self.cfg[name] = self.get_parameter(name).value
        if (self.cfg["grid_ids"] != list(range(1, 7))
                or set(self.cfg["supported_classes"]) != {"comb", "mouse"}
                or self.cfg["comb.destination_model"] != "red_bin"
                or self.cfg["mouse.destination_model"] != "blue_bin"):
            raise ValueError("Only the six-grid class mapping is valid")
        for name in ("detection_wait_timeout", "action_wait_timeout",
                     "maximum_assignment_distance", "startup_wait_timeout"):
            if not math.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (self.cfg["stable_detection_frames"] < 1
                or not 0 <= self.cfg["minimum_confidence"] <= 1):
            raise ValueError("Invalid stability/confidence settings")
        centers = {}
        for grid in self.cfg["grid_ids"]:
            name = f"grid_{grid}.pixel_center"
            self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            center = self.get_parameter(name).value
            if (center is None or len(center) != 2
                    or not all(math.isfinite(v) for v in center)):
                raise ValueError(f"{name} requires two finite coordinates")
            centers[grid] = tuple(center)
            model_key = f"grid_{grid}.model_name"
            self.declare_parameter(model_key, f"pick_object_{grid}")
            if self.get_parameter(model_key).value != f"pick_object_{grid}":
                raise ValueError(f"Invalid {model_key}")
        if len(set(centers.values())) != 6:
            raise ValueError("Pixel centers must be distinct")
        self.round = SortingRound(
            centers, self.cfg["stable_detection_frames"],
            self.cfg["maximum_assignment_distance"],
            self.cfg["minimum_confidence"],
            self.cfg["detection_wait_timeout"], time.monotonic())
        self.client = ActionClient(
            self, PickAndSort, self.cfg["action_name"])
        self.publisher = self.create_publisher(
            String, "/e3/sorting_status", 10)
        self.subscription = self.create_subscription(
            Detection2DArray, self.cfg["detections_topic"],
            self.on_detections, qos_profile_sensor_data)
        self.pending = None
        self.goal_handle = None
        self.phase = None
        self.wait_started = None
        self.last_report = None
        self.startup_phase = "waiting"
        self.startup_deadline = (
            time.monotonic() + self.cfg["startup_wait_timeout"])
        self.preparation = None
        self.prepare_client = self.create_client(
            Trigger, "/e3/prepare_observation")
        self.round.state = "waiting"
        self.round.message = (
            "Waiting for action, detector and observation service")
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(
            "Waiting for controlled observation preparation; "
            "no motion is issued by the constructor.")

    def on_detections(self, message):
        self.round.observe(message.detections, time.monotonic())

    def report(self):
        report = (self.round.state, self.round.message)
        if report != self.last_report:
            self.last_report = report
            message = String()
            message.data = f"{report[0]}: {report[1]}"
            self.publisher.publish(message)
            self.get_logger().info(message.data)

    def tick(self):
        now = time.monotonic()
        try:
            if not self.startup(now):
                self.report()
                return
            self.round.tick(now)
            self.advance(now)
        except Exception as exc:
            self.round.fail(f"Action transport error; no further goals: {exc}")
        self.report()

    def startup(self, now):
        if self.round.state == "failed":
            return False
        if self.startup_phase == "done":
            return True
        if self.startup_phase == "waiting":
            missing = []
            if not self.client.server_is_ready():
                missing.append("/e3/pick_and_sort")
            if not self.prepare_client.service_is_ready():
                missing.append("/e3/prepare_observation")
            if self.count_publishers(self.cfg["detections_topic"]) == 0:
                missing.append(self.cfg["detections_topic"])
            if missing:
                self.round.message = "Waiting for: " + ", ".join(missing)
                if now >= self.startup_deadline:
                    self.round.fail("Startup timeout; " + self.round.message)
                return False
            self.preparation = self.prepare_client.call_async(
                Trigger.Request())
            self.startup_phase = "preparing"
            self.round.state = "preparing_observation"
            self.round.message = "Preparing observation through action server"
            return False
        if not self.preparation.done():
            return False
        response = self.preparation.result()
        if not response.success:
            self.round.fail(response.message)
            return False
        self.startup_phase = "done"
        self.round.state = "observing"
        self.round.count = 0
        self.round.last = {}
        self.round.deadline = now + self.cfg["detection_wait_timeout"]
        self.round.message = "Observation ready; waiting for six stable grids"
        return True

    def advance(self, now):
        if self.round.state in ("failed", "sorting_completed", "observing"):
            return
        if self.pending is not None:
            if not self.pending.done():
                return
            response = self.pending.result()
            if self.phase == "acceptance":
                if not response.accepted:
                    self.round.fail("Action goal rejected; round stopped")
                    return
                self.goal_handle = response
                self.pending = response.get_result_async()
                self.phase = "result"
                return
            self.round.finish(response.status, response.result)
            self.pending = None
            self.goal_handle = None
            self.wait_started = None
            return
        if self.round.state != "ready":
            return
        if self.wait_started is None:
            self.wait_started = now
        if not self.client.server_is_ready():
            if now - self.wait_started >= self.cfg["action_wait_timeout"]:
                self.round.fail("Action server unavailable; round stopped")
            return
        grid, label, model = self.round.take()
        goal = PickAndSort.Goal(
            grid_id=grid, object_class=label, object_model=model)
        self.pending = self.client.send_goal_async(goal)
        self.phase = "acceptance"


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
