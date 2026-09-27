#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class ObjectDetector(Node):

    def __init__(self):
        super().__init__("e3_object_detector")

        self.declare_parameter(
            "image_topic",
            "/e3/camera/overhead_camera/image_raw",
        )
        self.declare_parameter(
            "detections_topic",
            "/e3/detections",
        )
        self.declare_parameter(
            "annotated_topic",
            "/e3/detections/image",
        )
        self.declare_parameter("minimum_area", 120.0)
        self.declare_parameter("maximum_area", 4000.0)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter(
            "detections_topic"
        ).value
        annotated_topic = self.get_parameter(
            "annotated_topic"
        ).value

        self.minimum_area = float(
            self.get_parameter("minimum_area").value
        )
        self.maximum_area = float(
            self.get_parameter("maximum_area").value
        )

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            detections_topic,
            10,
        )
        self.annotated_pub = self.create_publisher(
            Image,
            annotated_topic,
            10,
        )

        self.frame_count = 0

        self.get_logger().info(
            f"Subscribing to image topic: {image_topic}"
        )
        self.get_logger().info(
            f"Publishing detections on: {detections_topic}"
        )

    def find_objects(self, hsv_image, object_class):
        if object_class == "comb":
            lower_1 = np.array([0, 120, 100], dtype=np.uint8)
            upper_1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_2 = np.array([170, 120, 100], dtype=np.uint8)
            upper_2 = np.array([179, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(
                hsv_image,
                lower_1,
                upper_1,
            )
            mask |= cv2.inRange(
                hsv_image,
                lower_2,
                upper_2,
            )
        else:
            lower = np.array([100, 120, 80], dtype=np.uint8)
            upper = np.array([135, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_image, lower, upper)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        objects = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.minimum_area:
                continue

            # Large red/blue classification regions are excluded here.
            if area > self.maximum_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            objects.append(
                {
                    "class": object_class,
                    "area": area,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )

        return objects

    def make_detection(self, item, header):
        detection = Detection2D()
        detection.header = header

        bbox = BoundingBox2D()
        bbox.center.position.x = (
            float(item["x"]) + float(item["width"]) / 2.0
        )
        bbox.center.position.y = (
            float(item["y"]) + float(item["height"]) / 2.0
        )
        bbox.center.theta = 0.0
        bbox.size_x = float(item["width"])
        bbox.size_y = float(item["height"])
        detection.bbox = bbox

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item["class"]

        normalized_area = min(
            1.0,
            item["area"] / self.maximum_area,
        )
        hypothesis.hypothesis.score = min(
            0.99,
            0.75 + 0.20 * normalized_area,
        )

        detection.results.append(hypothesis)
        return detection

    def draw_detection(self, image, item, score):
        x = item["x"]
        y = item["y"]
        width = item["width"]
        height = item["height"]

        if item["class"] == "comb":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            2,
        )

        label = f'{item["class"]} {score:.2f}'
        cv2.putText(
            image,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().error(
                f"Image conversion failed: {error}"
            )
            return

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        objects = []
        objects.extend(self.find_objects(hsv_image, "comb"))
        objects.extend(self.find_objects(hsv_image, "mouse"))

        objects.sort(
            key=lambda item: (
                item["y"],
                item["x"],
            )
        )

        detection_array = Detection2DArray()
        detection_array.header = message.header

        annotated = image.copy()

        for item in objects:
            detection = self.make_detection(
                item,
                message.header,
            )
            detection_array.detections.append(detection)

            score = detection.results[0].hypothesis.score
            self.draw_detection(annotated, item, score)

        self.detections_pub.publish(detection_array)

        annotated_message = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding="bgr8",
        )
        annotated_message.header = message.header
        self.annotated_pub.publish(annotated_message)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            classes = [
                item["class"]
                for item in objects
            ]
            self.get_logger().info(
                f"Detected {len(objects)} objects: {classes}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()

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
