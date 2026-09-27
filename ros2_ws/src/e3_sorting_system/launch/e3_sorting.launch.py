"""Formal E3 system: include the simulator once and gate observation."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("e3_sorting_system")
    moveit_share = get_package_share_directory("mecharm_270_moveit_config")
    config = os.path.join(share, "config", "sorting_config.yaml")

    def include(directory, filename):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(directory, "launch", filename)))

    return LaunchDescription([
        include(share, "e3_sim.launch.py"),
        include(moveit_share, "move_group.launch.py"),
        Node(package="e3_sorting_system", executable="object_detector",
             name="e3_object_detector", output="screen",
             parameters=[config, {"use_sim_time": True}]),
        Node(package="e3_sorting_system", executable="pick_action_server",
             name="e3_pick_action_server", output="screen",
             parameters=[config, {
                 "use_sim_time": True, "dry_run": False,
                 "classification_transport_enabled": True,
                 "attach_enabled": True, "lift_calibration": False,
                 "direct_pick_calibration": False,
                 "local_center_calibration": False,
                 "bin_slot_calibration": False,
                 "grid_pick_calibration": False,
                 "calibration_execute": False}]),
        Node(package="e3_sorting_system", executable="task_manager",
             name="task_manager", output="screen",
             parameters=[config, {"use_sim_time": True}]),
    ])
