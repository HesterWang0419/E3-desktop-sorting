import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    e3_share = get_package_share_directory("e3_sorting_system")
    e5_share = get_package_share_directory("e5_arm_sim")
    gazebo_share = get_package_share_directory("gazebo_ros")
    description_share = get_package_share_directory(
        "mycobot_description"
    )

    world_file = os.path.join(
        e3_share,
        "worlds",
        "e3_sorting.world",
    )
    xacro_file = os.path.join(
        e5_share,
        "urdf",
        "mecharm_270_m5_sim.urdf.xacro",
    )

    gazebo_model_path = os.pathsep.join(
        [
            os.path.dirname(description_share),
            os.environ.get("GAZEBO_MODEL_PATH", ""),
        ]
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_share,
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "world": world_file,
            "verbose": "true",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_mecharm",
        output="screen",
        arguments=[
            "-entity",
            "mecharm_270",
            "-topic",
            "robot_description",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.0",
        ],
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_position_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    delayed_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster, arm_controller,
                     gripper_controller]))

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GAZEBO_MODEL_DATABASE_URI",
                value="",
            ),
            SetEnvironmentVariable(
                name="GAZEBO_MODEL_PATH",
                value=gazebo_model_path,
            ),
            gazebo,
            robot_state_publisher,
            spawn_robot,
            delayed_controllers,
        ]
    )
