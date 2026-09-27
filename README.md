# E3 formal desktop sorting system

## Build and launch

Requires Ubuntu 22.04 / ROS 2 Humble and the existing shared e5_arm_sim,
mycobot_description, Gazebo ROS2 control, MoveIt and attachment-plugin underlay.
Source that underlay before building E3. E5 sources are reused, not modified.

    source /opt/ros/humble/setup.bash
    # Source your installed shared robot/attachment underlay here.
    cd ~/robot-integration-project-1/E3_desktop_sorting/ros2_ws
    colcon build --symlink-install --packages-select e3_sorting_interfaces e3_sorting_system mecharm_270_moveit_config
    source install/setup.bash
    ros2 launch e3_sorting_system e3_sorting.launch.py

Use a clean session without an existing E3 Gazebo/MoveIt/controller stack.
The launch includes the simulator once and starts move_group, detector,
action server and task manager once each. RViz is optional and not launched.
All runtime nodes use simulation time; move_group config is permanent.

The formal launch authorizes automatic simulated motion. It overrides safe
standalone YAML defaults: dry_run=false, transport=true, attach_enabled=true.
All calibration flags are false. The shared YAML is passed to the E3 nodes.
It does not change calibrated poses or attach timeouts.

## Startup and sorting

1. Gazebo loads the existing world/plugins and spawns the robot.
   Controller spawners are triggered by spawn completion and wait on
   controller-manager services, replacing the fixed six-second delay.
2. Task manager waits for PickAndSort, detection publisher and
   /e3/prepare_observation (std_srvs/Trigger), with a 120-second startup limit.
3. Preparation is requested once. The server uses the transportation task lock,
   rejects busy/faulted/attached/pending states, checks fresh joints and runtime
   endpoints, opens, plans red_bin.approach_joints, checks target validity,
   executes, checks fresh actual joints and resulting state.
4. Only success starts the observation deadline. Three complete consecutive
   frames are required. Missing grids at timeout are explicitly reported.
   Constructors never command motion.
5. Freeze the six-goal queue, execute serially, process each grid once.
   Objects disappearing after removal do not trigger re-detection waits.
6. Pick -> local correction -> close/check/attach -> fixed per-grid lift ->
   follow check -> common clearance -> follow check -> bin path.
   Grid 1/3/5: red left/center/right; grid 2/4/6: blue left/center/right.
   Only the red center currently has an intermediate waypoint.
7. Detach success precedes opening and reverse retreat. Six successes produce
   sorting_completed in logs and /e3/sorting_status (std_msgs/String).

Preparation failure or rejected/failed/canceled Action stops the round.
No automatic preparation/attach retries, fault reset, uncertain detach or
continuation after failure. Existing attach timeout and recovery are retained.
Ctrl+C shuts down launch; inspect uncertain robot/attachment state before restart.
No automatic recovery motion is performed.

## Offline tests

    python3 -m pytest -q src/e3_sorting_system/test/test_task_manager.py src/e3_sorting_system/test/test_pick_action_server.py src/e3_sorting_system/test/test_bin_calibration.py src/e3_sorting_system/test/test_formal_launch.py -k 'not ros_action_transport'

Tests mock motion/services. Final Gazebo startup, shutdown and six-object
demonstration still need live validation. See FINAL_REVIEW.md.
