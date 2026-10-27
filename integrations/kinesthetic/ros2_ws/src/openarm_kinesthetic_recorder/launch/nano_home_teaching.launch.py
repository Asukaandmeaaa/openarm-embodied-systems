"""One-click fixed lifted HOME initialization followed by real teaching capture."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("openarm_kinesthetic_recorder")
    runtime_config = os.path.join(share, "config", "real_kinesthetic.yaml")
    home_config = os.path.join(share, "config", "home_initialization.yaml")
    allow = LaunchConfiguration("allow_real_execution")
    device = LaunchConfiguration("gamepad_device")
    output_root = LaunchConfiguration("output_root")
    start_global_camera = LaunchConfiguration("start_global_camera")
    diagnostic_only = LaunchConfiguration("diagnostic_only")
    global_camera = GroupAction(
        scoped=True,
        forwarding=False,
        condition=IfCondition(start_global_camera),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
                ),
                launch_arguments={
                    "camera_namespace": "camera",
                    "camera_name": "global_camera",
                    "serial_no": "'344322073972'",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "align_depth.enable": "true",
                }.items(),
            )
        ],
    )

    initializer = Node(
        package="openarm_kinesthetic_recorder",
        executable="home_guarded_executor",
        name="home_guarded_executor",
        output="screen",
        arguments=["--config", home_config, "--allow-real-execution"],
    )
    gamepad = Node(
        package="openarm_kinesthetic_recorder",
        executable="real_gamepad",
        name="openarm_real_gamepad",
        output="screen",
        parameters=[
            runtime_config,
            {"device": device, "diagnostic_only": diagnostic_only},
        ],
    )
    bridge = Node(
        package="openarm_kinesthetic_recorder",
        executable="right_kinesthetic_bridge",
        name="right_kinesthetic_bridge",
        output="screen",
        parameters=[runtime_config, {"diagnostic_only": diagnostic_only}],
    )
    wrist = Node(
        package="openarm_kinesthetic_recorder",
        executable="wrist_camera",
        name="wrist_right_camera",
        output="screen",
        condition=UnlessCondition(diagnostic_only),
        parameters=[runtime_config],
    )
    recorder = Node(
        package="openarm_kinesthetic_recorder",
        executable="real_episode_recorder",
        name="real_episode_recorder",
        output="screen",
        condition=UnlessCondition(diagnostic_only),
        parameters=[runtime_config, {"output_root": output_root}],
    )
    motion_recorder = Node(
        package="openarm_kinesthetic_recorder",
        executable="right_motion_recorder",
        name="right_motion_recorder",
        output="screen",
        condition=UnlessCondition(diagnostic_only),
        parameters=[runtime_config],
    )
    status = Node(
        package="openarm_kinesthetic_recorder",
        executable="teaching_status",
        name="teaching_status_console",
        output="screen",
        parameters=[runtime_config],
    )

    def authorize(context):
        if allow.perform(context) != "true":
            raise RuntimeError("fixed HOME motion requires allow_real_execution:=true")
        return []

    def after_home(event, _context):
        if event.returncode != 0:
            return [LogInfo(msg="WORKFLOW FAIL_RIGID: HOME failed; teaching blocked")]
        return [
            LogInfo(
                msg=(
                    "WORKFLOW HOME_READY -> TEACHING: LT Cartesian jog, LT+LB drag, "
                    "A/Y or LT+LB+L3 gripper, X motion recorder are ready"
                )
            ),
            wrist,
            gamepad,
            bridge,
            recorder,
            motion_recorder,
            status,
        ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "allow_real_execution", default_value="false", choices=["true", "false"]
            ),
            DeclareLaunchArgument("gamepad_device", default_value="auto"),
            DeclareLaunchArgument(
                "diagnostic_only",
                default_value="false",
                choices=["true", "false"],
                description=(
                    "after HOME, publish mapping/IK diagnostics without arm, "
                    "passive, gripper, camera, or recorder actuation"
                ),
            ),
            DeclareLaunchArgument(
                "start_global_camera",
                default_value="true",
                choices=["true", "false"],
                description=(
                    "start the fixed D435i for recording; this never starts table detection"
                ),
            ),
            DeclareLaunchArgument(
                "output_root", default_value="~/openarm_kinesthetic_data/raw"
            ),
            LogInfo(
                msg=(
                    "WORKFLOW FIXED_LIFTED_HOME: fresh rigid zero -> fixed inflated "
                    "collision -> side prehome -> lifted HOME -> teaching + recording"
                )
            ),
            OpaqueFunction(function=authorize),
            global_camera,
            RegisterEventHandler(OnProcessExit(target_action=initializer, on_exit=after_home)),
            initializer,
        ]
    )
