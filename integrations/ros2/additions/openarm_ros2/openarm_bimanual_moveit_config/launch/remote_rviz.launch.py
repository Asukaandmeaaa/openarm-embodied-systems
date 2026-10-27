import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    description_pkg = get_package_share_directory("openarm_description")
    moveit_pkg = get_package_share_directory("openarm_bimanual_moveit_config")

    config_dir = "openarm_v2.0"

    xacro_path = os.path.join(
        description_pkg,
        "assets",
        "robot",
        config_dir,
        "urdf",
        "openarm_v20.urdf.xacro",
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "openarm",
            package_name="openarm_bimanual_moveit_config",
        )
        .robot_description(
            file_path=xacro_path,
            mappings={
                "arm_type": "openarm_v2.0",
                "bimanual": "true",
                "use_fake_hardware": "true",
                "ros2_control": "true",
            },
        )
        .robot_description_semantic(
            file_path="config/openarm_v2.0/openarm_bimanual.srdf"
        )
        .robot_description_kinematics(
            file_path="config/openarm_v2.0/kinematics.yaml"
        )
        .joint_limits(
            file_path="config/openarm_v2.0/joint_limits.yaml"
        )
        .trajectory_execution(
            file_path="config/openarm_v2.0/moveit_controllers.yaml"
        )
        .planning_pipelines(
            pipelines=["ompl"],
            default_planning_pipeline="ompl",
        )
        .to_moveit_configs()
    )

    rviz_config = os.path.join(
        moveit_pkg,
        "config",
        config_dir,
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([rviz_node])
