"""Camera-only DECXIN wrist launch; it has no arm-control dependencies."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("openarm_kinesthetic_recorder"),
        "config",
        "real_kinesthetic.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="openarm_kinesthetic_recorder",
                executable="wrist_camera",
                name="wrist_right_camera",
                output="screen",
                parameters=[config],
            )
        ]
    )
