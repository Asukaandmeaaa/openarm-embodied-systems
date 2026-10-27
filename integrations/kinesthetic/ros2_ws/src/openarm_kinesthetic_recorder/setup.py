from glob import glob
from os.path import join

from setuptools import find_packages, setup

setup(
    name="openarm_kinesthetic_recorder", version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/openarm_kinesthetic_recorder"]),
        ("share/openarm_kinesthetic_recorder", ["package.xml"]),
        (join("share", "openarm_kinesthetic_recorder", "launch"), glob("launch/*.launch.py")),
        (join("share", "openarm_kinesthetic_recorder", "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"], tests_require=["pytest"], zip_safe=True,
    maintainer="OpenArm local project", maintainer_email="local@example.invalid",
    description="Direct-kinesthetic operator bridge and recorder core", license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "real_gamepad = openarm_kinesthetic_recorder.gamepad_node:main",
            "right_kinesthetic_bridge = openarm_kinesthetic_recorder.hardware_bridge:main",
            "teaching_status = openarm_kinesthetic_recorder.teaching_status_node:main",
            "wrist_camera = openarm_kinesthetic_recorder.wrist_camera_node:main",
            "home_guarded_executor = openarm_kinesthetic_recorder.home_guarded_executor:main",
            "real_episode_recorder = openarm_kinesthetic_recorder.real_episode_recorder:main",
            "episode_capture_button = openarm_kinesthetic_recorder.episode_capture_button:main",
            "right_motion_recorder = openarm_kinesthetic_recorder.motion_recorder_node:main",
            "recorded_motion_player = openarm_kinesthetic_recorder.recorded_motion_player:main",
            "act_executor = openarm_kinesthetic_recorder.act_executor:main",
        ],
    },
)
