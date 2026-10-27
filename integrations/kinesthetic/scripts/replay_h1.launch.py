"""Original ROS launch with the unused left gripper JTC omitted.

Matches the existing right-arm setup's controller selection. Hardware, gains,
URDF limits and original ROS endpoints remain unchanged; no ACT or gateway.
"""
import importlib.util
from pathlib import Path


def replay_spawner_arguments(arguments):
    values = list(arguments)
    if 'left_gripper_controller' in values and 'right_gripper_controller' in values:
        values.remove('left_gripper_controller')
    return values


def load_original_launch():
    from ament_index_python.packages import get_package_share_directory
    source = Path(get_package_share_directory('openarm_bimanual_moveit_config'))/'launch/demo.launch.py'
    spec = importlib.util.spec_from_file_location('openarm_original_replay_h1', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original


def configure_original_launch(original):
    original_node = original.Node

    def replay_node(*args, **kwargs):
        if kwargs.get('package') == 'controller_manager' and kwargs.get('executable') == 'spawner':
            kwargs['arguments'] = replay_spawner_arguments(kwargs.get('arguments', []))
        return original_node(*args, **kwargs)

    original.Node = replay_node
    return original.generate_launch_description()


def generate_launch_description():
    return configure_original_launch(load_original_launch())
