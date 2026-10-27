"""Inspect launch construction only; never run LaunchService or robot nodes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

PATH = Path('/home/lushilin/openarm_kinesthetic_ws/scripts/replay_h1.launch.py')
spec = importlib.util.spec_from_file_location('replay_h1', PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_spawner_only_removes_unused_left_gripper():
    args = ['left_gripper_controller', 'right_gripper_controller', '-c', '/controller_manager']
    assert replay.replay_spawner_arguments(args) == ['right_gripper_controller', '-c', '/controller_manager']
    assert args[0] == 'left_gripper_controller'
    for values in [['left_joint_trajectory_controller', 'right_joint_trajectory_controller'],
                   ['joint_state_broadcaster']]:
        assert replay.replay_spawner_arguments(values) == values


def test_original_installed_launch_keeps_both_arms_and_right_gripper():
    original = replay.load_original_launch()
    real_node = original.Node
    recorded = []
    def capture(*args, **kwargs):
        recorded.append(kwargs)
        return real_node(*args, **kwargs)
    original.Node = capture
    # Construction only: OpaqueFunctions for hardware/MoveIt are not executed.
    description = replay.configure_original_launch(original)
    assert description.entities
    original.controller_spawner(SimpleNamespace(perform_substitution=lambda _: 'joint_trajectory_controller'), None)
    controllers = [k['arguments'] for k in recorded if k.get('executable') == 'spawner']
    assert ['right_gripper_controller', '-c', '/controller_manager'] in controllers
    assert ['left_joint_trajectory_controller', 'right_joint_trajectory_controller', '-c', '/controller_manager'] in controllers
    assert all('left_gripper_controller' not in args for args in controllers)
    assert any('joint_state_broadcaster' in args for args in controllers)
    assert all('remappings' not in k for k in recorded)


def test_non_spawner_node_parameters_unchanged():
    seen = []
    original = SimpleNamespace(Node=lambda **kw: seen.append(kw), generate_launch_description=lambda: None)
    replay.configure_original_launch(original)
    parameters = [{'robot_description': 'unchanged'}, 'controllers.yaml']
    original.Node(package='controller_manager', executable='ros2_control_node', parameters=parameters)
    assert seen == [dict(package='controller_manager', executable='ros2_control_node', parameters=parameters)]
