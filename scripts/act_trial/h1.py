"""Explicit ACT H1: same closed startup as collection, streamed q/qd JTC.

Importing this file does not start hardware. The inherited main requires
--physical and performs all existing ownership/startup checks.
"""
from pathlib import Path
from vision_grasp_baseline import collection_h1
from vision_grasp_baseline.recovery_launch import linear_planner_node, BASE


def act_node(kwargs):
    kwargs = linear_planner_node(kwargs)
    if (kwargs.get('package') == 'controller_manager' and kwargs.get('executable') == 'spawner'
            and 'right_joint_trajectory_controller' in kwargs.get('arguments', [])):
        source = str(BASE/'config/right_arm_position_velocity.yaml')
        replacement = str(Path(__file__).with_name('position_velocity.yaml'))
        if kwargs['arguments'].count(source) != 1:
            raise ValueError('cannot bind ACT streaming controller configuration')
        kwargs['arguments'] = [replacement if v == source else v for v in kwargs['arguments']]
    return kwargs


if __name__ == '__main__':
    collection_h1.main(node_transform=act_node)
