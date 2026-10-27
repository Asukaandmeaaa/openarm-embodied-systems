"""Pure collection startup tests: never initialize ROS/CAN or launch motors."""
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from vision_grasp_baseline.collection_h1 import description_transform,require_library_marker


def robot():
    return '<robot>'+''.join(
        f'<ros2_control name="{side}"><hardware><plugin>openarm_hardware/OpenArmHW</plugin>'
        f'<param name="arm_prefix">{side}_</param><param name="ee_type">pinch_gripper</param>'
        '<param name="hand">true</param></hardware></ros2_control>'
        for side in ('right','left'))+'</robot>'


def test_only_collection_right_gripper_opts_into_closed_startup():
    original=robot();xml=description_transform(original)
    root=ET.fromstring(xml)
    right,left=[c.find('hardware') for c in root.findall('ros2_control')]
    assert right.findtext("param[@name='startup_gripper_closed']")=='true'
    assert left.find("param[@name='startup_gripper_closed']") is None
    assert 'startup_gripper_closed' not in original
    with pytest.raises(ValueError,match='preexisting'):description_transform(xml)


@pytest.mark.parametrize('old,new',[
    ('openarm_hardware/OpenArmHW','mock_components/GenericSystem'),
    ('right_','left_'),('pinch_gripper','parallel_link')])
def test_unsupported_hardware_refused(old,new):
    with pytest.raises(ValueError):description_transform(robot().replace(old,new))


def test_library_requires_explicit_new_startup_marker(monkeypatch):
    monkeypatch.setattr('vision_grasp_baseline.collection_h1.subprocess.run',
                        lambda *a,**k:SimpleNamespace(stdout='T openarm_gripper_startup_handoff_version\n'))
    with pytest.raises(ValueError,match='install-h1-fix'):require_library_marker('/fake/lib.so')
    monkeypatch.setattr('vision_grasp_baseline.collection_h1.subprocess.run',
                        lambda *a,**k:SimpleNamespace(stdout='T openarm_closed_transit_startup_version\n'))
    require_library_marker('/fake/lib.so')


def test_old_replay_is_rejected_before_cycle_hardware_io(tmp_path,monkeypatch):
    import json
    from vision_grasp_baseline import cycle
    path=tmp_path/'legacy.json';path.write_text(json.dumps({'events':[]}))
    monkeypatch.setattr('sys.argv',['cycle','--continuous','--replay-plan',str(path),
                                  '--output',str(tmp_path/'run')])
    monkeypatch.setattr(cycle.subprocess,'run',lambda *a,**k:pytest.fail('Hardware subprocess called'))
    with pytest.raises(ValueError,match='Legacy'):cycle.main()
    assert not (tmp_path/'run').exists()


def test_installed_physical_description_renders_offline_without_launch(tmp_path,monkeypatch):
    import importlib.util
    import inspect
    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from scripts.control_ownership.secure_launch import VALIDATED_ARGUMENTS
    path=Path(get_package_share_directory('openarm_bimanual_moveit_config'))/'launch/demo.launch.py'
    spec=importlib.util.spec_from_file_location('collection_offline_description',path)
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    # The upstream generator writes a gravity URDF. Isolate it so the test
    # cannot overwrite the robot's real /tmp model, even if ROS is running.
    monkeypatch.setattr(demo,'tempfile',SimpleNamespace(gettempdir=lambda:str(tmp_path)))
    context=LaunchContext()
    for entity in demo.generate_launch_description().entities:
        if isinstance(entity,DeclareLaunchArgument):entity.execute(context)
    context.launch_configurations.update(VALIDATED_ARGUMENTS)
    parameters={name:LaunchConfiguration(name) for name in inspect.signature(
        demo.generate_robot_description).parameters if name!='context'}
    xml=demo.generate_robot_description(context,**parameters)
    root=ET.fromstring(description_transform(xml))
    hardware=root.findall('ros2_control/hardware')
    assert len(hardware)==2
    right=next(h for h in hardware if h.findtext("param[@name='arm_prefix']")=='right_')
    assert right.findtext("param[@name='startup_gripper_closed']")=='true'
    assert all(h.findtext('plugin')=='openarm_hardware/OpenArmHW' for h in hardware)
    assert (tmp_path/'openarm_bimanual.urdf').is_file()
