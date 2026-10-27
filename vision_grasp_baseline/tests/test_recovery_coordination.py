import json
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest

from vision_grasp_baseline.cycle import wait_for_gateway_handoff
from vision_grasp_baseline.recovery_launch import description_transform,linear_planner_node,startup_joint4_model


def reply(state, *, fault=None):
    return SimpleNamespace(stdout=json.dumps(dict(
        status=dict(state=state, draining=state!='idle', fault=fault),
        pending=[], synchronized=True, hold_pending=[])))


def test_next_stage_waits_for_asynchronous_release():
    with patch('vision_grasp_baseline.cycle.subprocess.run',
               side_effect=[reply('draining'), reply('draining'), reply('idle')]) as query:
        with patch('vision_grasp_baseline.cycle.time.sleep') as sleep:
            wait_for_gateway_handoff()
    assert query.call_count == 3
    assert sleep.call_count == 2


def test_handoff_does_not_clear_gateway_fault():
    with patch('vision_grasp_baseline.cycle.subprocess.run', return_value=reply('fault', fault='motor')):
        with pytest.raises(RuntimeError, match='gateway fault'):
            wait_for_gateway_handoff()


def test_manual_synchronization_is_waited_for_not_bypassed():
    with patch('vision_grasp_baseline.cycle.subprocess.run',side_effect=[
        reply('fault',fault='manual_hold_synchronization'),reply('draining'),reply('idle')]) as query:
        with patch('vision_grasp_baseline.cycle.time.sleep'):
            result=wait_for_gateway_handoff()
    assert query.call_count==3
    assert result['status']['state']=='idle'


def test_stuck_synchronization_still_times_out():
    with patch('vision_grasp_baseline.cycle.subprocess.run',return_value=reply('fault',fault='manual_hold_synchronization')):
        with pytest.raises(RuntimeError,match='hold_pending'):
            wait_for_gateway_handoff(timeout=0)


def test_collection_initializes_idle_unsynchronized_gateway_once():
    initial=json.loads(reply('idle').stdout);initial['synchronized']=False
    with patch('vision_grasp_baseline.cycle.subprocess.run',side_effect=[
        SimpleNamespace(stdout=json.dumps(initial)),SimpleNamespace(returncode=0),
        reply('fault',fault='manual_hold_synchronization'),reply('idle')]) as query:
        with patch('vision_grasp_baseline.cycle.time.sleep'):
            wait_for_gateway_handoff(synchronize_if_needed=True)
    assert [call.args[0][1] for call in query.call_args_list]==['status','synchronize-hold','status','status']


def test_automatic_synchronization_never_overrides_active_trajectory():
    initial=json.loads(reply('idle').stdout);initial['synchronized']=False
    initial['raw_active']={'right_joint_trajectory_controller':['active_goal']}
    with patch('vision_grasp_baseline.cycle.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(initial))) as query:
        with pytest.raises(RuntimeError,match='raw_active'):
            wait_for_gateway_handoff(timeout=0,synchronize_if_needed=True)
    assert query.call_count==1


def test_recovery_changes_only_expected_physical_plugins():
    control='<ros2_control><hardware><plugin>openarm_hardware/OpenArmHW</plugin></hardware></ros2_control>'
    result=ET.fromstring(description_transform('<robot>'+control*2+'</robot>'))
    assert [p.text for p in result.findall("ros2_control/hardware/param[@name='startup_hold_current']")] == ['true','true']
    with pytest.raises(ValueError, match='Unexpected physical hardware plugin'):
        description_transform(('<robot>'+control*2+'</robot>').replace('openarm_hardware/OpenArmHW','mock_components/GenericSystem'))


def test_lin_pipeline_is_only_added_to_independent_move_group():
    backend=dict(package='controller_manager',parameters=[{'unchanged':True}])
    assert linear_planner_node(backend)==backend
    moveit=linear_planner_node(dict(package='moveit_ros_move_group',parameters=[{'original':1}]))
    assert moveit['parameters'][0]=={'original':1}
    assert 'pilz_industrial_motion_planner' in moveit['parameters'][-1]['planning_pipelines']


def test_explicit_startup_bound_changes_only_right_elbow_lower():
    xml='<robot><joint name="openarm_right_joint4"><limit lower="-.02" upper="2.4435"/></joint><joint name="other"><limit lower="-1" upper="1"/></joint></robot>'
    assert startup_joint4_model(xml,None)==xml
    root=ET.fromstring(startup_joint4_model(xml,-.08))
    assert root.find("joint[@name='openarm_right_joint4']/limit").attrib=={'lower':'-0.08','upper':'2.4435'}
    assert root.find("joint[@name='other']/limit").get('lower')=='-1'
    for value in [-.081,0,float('nan')]:
        with pytest.raises(ValueError):startup_joint4_model(xml,value)
    moveit=linear_planner_node(dict(package='moveit_ros_move_group',parameters=[{'robot_description':xml}]),joint4_lower=-.08)
    assert moveit['parameters'][0]['robot_description']==startup_joint4_model(xml,-.08)
