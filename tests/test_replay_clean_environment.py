"""CLI list-only checks; never launch ROS, read CAN, or open a joystick."""
import os
from pathlib import Path
import subprocess

SCRIPT = '/home/lushilin/openarm_kinesthetic_ws/scripts/replay.sh'


def test_list_works_from_polluted_act_environment(tmp_path):
    sentinel = tmp_path/'unexpected_ros_launch'
    fake_ros = tmp_path/'ros2'
    fake_ros.write_text('#!/bin/sh\ntouch "'+str(sentinel)+'"\nexit 99\n')
    fake_ros.chmod(0o755)
    env = dict(os.environ)
    env.pop('OPENARM_REPLAY_CLEAN_ENV', None)
    env.update(AMENT_PREFIX_PATH='/tmp/.control_ownership_overlay/install',
               LD_LIBRARY_PATH='/tmp/.control_ownership_overlay/install/lib',
               PYTHONHOME='/nonexistent/old-act-python', PYTHONPATH='/nonexistent/old-act',
               ROS_DOMAIN_ID='117', ROS_SECURITY_ENABLE='true',
               ROS_SECURITY_ENCLAVE_OVERRIDE='/openarm/backend',
               FASTDDS_DEFAULT_PROFILES_FILE='/nonexistent/old-dds.xml',
               PATH=str(tmp_path)+':/usr/bin:/bin')
    result = subprocess.run(['/bin/bash', SCRIPT, 'list'], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '20260928_151345_001.json' in result.stdout
    assert '20260929_115522_015.json' in result.stdout
    assert not sentinel.exists()
    assert env['ROS_DOMAIN_ID'] == '117'


def test_fresh_environment_loads_original_player_only():
    env = {'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
    command = '''source /home/lushilin/openarm_kinesthetic_ws/scripts/original_replay_env.sh &&
/usr/bin/python3 -c 'import os; import openarm_kinesthetic_recorder.recorded_motion_player as p; print(p.__file__); assert os.environ["ROS_DOMAIN_ID"] == "97"; assert ".control_ownership_overlay" not in os.environ.get("AMENT_PREFIX_PATH", "")'
'''
    result = subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c', command],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '/openarm_kinesthetic_recorder/recorded_motion_player.py' in result.stdout
