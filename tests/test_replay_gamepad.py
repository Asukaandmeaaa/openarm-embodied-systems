"""Both USB input profiles, simulated events only; no physical joystick or ROS."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from copy import deepcopy

import numpy as np
import pytest

ROOT = Path('/home/lushilin/openarm_kinesthetic_ws')
sys.path.insert(0, str(ROOT/'src'))
from openarm_kinesthetic.teaching.gamepad import load_yaml
spec = importlib.util.spec_from_file_location('replay_gamepad_test', ROOT/'scripts/button_play_recorded_motion.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def pad_for(profile):
    xinput = profile == 'xinput'
    codes = [304, 305, 307, 308, 310, 311, 314, 315, 316, 317, 318] if xinput else [304,305,306,307,308,311,310]
    return NS(name='Generic X-Box pad' if xinput else 'Gamepad MACHENIKE G3V2',
              axis_codes=[0,1,2,3,4,5,16,17] if xinput else [0,1,2,5,9,10,16,17],
              button_index_for_code=lambda code: codes.index(code) if code in codes else None)


@pytest.mark.parametrize('profile', ['original','xinput'])
@pytest.mark.parametrize('control', ['idle','lt','rt','lb','rb','b','right_stick'])
def test_start_stop_and_interlock_decoding(profile, control):
    cfg = load_yaml('gamepad_control.yaml'); before = deepcopy(cfg)
    mapper = replay.configure_replay_gamepad(pad_for(profile), cfg, xinput=profile=='xinput')
    assert cfg == before
    axes = np.zeros(8, dtype=np.int16); buttons = np.zeros(16, dtype=np.uint8)
    lt,rt = (2,5) if profile=='xinput' else (5,4)
    axes[lt] = axes[rt] = -32767
    if control=='lt': axes[lt]=32767
    if control=='rt': axes[rt]=32767
    if control=='lb': buttons[4 if profile=='xinput' else 6]=1
    if control=='rb': buttons[5]=1
    if control=='b': buttons[1]=1
    if control=='right_stick': axes[4 if profile=='xinput' else 3]=32767
    command = mapper.update(axes,buttons,connected=True)
    assert command.stop_latched == (control in ('rt','rb'))
    assert command.deadman_pressed == (control=='lt')
    assert command.drag_modifier_pressed == (control=='lb')
    assert mapper.button['episode_capture_b']==1


@pytest.mark.parametrize('fault',['name','axis','button'])
def test_unknown_or_incomplete_alternate_layout_rejected(fault):
    pad=pad_for('xinput')
    if fault=='name': pad.name='unidentified controller'
    if fault=='axis': pad.axis_codes=[0,1]
    if fault=='button': pad.button_index_for_code=lambda _:None
    with pytest.raises(ValueError):
        replay.configure_replay_gamepad(pad,load_yaml('gamepad_control.yaml'),xinput=True)
