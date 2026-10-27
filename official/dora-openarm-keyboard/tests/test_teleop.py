# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dora_openarm_keyboard.keymap import (
    GRIP_KEYS,
    LEFT,
    MOTION_KEYS,
    PITCH,
    RIGHT,
    ROLL,
    X,
    Y,
    YAW,
    Z,
)
from dora_openarm_keyboard.main import KeyboardTeleop
from dora_openarm_keyboard.teleop import TeleopState


def make_state(**kwargs) -> TeleopState:
    defaults = {
        "home_right": np.zeros(3),
        "home_left": np.zeros(3),
        "home_rotation": Rotation.identity(),
        "linear_speed": 1.0,
        "angular_speed": 1.0,
        "grip_speed": 1.0,
        "pos_min": np.full(3, -10.0),
        "pos_max": np.full(3, 10.0),
    }
    return TeleopState(**{**defaults, **kwargs})


def test_motion_keys_are_per_arm():
    assert MOTION_KEYS["w"] == (LEFT, X, PITCH, +1)
    assert MOTION_KEYS["s"] == (LEFT, X, PITCH, -1)
    assert MOTION_KEYS["a"] == (LEFT, Y, ROLL, +1)
    assert MOTION_KEYS["f"] == (LEFT, Z, YAW, -1)
    assert MOTION_KEYS["i"] == (RIGHT, X, PITCH, +1)
    assert MOTION_KEYS["j"] == (RIGHT, Y, ROLL, +1)
    assert MOTION_KEYS["y"] == (RIGHT, Z, YAW, +1)
    assert MOTION_KEYS["h"] == (RIGHT, Z, YAW, -1)


def test_grip_keys_are_per_arm():
    assert GRIP_KEYS == {
        "c": (LEFT, -1),
        "x": (LEFT, +1),
        "n": (RIGHT, -1),
        "m": (RIGHT, +1),
    }


def test_each_arm_moves_only_on_its_own_keys():
    state = make_state()

    state.step(1.0, {"w"})

    np.testing.assert_allclose(state.arms[LEFT].pos, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(state.arms[RIGHT].pos, np.zeros(3))


def test_both_arms_move_at_the_same_time():
    state = make_state()

    state.step(1.0, {"w", "j", "y"})

    np.testing.assert_allclose(state.arms[LEFT].pos, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(state.arms[RIGHT].pos, [0.0, 1.0, 1.0])


def test_opposite_keys_cancel():
    state = make_state()

    state.step(1.0, {"w", "s"})

    np.testing.assert_allclose(state.arms[LEFT].pos, np.zeros(3))


def test_shift_switches_the_motion_keys_to_rotation():
    state = make_state()

    state.step(1.0, {"w", "shift"})

    np.testing.assert_allclose(state.arms[LEFT].pos, np.zeros(3))
    np.testing.assert_allclose(state.arms[LEFT].rot.as_rotvec(), [0.0, 1.0, 0.0])


def test_releasing_shift_returns_to_translation():
    state = make_state()

    state.step(1.0, {"a", "shift"})
    np.testing.assert_allclose(state.arms[LEFT].rot.as_rotvec(), [1.0, 0.0, 0.0])

    state.step(1.0, {"a"})
    np.testing.assert_allclose(state.arms[LEFT].pos, [0.0, 1.0, 0.0])


def test_gripper_keys_close_and_open_their_own_arm():
    state = make_state()

    state.step(1.0, {"x", "m"})
    assert state.arms[LEFT].grip == 1.0
    assert state.arms[RIGHT].grip == 1.0

    state.step(0.5, {"c"})
    assert state.arms[LEFT].grip == 0.5
    assert state.arms[RIGHT].grip == 1.0


def test_disable_stops_motion():
    state = make_state()
    state.enabled = False

    state.step(1.0, {"w", "i"})

    np.testing.assert_allclose(state.arms[RIGHT].pos, np.zeros(3))
    np.testing.assert_allclose(state.arms[LEFT].pos, np.zeros(3))


def test_escape_requests_quit_and_stops_the_arms():
    teleop = KeyboardTeleop(make_state())

    teleop.enqueue("press", "w")
    teleop.enqueue("press", "escape")
    teleop.step(1.0)
    assert teleop.quit_requested
    assert not teleop.state.enabled
    np.testing.assert_allclose(teleop.state.arms[LEFT].pos, np.zeros(3))

    # Esc quits rather than toggles: another press must not re-enable teleop.
    teleop.enqueue("release", "escape")
    teleop.enqueue("press", "escape")
    teleop.enqueue("press", "w")
    teleop.step(1.0)
    assert teleop.quit_requested
    assert not teleop.state.enabled
    np.testing.assert_allclose(teleop.state.arms[LEFT].pos, np.zeros(3))


def test_home_return_walks_the_target_back_at_the_teleop_speed():
    state = make_state(home_left=np.array([3.0, 0.0, 0.0]))

    state.step(1.0, {"s"})  # left arm -X, one metre away from home
    np.testing.assert_allclose(state.arms[LEFT].pos, [2.0, 0.0, 0.0])

    state.start_home()
    state.step(0.5, set())
    np.testing.assert_allclose(state.arms[LEFT].pos, [2.5, 0.0, 0.0])
    assert state.homing

    state.step(0.5, set())
    np.testing.assert_allclose(state.arms[LEFT].pos, [3.0, 0.0, 0.0])
    assert not state.homing


def test_home_return_unwinds_the_orientation_too():
    state = make_state()

    state.step(1.0, {"a", "shift"})  # left arm +Roll, one radian from home
    state.start_home()

    state.step(0.5, set())
    assert np.linalg.norm(state.arms[LEFT].rot.as_rotvec()) == pytest.approx(0.5)
    assert state.homing

    state.step(0.5, set())
    np.testing.assert_allclose(state.arms[LEFT].rot.as_rotvec(), np.zeros(3))
    assert not state.homing


def test_home_return_keeps_the_grippers_where_they_are():
    state = make_state(home_left=np.array([1.0, 0.0, 0.0]))

    state.step(1.0, {"x"})  # left gripper closes
    state.start_home()
    for _ in range(10):
        state.step(0.5, set())

    np.testing.assert_allclose(state.arms[LEFT].pos, [1.0, 0.0, 0.0])
    assert state.arms[LEFT].grip == 1.0


def test_a_motion_key_cancels_the_home_return():
    teleop = KeyboardTeleop(make_state(home_left=np.array([5.0, 0.0, 0.0])))
    state = teleop.state

    teleop.enqueue("press", "s")  # left arm -X, away from home
    teleop.step(1.0)
    np.testing.assert_allclose(state.arms[LEFT].pos, [4.0, 0.0, 0.0])

    # Pressing 0 releases the held key, so only the home return drives the arm.
    teleop.enqueue("press", "0")
    teleop.step(0.5)
    assert state.homing
    np.testing.assert_allclose(state.arms[LEFT].pos, [4.5, 0.0, 0.0])

    teleop.enqueue("press", "s")
    teleop.step(1.0)
    assert not state.homing
    np.testing.assert_allclose(state.arms[LEFT].pos, [3.5, 0.0, 0.0])


def test_escape_stops_a_home_return_where_the_arms_are():
    teleop = KeyboardTeleop(make_state(home_left=np.array([5.0, 0.0, 0.0])))
    state = teleop.state
    state.arms[LEFT].pos = np.zeros(3)

    teleop.enqueue("press", "0")
    teleop.step(1.0)
    assert state.homing
    np.testing.assert_allclose(state.arms[LEFT].pos, [1.0, 0.0, 0.0])

    teleop.enqueue("press", "escape")
    teleop.step(1.0)
    assert teleop.quit_requested
    assert not state.homing
    np.testing.assert_allclose(state.arms[LEFT].pos, [1.0, 0.0, 0.0])


def test_home_key_does_nothing_after_escape_quits():
    teleop = KeyboardTeleop(make_state(home_left=np.array([1.0, 2.0, 3.0])))
    state = teleop.state
    state.arms[LEFT].pos = np.zeros(3)

    teleop.enqueue("press", "escape")
    teleop.enqueue("press", "0")
    teleop.step(1.0)

    assert not state.homing
    np.testing.assert_allclose(state.arms[LEFT].pos, np.zeros(3))


def test_disable_stops_a_lifter_and_quits_the_tick_nodes():
    teleop = KeyboardTeleop(make_state())

    assert teleop.take_command() is None

    teleop.disable()
    # The lifter stop goes first; the quit then lets the quittable tick nodes
    # exit so the whole dataflow can come down.
    assert teleop.take_command() == "lifter-stop"
    assert teleop.take_command() == "quit"
    assert teleop.take_command() is None


def test_episode_reset_snaps_targets_and_opens_grippers():
    state = make_state(
        home_right=np.array([1.0, 2.0, 3.0]),
        home_left=np.array([-1.0, -2.0, -3.0]),
    )
    state.arms[RIGHT].pos[:] = 7.0
    state.arms[LEFT].pos[:] = 8.0
    state.arms[RIGHT].grip = 1.0
    state.arms[LEFT].grip = 1.0
    state.homing = True

    state.reset_home()

    np.testing.assert_allclose(state.arms[RIGHT].pos, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.arms[LEFT].pos, [-1.0, -2.0, -3.0])
    assert state.arms[RIGHT].grip == 0.0
    assert state.arms[LEFT].grip == 0.0
    assert not state.homing
