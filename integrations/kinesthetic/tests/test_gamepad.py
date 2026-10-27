import struct

import numpy as np
import pytest

from openarm_kinesthetic.config import load_yaml
from openarm_kinesthetic.teaching.gamepad import (
    BTN_TL,
    GamepadMapper,
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    LinuxJoystick,
    decode_js_events,
    normalize_device_name,
)


def neutral_state():
    axes = np.zeros(8, dtype=np.int16)
    axes[4:6] = -32767
    return axes, np.zeros(15, dtype=np.uint8)


def test_device_name_normalization_ignores_cosmetic_spacing_and_punctuation():
    configured = normalize_device_name("Gamepad MACHENIKE G3V2")
    assert normalize_device_name("Gamepad MACHENIKEG3V2") == configured
    assert normalize_device_name("gamepad-machenike_g3v2") == configured


def test_linux_joystick_event_decoder_preserves_device_and_host_clocks():
    raw = struct.pack("<IhBB", 12, -123, JS_EVENT_AXIS | JS_EVENT_INIT, 4)
    raw += struct.pack("<IhBB", 13, 1, JS_EVENT_BUTTON, 0)
    events = decode_js_events(raw, host_monotonic_s=20.0)
    assert events[0].kind == "axis" and events[0].initial
    assert events[0].device_time_ms == 12 and events[0].value == -123 and events[0].index == 4
    assert events[1].kind == "button" and not events[1].initial
    assert events[1].host_monotonic_s > events[0].host_monotonic_s


def test_linux_button_map_resolves_physical_left_bumper():
    joystick = LinuxJoystick.__new__(LinuxJoystick)
    joystick.button_codes = [0x130, 0x131, 0x13A, 0x133, 0x134, 0x137, BTN_TL]
    assert joystick.button_index_for_code(BTN_TL) == 6
    assert joystick.button_index_for_code(0xFFFF) is None


def test_g3v2_deadman_translation_rotation_and_gripper_edges():
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    assert mapper.axis["lt"] == 5
    assert mapper.axis["rt"] == 4
    assert mapper.button["gripper_open_a"] == 0
    assert mapper.button["record_toggle_x"] == 3
    assert mapper.button["hand_drag_lb"] == 6
    assert mapper.button["hand_drag_gripper_toggle_l3"] == 13
    assert mapper.button["gripper_close_y"] == 4
    assert mapper.translation_speed == 0.04
    assert mapper.vertical_speed == 0.03
    assert mapper.angular_speed == 0.25
    axes, buttons = neutral_state()
    command = mapper.update(axes, buttons)
    assert not command.active and not command.stop_latched

    axes[5] = 32767  # LT held
    axes[0] = 32767  # left stick right -> -world Y
    axes[1] = -32767  # left stick up -> +world X
    axes[2] = 32767  # right stick right -> +local yaw
    axes[3] = -32767  # right stick up -> +local pitch
    axes[6] = 32767  # D-pad right -> +local roll
    axes[7] = -32767  # D-pad up -> +world Z
    buttons[0] = 1
    command = mapper.update(axes, buttons)
    assert command.active and command.deadman_pressed
    assert not command.drag_modifier_pressed
    assert command.linear_velocity_world_m_s[0] > 0 and command.linear_velocity_world_m_s[1] < 0
    assert command.linear_velocity_world_m_s[2] > 0
    assert np.all(command.angular_velocity_local_rad_s > 0)
    assert command.open_pressed and not command.close_pressed
    assert not mapper.update(axes, buttons).open_pressed
    buttons[0] = 0; buttons[4] = 1
    command = mapper.update(axes, buttons)
    assert command.open_released and command.close_pressed
    buttons[6] = 1
    command = mapper.update(axes, buttons)
    assert command.drag_modifier_pressed
    assert not command.waypoint_capture_pressed
    buttons[13] = 1
    assert mapper.update(axes, buttons).hand_drag_gripper_toggle_pressed
    assert not mapper.update(axes, buttons).hand_drag_gripper_toggle_pressed
    buttons[3] = 1
    assert mapper.update(axes, buttons).record_toggle_pressed
    assert not mapper.update(axes, buttons).record_toggle_pressed


def test_standalone_lb_is_keyframe_capture_but_lt_plus_lb_is_not():
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state()
    buttons[6] = 1
    assert mapper.update(axes, buttons).waypoint_capture_pressed
    assert not mapper.update(axes, buttons).waypoint_capture_pressed

    buttons[6] = 0
    mapper.update(axes, buttons)
    axes[5] = 32767
    buttons[6] = 1
    assert not mapper.update(axes, buttons).waypoint_capture_pressed


def test_g3v2_stop_requires_lt_release_before_restart():
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state(); axes[5] = 32767
    assert mapper.update(axes, buttons).active
    buttons[5] = 1  # RB / R1
    assert mapper.update(axes, buttons).stop_latched
    buttons[5] = 0
    assert not mapper.update(axes, buttons).active
    axes[5] = -32767
    assert mapper.update(axes, buttons).stop_latched
    axes[5] = 32767
    command = mapper.update(axes, buttons)
    assert command.active and not command.stop_latched

    axes[4] = 32767  # RT is also stop
    assert mapper.update(axes, buttons).stop_latched


def test_gamepad_disconnect_zeroes_motion_and_latches_stop():
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state(); axes[5] = 32767; axes[0] = 32767
    assert mapper.update(axes, buttons).active
    command = mapper.update(axes, buttons, connected=False)
    assert command.stop_latched and not command.active
    assert not command.linear_velocity_world_m_s.any()
    assert not command.angular_velocity_local_rad_s.any()


def test_kernel_axis_identity_overrides_shuffled_numeric_indices():
    config = load_yaml("gamepad_control.yaml")
    mapper = GamepadMapper(config)
    # Deliberately shuffled js indices. Semantic mapping must come from ABS
    # identities, not guesses about axes[0]/axes[1].
    axis_codes = [0x10, 0x02, 0x0A, 0x00, 0x11, 0x05, 0x09, 0x01]
    assert mapper.configure_axis_codes(axis_codes)
    assert mapper.axis == {
        "left_x": 3,
        "left_y": 7,
        "right_x": 1,
        "right_y": 5,
        "lt": 2,
        "rt": 6,
        "dpad_x": 0,
        "dpad_y": 4,
    }


@pytest.mark.parametrize(
    ("axis", "raw", "linear_index", "linear_sign", "angular_index", "angular_sign"),
    [
        ("left_y", -32767, 0, 1, None, 0),
        ("left_y", 32767, 0, -1, None, 0),
        ("left_x", -32767, 1, 1, None, 0),
        ("left_x", 32767, 1, -1, None, 0),
        ("right_y", -32767, None, 0, 1, 1),
        ("right_y", 32767, None, 0, 1, -1),
        ("right_x", -32767, None, 0, 2, -1),
        ("right_x", 32767, None, 0, 2, 1),
        ("dpad_y", -32767, 2, 1, None, 0),
        ("dpad_y", 32767, 2, -1, None, 0),
        ("dpad_x", -32767, None, 0, 0, -1),
        ("dpad_x", 32767, None, 0, 0, 1),
    ],
)
def test_cartesian_direction_table(
    axis, raw, linear_index, linear_sign, angular_index, angular_sign
):
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state()
    axes[mapper.axis["lt"]] = 32767
    axes[mapper.axis[axis]] = raw
    command = mapper.update(axes, buttons)
    linear = command.linear_velocity_world_m_s
    angular = command.angular_velocity_local_rad_s
    if linear_index is None:
        assert not linear.any()
    else:
        assert np.sign(linear[linear_index]) == linear_sign
        assert np.count_nonzero(linear) == 1
    if angular_index is None:
        assert not angular.any()
    else:
        assert np.sign(angular[angular_index]) == angular_sign
        assert np.count_nonzero(angular) == 1


def test_dpad_up_is_pure_positive_world_z_not_wrist_rotation():
    mapper = GamepadMapper(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state()
    axes[mapper.axis["lt"]] = 32767
    axes[mapper.axis["dpad_y"]] = -32767
    command = mapper.update(axes, buttons)
    assert command.linear_velocity_world_m_s[2] > 0
    assert np.allclose(command.linear_velocity_world_m_s[:2], 0.0)
    assert np.allclose(command.angular_velocity_local_rad_s, 0.0)


def test_simulation_and_real_paths_share_identical_normalized_command():
    config = load_yaml("gamepad_control.yaml")
    simulation_mapper = GamepadMapper(config)
    real_mapper = GamepadMapper(config)
    assert real_mapper.configure_axis_codes(
        [0x00, 0x01, 0x02, 0x05, 0x09, 0x0A, 0x10, 0x11]
    )
    axes, buttons = neutral_state()
    axes[5] = 32767
    axes[1] = 20000
    axes[7] = -32767
    sim = simulation_mapper.update(axes, buttons)
    real = real_mapper.update(axes, buttons)
    assert np.allclose(sim.linear_velocity_world_m_s, real.linear_velocity_world_m_s)
    assert np.allclose(sim.angular_velocity_local_rad_s, real.angular_velocity_local_rad_s)
