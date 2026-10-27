import numpy as np

from openarm_kinesthetic.config import load_yaml
from openarm_kinesthetic.teaching.gamepad_overlay import GamepadOverlay


def neutral_state():
    axes = np.zeros(8, dtype=np.int16)
    axes[4:6] = -32767
    return axes, np.zeros(15, dtype=np.uint8)


def test_overlay_has_fixed_rgb_shape_and_lights_physical_controls():
    overlay = GamepadOverlay(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state()
    neutral = overlay.render(
        axes, buttons, connected=True, active=False, stop_latched=False, now_s=0.0
    )
    assert neutral.shape == (GamepadOverlay.HEIGHT, GamepadOverlay.WIDTH, 3)
    assert neutral.dtype == np.uint8

    axes[5] = 32767
    buttons[0] = 1
    active = overlay.render(
        axes, buttons, connected=True, active=True, stop_latched=False, now_s=1.0
    )
    assert tuple(active[47, 55]) == GamepadOverlay.ACTIVE  # LT, left of label
    assert tuple(active[153, 342]) == GamepadOverlay.ACTIVE  # A, below label
    assert np.count_nonzero(active != neutral) > 100


def test_overlay_uses_measured_y_button_four_and_retains_last_label_briefly():
    overlay = GamepadOverlay(load_yaml("gamepad_control.yaml"))
    axes, buttons = neutral_state()
    buttons[4] = 1
    pressed = overlay.render(
        axes, buttons, connected=True, active=False, stop_latched=False, now_s=2.0
    )
    assert tuple(pressed[95, 342]) == GamepadOverlay.YELLOW  # below label
    assert overlay.last_control == "Y / CLOSE"

    buttons[4] = 0
    overlay.render(axes, buttons, connected=True, active=False, stop_latched=False, now_s=2.5)
    assert overlay.last_control == "Y / CLOSE"
