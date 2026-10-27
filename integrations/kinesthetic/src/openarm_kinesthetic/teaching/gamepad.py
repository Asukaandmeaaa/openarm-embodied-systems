"""Dependency-free Linux joystick input for simulation-only 6D teaching."""

from __future__ import annotations

import argparse
import array
import fcntl
import glob
import os
import struct
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_yaml


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_SIZE = 8
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGAXMAP = 0x80406A32
BTN_TL = 0x136
KEY_MAX = 0x2FF
BTN_MISC = 0x100

ABS_CODE_NAMES = {
    0x00: "ABS_X",
    0x01: "ABS_Y",
    0x02: "ABS_Z",
    0x03: "ABS_RX",
    0x04: "ABS_RY",
    0x05: "ABS_RZ",
    0x09: "ABS_GAS",
    0x0A: "ABS_BRAKE",
    0x10: "ABS_HAT0X",
    0x11: "ABS_HAT0Y",
}
ABS_NAME_CODES = {name: code for code, name in ABS_CODE_NAMES.items()}


def normalize_axis_value(raw_value: int | float) -> float:
    return float(np.clip(float(raw_value) / 32767.0, -1.0, 1.0))


def _jsiocgname(length: int) -> int:
    return 0x80006A13 | (int(length) << 16)


def _jsiocgbtnmap(length: int) -> int:
    return 0x80006A34 | (int(length) << 16)


def normalize_device_name(name: str) -> str:
    """Normalize cosmetic USB-driver name differences for auto detection."""
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", str(name))
        if character.isalnum()
    )


@dataclass(frozen=True)
class JoystickEvent:
    device_time_ms: int
    host_monotonic_s: float
    kind: str
    index: int
    value: int
    initial: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_time_ms": self.device_time_ms,
            "host_monotonic_s": self.host_monotonic_s,
            "kind": self.kind,
            "index": self.index,
            "value": self.value,
            "initial": self.initial,
        }


def decode_js_events(raw: bytes, *, host_monotonic_s: float | None = None) -> list[JoystickEvent]:
    if len(raw) % JS_EVENT_SIZE:
        raise ValueError("Linux joystick event buffer is not 8-byte aligned")
    base = time.monotonic() if host_monotonic_s is None else float(host_monotonic_s)
    events = []
    for offset in range(0, len(raw), JS_EVENT_SIZE):
        device_ms, value, event_type, index = struct.unpack_from("<IhBB", raw, offset)
        initial = bool(event_type & JS_EVENT_INIT)
        event_type &= ~JS_EVENT_INIT
        kind = "button" if event_type == JS_EVENT_BUTTON else "axis" if event_type == JS_EVENT_AXIS else "unknown"
        events.append(JoystickEvent(int(device_ms), base + (offset // JS_EVENT_SIZE) * 1e-6, kind, int(index), int(value), initial))
    return events


class LinuxJoystick:
    """Nonblocking reader for the stable /dev/input/js* userspace ABI."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        axes = array.array("B", [0]); buttons = array.array("B", [0]); name = array.array("B", [0] * 128)
        fcntl.ioctl(self.fd, JSIOCGAXES, axes, True)
        fcntl.ioctl(self.fd, JSIOCGBUTTONS, buttons, True)
        fcntl.ioctl(self.fd, _jsiocgname(len(name)), name, True)
        self.name = name.tobytes().split(b"\0", 1)[0].decode(errors="replace")
        self.axes = np.zeros(int(axes[0]), dtype=np.int16)
        self.buttons = np.zeros(int(buttons[0]), dtype=np.uint8)
        axis_map = array.array("B", [0] * 64)
        try:
            fcntl.ioctl(self.fd, JSIOCGAXMAP, axis_map, True)
            self.axis_codes = [int(code) for code in axis_map[: int(axes[0])]]
        except OSError:
            self.axis_codes = []
        button_map = array.array("H", [0] * (KEY_MAX - BTN_MISC + 1))
        try:
            fcntl.ioctl(
                self.fd,
                _jsiocgbtnmap(button_map.itemsize * len(button_map)),
                button_map,
                True,
            )
            self.button_codes = [int(code) for code in button_map[: int(buttons[0])]]
        except OSError:
            # Some older joystick drivers do not expose JSIOCGBTNMAP.  The
            # reviewed YAML indices remain the deterministic fallback.
            self.button_codes = []
        self.connected = True

    def button_index_for_code(self, code: int) -> int | None:
        try:
            return self.button_codes.index(int(code))
        except ValueError:
            return None

    def axis_code_for_index(self, index: int) -> tuple[int | None, str]:
        if index < 0 or index >= len(self.axis_codes):
            return None, "UNKNOWN"
        code = self.axis_codes[index]
        return code, ABS_CODE_NAMES.get(code, f"ABS_0x{code:02x}")

    @classmethod
    def open_configured(cls, requested: str = "auto", config: dict[str, Any] | None = None) -> "LinuxJoystick | None":
        if requested.lower() == "off":
            return None
        cfg = config or load_yaml("gamepad_control.yaml")
        candidates: list[str]
        if requested.lower() == "auto":
            candidates = sorted(glob.glob(str(cfg["device"]["auto_glob"])))
            fallback = str(cfg["device"].get("linux_js_fallback", ""))
            if not candidates and fallback:
                candidates = [fallback]
        else:
            candidates = [requested]
        for candidate in candidates:
            try:
                joystick = cls(candidate)
            except (FileNotFoundError, PermissionError, OSError):
                if requested.lower() != "auto":
                    raise
                continue
            expected = str(cfg["device"].get("expected_name", ""))
            if (
                requested.lower() == "auto"
                and expected
                and normalize_device_name(joystick.name) != normalize_device_name(expected)
            ):
                joystick.close()
                continue
            joystick.poll()  # consume kernel-generated initial state events
            return joystick
        if requested.lower() == "auto":
            return None
        raise FileNotFoundError(f"gamepad not found: {requested}")

    def poll(self) -> list[JoystickEvent]:
        if not self.connected or self.fd < 0:
            return []
        events: list[JoystickEvent] = []
        while True:
            try:
                raw = os.read(self.fd, JS_EVENT_SIZE * 64)
            except BlockingIOError:
                break
            except OSError:
                self.connected = False
                break
            if not raw:
                self.connected = False
                break
            batch = decode_js_events(raw)
            for event in batch:
                if event.kind == "axis" and event.index < len(self.axes):
                    self.axes[event.index] = event.value
                elif event.kind == "button" and event.index < len(self.buttons):
                    self.buttons[event.index] = event.value
            events.extend(batch)
        return events

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        self.connected = False

    def __enter__(self) -> "LinuxJoystick":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@dataclass(frozen=True)
class GamepadCommand:
    active: bool
    stop_latched: bool
    deadman_pressed: bool
    drag_modifier_pressed: bool
    hand_drag_gripper_toggle_pressed: bool
    record_toggle_pressed: bool
    waypoint_capture_pressed: bool
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_local_rad_s: np.ndarray
    open_pressed: bool
    open_released: bool
    close_pressed: bool
    close_released: bool


class GamepadMapper:
    def __init__(self, config: dict[str, Any]):
        mapping = config["mapping"]; control = config["control"]
        self.axis = {name: int(index) for name, index in mapping["axes"].items()}
        self.configured_axis_codes = {
            str(name): str(code) for name, code in mapping.get("axis_codes", {}).items()
        }
        self.button = {name: int(index) for name, index in mapping["buttons"].items()}
        self.trigger_threshold = float(control["trigger_press_threshold"])
        self.deadzone = float(control["stick_deadzone"])
        self.exponent = float(control["response_exponent"])
        self.translation_speed = float(control["maximum_translation_velocity_m_s"])
        self.vertical_speed = float(control["maximum_vertical_velocity_m_s"])
        self.angular_speed = float(control["maximum_angular_velocity_rad_s"])
        self.maximum_target_error = float(control["maximum_target_position_error_m"])
        self.stop_latched = False
        self.restart_ready = True
        self.previous_buttons: dict[str, bool] = {name: False for name in self.button}

    def configure_axis_codes(self, axis_codes: list[int]) -> bool:
        """Resolve semantic axes from the kernel map, preserving YAML fallback."""
        configured = self.configured_axis_codes
        if not configured or not axis_codes:
            return False
        resolved: dict[str, int] = {}
        for semantic, code_name in configured.items():
            code = ABS_NAME_CODES.get(code_name)
            if code is None or code not in axis_codes:
                return False
            resolved[semantic] = axis_codes.index(code)
        self.axis.update(resolved)
        return True

    @staticmethod
    def _axis_value(axes: np.ndarray, index: int) -> float:
        return normalize_axis_value(axes[index]) if index < len(axes) else 0.0

    def _stick(self, axes: np.ndarray, name: str) -> float:
        value = self._axis_value(axes, self.axis[name])
        magnitude = abs(value)
        if magnitude <= self.deadzone:
            return 0.0
        scaled = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        return float(np.sign(value) * scaled**self.exponent)

    def _trigger(self, axes: np.ndarray, name: str) -> float:
        return 0.5 * (self._axis_value(axes, self.axis[name]) + 1.0)

    def _button(self, buttons: np.ndarray, name: str) -> bool:
        index = self.button[name]
        return bool(buttons[index]) if index < len(buttons) else False

    def update(self, axes: np.ndarray, buttons: np.ndarray, *, connected: bool = True) -> GamepadCommand:
        lt_pressed = connected and self._trigger(axes, "lt") >= self.trigger_threshold
        rt_pressed = connected and self._trigger(axes, "rt") >= self.trigger_threshold
        rb_pressed = connected and self._button(buttons, "stop_rb_r1")
        if rt_pressed or rb_pressed or not connected:
            self.stop_latched = True
        if not lt_pressed:
            self.restart_ready = True
        elif self.stop_latched and self.restart_ready and not (rt_pressed or rb_pressed):
            self.stop_latched = False
            self.restart_ready = False
        elif lt_pressed:
            self.restart_ready = False
        active = bool(connected and lt_pressed and not self.stop_latched)

        lx = self._stick(axes, "left_x"); ly = self._stick(axes, "left_y")
        rx = self._stick(axes, "right_x"); ry = self._stick(axes, "right_y")
        dx = self._axis_value(axes, self.axis["dpad_x"])
        dy = self._axis_value(axes, self.axis["dpad_y"])
        linear = np.asarray([-ly * self.translation_speed, -lx * self.translation_speed, -dy * self.vertical_speed])
        angular = np.asarray([dx * self.angular_speed, -ry * self.angular_speed, rx * self.angular_speed])
        if not active:
            linear[:] = 0.0; angular[:] = 0.0

        current = {name: connected and self._button(buttons, name) for name in self.button}
        result = GamepadCommand(
            active=active,
            stop_latched=self.stop_latched,
            deadman_pressed=lt_pressed,
            drag_modifier_pressed=current["hand_drag_lb"],
            hand_drag_gripper_toggle_pressed=(
                current["hand_drag_gripper_toggle_l3"]
                and not self.previous_buttons["hand_drag_gripper_toggle_l3"]
            ),
            record_toggle_pressed=(
                current["record_toggle_x"]
                and not self.previous_buttons["record_toggle_x"]
            ),
            # LB is already the physical-drag modifier while LT is held.  A
            # standalone LB edge (LT released) is therefore the only reviewed
            # button combination available for keyframe capture without
            # inventing an unverified G3V2 button index.
            waypoint_capture_pressed=(
                current["hand_drag_lb"]
                and not self.previous_buttons["hand_drag_lb"]
                and not lt_pressed
            ),
            linear_velocity_world_m_s=linear,
            angular_velocity_local_rad_s=angular,
            open_pressed=current["gripper_open_a"] and not self.previous_buttons["gripper_open_a"],
            open_released=not current["gripper_open_a"] and self.previous_buttons["gripper_open_a"],
            close_pressed=current["gripper_close_y"] and not self.previous_buttons["gripper_close_y"],
            close_released=not current["gripper_close_y"] and self.previous_buttons["gripper_close_y"],
        )
        self.previous_buttons = current
        return result


def main() -> None:  # pragma: no cover - interactive device diagnostic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    cfg = load_yaml("gamepad_control.yaml")
    joystick = LinuxJoystick.open_configured(args.device, cfg)
    if joystick is None:
        raise SystemExit("G3V2 gamepad not found")
    mapper = GamepadMapper(cfg)
    print(f"device={joystick.path} name={joystick.name!r} axes={len(joystick.axes)} buttons={len(joystick.buttons)}")
    print("Move each stick, press LT/RT, D-pad, A, Y, and RB/R1; raw changes follow.")
    print("Motion requires holding LT while moving a stick; pressing and releasing LT is not a toggle.")
    deadline = time.monotonic() + max(0.1, args.seconds)
    observed_axes: set[int] = set()
    observed_buttons: set[int] = set()
    deadman_seen = False
    try:
        while time.monotonic() < deadline and joystick.connected:
            for event in joystick.poll():
                if not event.initial:
                    if event.kind == "axis":
                        code, code_name = joystick.axis_code_for_index(event.index)
                        normalized = normalize_axis_value(event.value)
                        print(
                            f"axis[{event.index}] code={code_name}({code}) "
                            f"raw={event.value} normalized={normalized:+.4f}"
                        )
                    else:
                        print(f"{event.kind}[{event.index}]={event.value}")
                    if event.kind == "axis":
                        observed_axes.add(event.index)
                    elif event.kind == "button":
                        observed_buttons.add(event.index)
            command = mapper.update(joystick.axes, joystick.buttons, connected=joystick.connected)
            deadman_seen = deadman_seen or command.deadman_pressed
            lt_raw = int(joystick.axes[mapper.axis["lt"]]) if mapper.axis["lt"] < len(joystick.axes) else None
            print(
                "\rLT_raw=%s deadman=%s active=%s stop=%s linear=%s angular=%s" % (
                    lt_raw,
                    command.deadman_pressed,
                    command.active,
                    command.stop_latched,
                    np.round(command.linear_velocity_world_m_s, 3).tolist(),
                    np.round(command.angular_velocity_local_rad_s, 3).tolist(),
                ),
                end="",
                flush=True,
            )
            time.sleep(0.02)
    finally:
        joystick.close()
        print()
        print(f"observed_axes={sorted(observed_axes)} observed_buttons={sorted(observed_buttons)}")
        if not deadman_seen:
            print(
                "WARNING: LT was never detected; hold LT and confirm "
                f"axis[{mapper.axis['lt']}] approaches +32767."
            )


if __name__ == "__main__":
    main()
