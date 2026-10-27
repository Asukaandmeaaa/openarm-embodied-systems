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

"""Key bindings for keyboard teleoperation.

Each arm has its own motion and gripper keys, so both can be driven at the
same time without selecting one first.  Shift is a momentary modifier: while
it is held, the motion keys drive their rotation axis instead of their
translation axis, and releasing it always returns to translation.
"""

RIGHT = "right"
LEFT = "left"

# Axis indices shared by the linear (x, y, z) and angular (roll, pitch, yaw)
# parts of a motion binding.
X = ROLL = 0
Y = PITCH = 1
Z = YAW = 2

# key -> (side, linear axis, angular axis, sign).  A held key advances its
# linear axis, or its angular axis while ROTATION_KEY is also held.
MOTION_KEYS: dict[str, tuple[str, int, int, int]] = {
    # left arm
    "w": (LEFT, X, PITCH, +1),
    "s": (LEFT, X, PITCH, -1),
    "a": (LEFT, Y, ROLL, +1),
    "d": (LEFT, Y, ROLL, -1),
    "r": (LEFT, Z, YAW, +1),
    "f": (LEFT, Z, YAW, -1),
    # right arm
    "i": (RIGHT, X, PITCH, +1),
    "k": (RIGHT, X, PITCH, -1),
    "j": (RIGHT, Y, ROLL, +1),
    "l": (RIGHT, Y, ROLL, -1),
    "y": (RIGHT, Z, YAW, +1),
    "h": (RIGHT, Z, YAW, -1),
}

# key -> (side, sign), where +1 closes the gripper and -1 opens it.
GRIP_KEYS: dict[str, tuple[str, int]] = {
    "c": (LEFT, -1),
    "x": (LEFT, +1),
    "n": (RIGHT, -1),
    "m": (RIGHT, +1),
}

# Held to reinterpret the motion keys as rotation.
ROTATION_KEY = "shift"
# Edge-triggered controls.
HOME_KEY = "0"
QUIT_KEY = "escape"


def drives_motion(key: str) -> bool:
    """Whether a key moves an arm or one of the grippers."""
    return key in MOTION_KEYS or key in GRIP_KEYS


# No key drives the shared lifter, but a dataflow with a physical one still
# needs it stopped when the node exits.
LIFTER_STOP_COMMAND = "lifter-stop"
# Makes every dora-openarm-quitter tick node with this node's command output
# wired in exit too, so the dataflow's timers stop and it can shut down.
QUIT_COMMAND = "quit"

HELP_TEXT = """\
Left arm
--------
  W / S      +/- X   (+/- Pitch with Shift)
  A / D      +/- Y   (+/- Roll  with Shift)
  R / F      +/- Z   (+/- Yaw   with Shift)

Right arm
---------
  I / K      +/- X   (+/- Pitch with Shift)
  J / L      +/- Y   (+/- Roll  with Shift)
  Y / H      +/- Z   (+/- Yaw   with Shift)

  Shift      hold to rotate instead of translate

Left gripper
------------
  C          Open
  X          Close

Right gripper
-------------
  N          Open
  M          Close

Control
-------
  0          Return both arms home; any motion key aborts
  Esc        Quit teleoperation\
"""
