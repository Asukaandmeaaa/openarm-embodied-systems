"""Run the existing ROS H2 entrypoint with connection-bound goal capabilities.

The parent ``client run --owner h2`` supplies OPENARM_AUTHORITY_TOKEN only to
this child. A deep copy is stamped at the ActionClient send boundary; the
original planned trajectory, timestamps, reports and hashes remain untouched.
The gateway restores the original frame before forwarding to a controller.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from copy import deepcopy
import os
import re
import secrets
import sys


def capability_frame(original_frame, token, nonce=None):
    """Encode a capability without logging it or discarding the original frame."""
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("H2 requires the connection-bound parent capability")
    if not isinstance(original_frame, str) or len(original_frame.encode("utf-8")) > 1024:
        raise ValueError("invalid original trajectory frame")
    if original_frame.startswith("openarm-lease:"):
        raise ValueError("trajectory already contains an authority capability")
    nonce = secrets.token_hex(16) if nonce is None else nonce
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise ValueError("invalid H2 request nonce")
    encoded = base64.urlsafe_b64encode(original_frame.encode("utf-8")).decode("ascii")
    return f"openarm-lease:{token}:{nonce}:{encoded}"


def decorated_goal(goal, token, execute_goal_type, follow_goal_type):
    """Copy exactly one supported goal and change only its joint frame metadata."""
    if isinstance(goal, execute_goal_type):
        copy = deepcopy(goal)
        trajectory = copy.trajectory.joint_trajectory
    elif isinstance(goal, follow_goal_type):
        copy = deepcopy(goal)
        trajectory = copy.trajectory
    else:
        raise ValueError("H2 attempted an unsupported action goal type")
    trajectory.header.frame_id = capability_frame(trajectory.header.frame_id, token)
    return copy


@contextmanager
def inject_action_capabilities(token, *, action_client_type=None,
                               execute_goal_type=None, follow_goal_type=None):
    """Patch only this wrapper process; installed H2 packages are unchanged."""
    capability_frame("", token)  # Validate before importing/starting ROS code.
    if action_client_type is None:
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        from control_msgs.action import FollowJointTrajectory
        action_client_type = ActionClient
        execute_goal_type = ExecuteTrajectory.Goal
        follow_goal_type = FollowJointTrajectory.Goal
    if execute_goal_type is None or follow_goal_type is None:
        raise ValueError("both allowed goal types are required")
    original = action_client_type.send_goal_async
    if getattr(original, "_openarm_capability_injection", False):
        raise ValueError("H2 capability injection already installed")

    def send(self, goal, *args, **kwargs):
        outbound = decorated_goal(goal, token, execute_goal_type, follow_goal_type)
        return original(self, outbound, *args, **kwargs)

    send._openarm_capability_injection = True
    action_client_type.send_goal_async = send
    try:
        yield
    finally:
        action_client_type.send_goal_async = original


def validated_token(require_token=True):
    # Do not pass this capability to any subprocess the original entrypoint may
    # create. The wrapper's closure retains it for this connection's lifetime.
    token = os.environ.pop("OPENARM_AUTHORITY_TOKEN", None)
    if require_token:
        capability_frame("", token)
    from .security import environment
    expected = environment(os.environ.get("ROS_SECURITY_KEYSTORE", ""),
                           os.environ.get("ROS_DOMAIN_ID"))
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError("H2 wrapper requires the reviewed enforced DDS environment")
    if os.environ.get("ROS_SECURITY_ENCLAVE_OVERRIDE"):
        raise ValueError("H2 wrapper forbids enclave override")
    if (sys.argv.count("--enclave") != 1
            or any(arg == "-e" or arg.startswith("--enclave=") for arg in sys.argv)
            or not any(sys.argv[i:i+2] == ["--enclave", "/openarm/h2"]
                       for i in range(1, len(sys.argv)))):
        raise ValueError("H2 wrapper requires explicit --ros-args --enclave /openarm/h2")
    return token


def main():
    with inject_action_capabilities(validated_token()):
        from openarm_kinesthetic_recorder.home_guarded_executor import main as original_main
        return original_main()


if __name__ == "__main__":
    raise SystemExit(main())
