"""Bounded Damiao parameter reads; never enable, disable, move or write motors.

Uses SocketCAN directly to distinguish transport failures from driver parsing.
Requires an already configured CAN-FD interface; does not reconfigure it.
Protocol: enactic/openarm_can CanPacketEncoder::pack_query_param_data.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

# The probe sends parameter-read requests, so it participates in the same
# writer exclusion as ROS/Dora despite never enabling or moving a motor.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "official/dora-openarm-evaluation/nodes/dora-openarm/src"))
from dora_openarm.can_ownership import CanOwnership

PARAMS = {7: "MST_ID", 8: "ESC_ID", 10: "CTRL_MODE"}
ERR_FLAG = 0x20000000
INVALID_REPLY_FLAGS = 0xC0000000  # extended or RTR


def snapshot(interface: str) -> dict:
    result = subprocess.run(
        ["ip", "-j", "-d", "-s", "link", "show", "dev", interface],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)[0]


def check_link(link: dict) -> None:
    info = link.get("linkinfo", {})
    if info.get("info_kind") != "can":
        raise RuntimeError("Expected a physical CAN interface")
    if "UP" not in link.get("flags", []) or info["info_data"].get("state") != "ERROR-ACTIVE":
        raise RuntimeError("CAN must be UP / ERROR-ACTIVE before probing")
    if link.get("mtu") != 72:
        raise RuntimeError("This probe requires CAN-FD (MTU 72)")


def read_request(motor_id: int, rid: int) -> bytes:
    if not 1 <= motor_id <= 8 or rid not in PARAMS:
        raise ValueError("Only configured IDs 1..8 and read-only ID/mode queries allowed")
    payload = struct.pack("<HBBI", motor_id, 0x33, rid, 0)
    return struct.pack("=IBB2x64s", 0x7FF, 8, 1, payload)  # CANFD_BRS


def parse_reply(frame: bytes, motor_id: int, rid: int) -> dict | None:
    if len(frame) not in (16, 72):
        return None
    can_id, length = struct.unpack_from("=IB", frame)
    if can_id & ERR_FLAG:
        raise RuntimeError(f"CAN error frame: 0x{can_id:x}, data={frame[8:16].hex()}")
    if can_id & INVALID_REPLY_FLAGS or length != 8:
        return None
    reply_id, opcode, reply_rid, value = struct.unpack_from("<HBBI", frame, 8)
    if reply_id != motor_id or opcode != 0x33 or reply_rid != rid:
        return None
    return {"reply_can_id": can_id, "value": value}


def probe(interface: str, motor_ids: list[int]) -> dict:
    before = snapshot(interface)
    report = {"interface": interface, "before": before, "motors": {}, "ok": False}
    try:
        check_link(before)
        with CanOwnership(interface, "parameter-read-probe"), \
                socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
            # Receive errors as well as data; local TX echo cannot count as a reply.
            # CPython does not expose CAN_RAW_ERR_FILTER on all versions.
            sock.setsockopt(socket.SOL_CAN_RAW, 2, struct.pack("=I", 0x1FFFFFFF))
            sock.bind((interface,))
            for motor_id in motor_ids:
                values = {}
                report["motors"][str(motor_id)] = values
                for rid, name in PARAMS.items():
                    check_link(snapshot(interface))
                    sock.settimeout(0.15)
                    sock.send(read_request(motor_id, rid))
                    deadline = time.monotonic() + 0.15
                    reply = None
                    # Always drain for 150 ms: <= 6.7 requests/s, no retry flood.
                    while (remaining := deadline - time.monotonic()) > 0:
                        sock.settimeout(remaining)
                        try:
                            frame = sock.recv(72)
                        except TimeoutError:
                            break
                        parsed = parse_reply(frame, motor_id, rid)
                        if parsed is not None:
                            reply = parsed
                    values[name] = reply
                values["verified"] = (
                    all(values[name] is not None for name in PARAMS.values())
                    and values["MST_ID"]["value"] == motor_id + 0x10
                    and values["ESC_ID"]["value"] == motor_id
                    and all(values[name]["reply_can_id"] == motor_id + 0x10 for name in PARAMS.values())
                )
                values["mit_mode"] = bool(values["CTRL_MODE"] and values["CTRL_MODE"]["value"] == 1)
            report["ok"] = all(m["verified"] and m["mit_mode"] for m in report["motors"].values())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
    finally:
        report["after"] = snapshot(interface)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", choices=["can0", "can1"])
    parser.add_argument("--ids", nargs="+", type=int, choices=range(1, 9), default=list(range(1, 9)))
    args = parser.parse_args()
    try:
        report = probe(args.interface, list(dict.fromkeys(args.ids)))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report = {"interface": args.interface, "ok": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
