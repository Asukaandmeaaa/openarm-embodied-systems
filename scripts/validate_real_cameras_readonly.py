#!/usr/bin/env python3
"""Read-only, bounded V4L2 camera check; never opens CAN, ROS, or Dora."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2


def _capture_worker(name: str, path: str, frames: int, width: int, height: int,
                    output_dir: Path) -> dict:
    """Child process body so a stuck V4L2 read can be killed by the parent."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"{name}: cannot open {path}")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        requested = {"width": width, "height": height}
        negotiated = {
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": round(capture.get(cv2.CAP_PROP_FPS), 3),
            "fourcc": int(capture.get(cv2.CAP_PROP_FOURCC)),
        }
        started = time.monotonic()
        first = last = None
        means = []
        raw_shapes = []
        for index in range(frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"{name}: frame read failed at {index + 1}/{frames}")
            raw_shapes.append(list(frame.shape))
            means.append(float(frame.mean()))
            if first is None:
                first = frame.copy()
            last = frame.copy()
        elapsed = time.monotonic() - started
    finally:
        capture.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    assert first is not None and last is not None
    saved = {}
    for label, frame in (("first", first), ("last", last)):
        raw_path = output_dir / f"{name}_{label}_captured.png"
        if not cv2.imwrite(str(raw_path), frame):
            raise RuntimeError(f"{name}: failed saving {raw_path}")
        policy_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        policy_path = output_dir / f"{name}_{label}_policy_{width}x{height}.png"
        if not cv2.imwrite(str(policy_path), policy_frame):
            raise RuntimeError(f"{name}: failed saving {policy_path}")
        saved[label] = {"captured": str(raw_path), "policy_sized": str(policy_path)}

    return {
        "name": name,
        "path": str(Path(path)),
        "frames": frames,
        "requested_mode": requested,
        "negotiated_capture_properties": negotiated,
        "captured_shape_first": raw_shapes[0],
        "captured_shape_last": raw_shapes[-1],
        "captured_shapes_consistent": len({tuple(shape) for shape in raw_shapes}) == 1,
        "policy_input_size": {"width": width, "height": height},
        "policy_resize_required": raw_shapes[-1][:2] != [height, width],
        "measured_fps": round(frames / max(elapsed, 1e-9), 2),
        "mean_range": [round(min(means), 2), round(max(means), 2)],
        "samples": saved,
    }


def check_camera(name: str, path: str, frames: int, width: int, height: int,
                 output_dir: Path, timeout: float) -> dict:
    """Run capture in a killable subprocess with an end-to-end deadline."""
    with tempfile.TemporaryDirectory(prefix="openarm-camera-check-") as tmp:
        result_path = Path(tmp) / "result.json"
        worker_output = output_dir / name
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--_worker",
            "--_name", name, "--_path", path, "--_frames", str(frames),
            "--_width", str(width), "--_height", str(height),
            "--output-dir", str(worker_output), "--_result", str(result_path),
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"{name}: timed out after {timeout:g}s; child capture stopped") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"{name}: capture worker failed: {detail or completed.returncode}")
        return json.loads(result_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # These are required for the parent invocation only.  The parent starts
    # this file again in a private ``--_worker`` mode, where it passes one
    # camera as ``--_path`` instead of the two public camera arguments.
    parser.add_argument("--ceiling")
    parser.add_argument("--wrist-right")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640,
                        help="policy input width; capture driver may negotiate another size")
    parser.add_argument("--height", type=int, default=360,
                        help="policy input height; captured frames are resized for saved policy previews")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="maximum seconds per camera, including open and reads")
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_camera_readonly"),
                        help="save first/last captured frames and policy-sized previews here")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_name", help=argparse.SUPPRESS)
    parser.add_argument("--_path", help=argparse.SUPPRESS)
    parser.add_argument("--_frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_height", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if not all((args._name, args._path, args._frames, args._width,
                    args._height, args._result)):
            parser.error("worker mode requires --_name, --_path, --_frames, "
                         "--_width, --_height, and --_result")
        result = _capture_worker(args._name, args._path, args._frames,
                                 args._width, args._height, args.output_dir)
        args._result.write_text(json.dumps(result))
        return 0
    if not args.ceiling or not args.wrist_right:
        parser.error("--ceiling and --wrist-right are required")
    if args.frames < 1 or args.width < 1 or args.height < 1 or args.timeout <= 0:
        parser.error("frames, width, height, and timeout must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ceiling": check_camera("ceiling", args.ceiling, args.frames, args.width,
                                args.height, output_dir, args.timeout),
        "wrist_right": check_camera("wrist_right", args.wrist_right, args.frames,
                                    args.width, args.height, output_dir, args.timeout),
        "scope": "read-only V4L2 capture; no CAN, ROS, or Dora hardware node",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
