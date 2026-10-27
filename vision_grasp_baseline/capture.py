#!/usr/bin/env python3
"""Camera-only RGB-D capture. No ROS, CAN, or arm-control imports."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def intrinsics(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy,
                ppx=i.ppx, ppy=i.ppy, model=str(i.model), coeffs=list(i.coeffs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--serial', default='344322073972')
    ap.add_argument('--samples', type=int, default=10)
    args = ap.parse_args()
    if args.samples < 1:
        ap.error('samples must be positive')
    if os.environ.get('VISION_CAPTURE_SERVICE'):
        # The collection service owns the USB camera continuously. Never open
        # a second RealSense pipeline for perception/verification snapshots.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from vision_grasp_baseline.collection_io import rpc
        print(json.dumps(rpc('capture', output=str(Path(args.output).resolve()), samples=args.samples)))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    context = rs.context()
    devices = {d.get_info(rs.camera_info.serial_number): d
               for d in context.query_devices()}
    if args.serial not in devices:
        raise RuntimeError(f'Requested camera absent. Connected: {list(devices)}')
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_profile = profile.get_stream(rs.stream.depth)
        color_profile = profile.get_stream(rs.stream.color)
        ex = depth_profile.get_extrinsics_to(color_profile)
        align = rs.align(rs.stream.color)
        for _ in range(45):
            pipeline.wait_for_frames(5000)
        records = []
        for index in range(args.samples):
            frames = align.process(pipeline.wait_for_frames(5000))
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                raise RuntimeError('Incomplete aligned frame set')
            bgr = np.asanyarray(color.get_data()).copy()
            z16 = np.asanyarray(depth.get_data()).copy()
            cv2.imwrite(str(out / f'color_{index:03d}.png'), bgr)
            np.save(out / f'depth_{index:03d}.npy', z16)
            records.append(dict(index=index, wall_ns=time.time_ns(),
                                color_timestamp_ms=color.get_timestamp(),
                                depth_timestamp_ms=depth.get_timestamp(),
                                color_frame_number=color.get_frame_number(),
                                valid_depth_fraction=float(np.mean(z16 > 0))))
        metadata = dict(schema_version=1, serial=args.serial,
                        name=devices[args.serial].get_info(rs.camera_info.name),
                        sdk_version=importlib.metadata.version('pyrealsense2'),
                        depth_scale_m=scale, aligned_to='color',
                        color_intrinsics=intrinsics(color_profile),
                        aligned_depth_intrinsics=intrinsics(depth.profile),
                        depth_to_color_extrinsics=dict(
                            rotation_column_major=list(ex.rotation),
                            translation_m=list(ex.translation)),
                        frames=records, hardware_motion=False)
        (out / 'capture.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(dict(output=str(out), serial=args.serial,
                              samples=len(records), motion=False)))
    finally:
        if started:
            pipeline.stop()


if __name__ == '__main__':
    main()
