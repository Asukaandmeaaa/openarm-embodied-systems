#!/usr/bin/env python3
"""Load the frozen scene and write an evidence-backed feature contract dump."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
CAMERAS = {
    "ceiling": "camera_ceiling",
    "wrist_right": "camera_wrist_right",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/p1_openarm_cube_pick/feature_contract_dump.json",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    requested_resolutions = []
    for camera_name in CAMERAS.values():
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id >= 0:
            requested_resolutions.append(tuple(map(int, model.cam_resolution[camera_id])))
    model.vis.global_.offwidth = max(width for width, _ in requested_resolutions)
    model.vis.global_.offheight = max(height for _, height in requested_resolutions)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    features = {
        "observation.state": {"shape": [16], "dtype": "float32"},
        "action": {"shape": [16], "dtype": "float32"},
        "task": {"shape": [], "dtype": "string"},
    }
    evidence = {}
    renderers = []
    try:
        for feature_name, camera_name in CAMERAS.items():
            camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if camera_id < 0:
                raise RuntimeError(f"camera missing from scene: {camera_name}")
            width, height = map(int, model.cam_resolution[camera_id])
            renderer = mujoco.Renderer(model, height=height, width=width)
            renderers.append(renderer)
            renderer.update_scene(data, camera=camera_id)
            rgb = renderer.render()
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"unexpected rendered image: {rgb.shape} {rgb.dtype}")
            if float(rgb.std()) < 1.0:
                raise RuntimeError(f"near-constant camera image: {camera_name}")
            features[f"observation.images.{feature_name}"] = {
                "shape": [3, height, width],
                "dtype": "float32",
                "source_encoding": "jpeg",
                "decoded_color": "RGB",
            }
            evidence[camera_name] = {
                "render_shape_hwc": list(rgb.shape),
                "render_dtype": str(rgb.dtype),
                "pixel_mean": float(rgb.mean()),
                "pixel_std": float(rgb.std()),
            }
    finally:
        for renderer in renderers:
            renderer.close()

    dump = {
        "contract": "openarm_cube_pick_v1",
        "scene": str(SCENE),
        "task": "Pick up the orange cube with the right arm.",
        "state_action_order": "right[8] + left[8]",
        "features": features,
        "render_evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dump, indent=2))


if __name__ == "__main__":
    main()
