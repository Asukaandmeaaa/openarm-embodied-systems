"""Gate A: geometry, collision, cameras, and 16D interface acceptance."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..scene.model import load_model, reset_task_ready


def validate_gate_a(*, render: bool = True) -> dict:
    import mujoco

    model = load_model()
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)

    def geom(name: str) -> int:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index < 0:
            raise RuntimeError(f"required scene collision geom missing: {name}")
        if not model.geom_contype[index] or not model.geom_conaffinity[index]:
            raise RuntimeError(f"collision disabled unexpectedly: {name}")
        return int(index)

    table = geom("table_top_collision")
    cube = geom("blue_cube_collision")
    box_geoms = [geom("black_box_floor")] + [
        geom(f"black_box_wall_{axis}_{side}")
        for axis in ("x", "y")
        for side in ("neg", "pos")
    ]
    expected_half_size = np.asarray([0.30, 0.20, 0.02])
    if not np.allclose(model.geom_size[table], expected_half_size, atol=1e-12):
        raise RuntimeError(f"table half-size mismatch: {model.geom_size[table].tolist()}")

    side_counts: dict[str, int] = {}
    for side in ("left", "right"):
        count = 0
        visible = 0
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if side not in name and side not in body_name:
                continue
            if model.geom_contype[gid] and model.geom_conaffinity[gid]:
                count += 1
            if model.geom_rgba[gid, 3] > 0:
                visible += 1
        if count == 0 or visible == 0:
            raise RuntimeError(f"{side} arm must remain visible and collidable")
        side_counts[side] = count

    state = mapping.dataset_vector(data.qpos)
    if state.shape != (16,) or state.dtype != np.float32:
        raise RuntimeError("dataset vector is not float32[16]")
    expected_right = data.qpos[mapping.right_qpos]
    if not np.allclose(state[:7], expected_right):
        raise RuntimeError("dataset vector is not right8 then left8")

    rendered: dict[str, dict] = {}
    if render:
        # EGL is deterministic and does not connect to cameras or a display.
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(model, height=240, width=320)
        scene_option = mujoco.MjvOption(); scene_option.geomgroup[5] = 0
        try:
            for camera in ("ceiling", "camera_wrist_right"):
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                image = renderer.render()
                if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                    raise RuntimeError(f"{camera}: unexpected RGB render {image.shape}/{image.dtype}")
                rendered[camera] = {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "mean_rgb": np.mean(image, axis=(0, 1)).round(3).tolist(),
                }
        finally:
            renderer.close()

    return {
        "gate": "A",
        "status": "PASS",
        "scene": "real_table_v1",
        "provisional": True,
        "nq": int(model.nq),
        "nu": int(model.nu),
        "table_top_full_size_m": (2 * model.geom_size[table]).tolist(),
        "cube_full_size_m": (2 * model.geom_size[cube]).tolist(),
        "black_box_collision_geoms": len(box_geoms),
        "arm_collision_geom_counts": side_counts,
        "dataset_state": {"shape": list(state.shape), "dtype": str(state.dtype), "order": "right8_then_left8"},
        "renders": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    report=validate_gate_a(render=not args.no_render)
    from ..paths import workspace_path
    output=workspace_path("results","geometry","gate_a_report.json");output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
