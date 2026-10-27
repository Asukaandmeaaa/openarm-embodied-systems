"""Deterministic real_table_v1 layout search with classified failures."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..manifest import sha256
from ..paths import config_path, workspace_path
from ..scene.model import load_model, reset_task_ready
from .box_path import BoxPathPlanningError, plan_box_path
from .kinematics import check_joint_path, normalized_joint_margin, solve_site_pose


@dataclass
class LayoutResult:
    index: int
    table_x: float
    table_y: float
    box_table_x: float
    box_table_y: float
    cube_table_x: float
    cube_table_y: float
    ready_index: int
    success: bool
    failure: str
    minimum_clearance_m: float
    minimum_joint_margin_normalized: float
    maximum_ik_error_m: float


def _inside_table(center, half_extent, table_half, margin) -> bool:
    return all(abs(float(c)) + float(h) + margin <= float(t) for c, h, t in zip(center, half_extent, table_half))


def _candidate_product(cfg: dict, max_candidates: int | None) -> list[tuple]:
    table = list(itertools.product(cfg["table_center_xy_candidates_m"]["x"], cfg["table_center_xy_candidates_m"]["y"]))
    box = list(itertools.product(cfg["black_box_center_table_xy_candidates_m"]["x"], cfg["black_box_center_table_xy_candidates_m"]["y"]))
    cube = list(itertools.product(cfg["cube_source_center_table_xy_candidates_m"]["x"], cfg["cube_source_center_table_xy_candidates_m"]["y"]))
    ready = list(enumerate(cfg["q_right_task_ready_candidates"]))
    values = list(itertools.product(table, box, cube, ready))
    # Preserve nominal and use a seeded, reproducible subset for routine acceptance.
    if max_candidates is not None and len(values) > max_candidates:
        rng = np.random.default_rng(int(cfg["seed"]))
        chosen = sorted(rng.choice(len(values), size=max_candidates - 1, replace=False).tolist())
        nominal = (((0.43, 0.0), (-0.05, -0.08), (-0.21, -0.15), ready[1]))
        subset = [values[i] for i in chosen]
        if nominal not in subset:
            subset.append(nominal)
        values = subset
    return values


def scan_layouts(max_candidates: int | None = 120) -> tuple[list[LayoutResult], dict]:
    import mujoco

    cfg = load_yaml("layout_scan.yaml")
    model = load_model()
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    base_qpos = data.qpos.copy()
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_table_v1_table")
    box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "black_box")
    cube_body = mapping.cube_body
    results: list[LayoutResult] = []
    box_paths: dict[int, dict] = {}
    margin = float(cfg["table_edge_margin_m"])
    for index, (table_xy, box_xy, cube_xy, ready_item) in enumerate(_candidate_product(cfg, max_candidates)):
        ready_index, ready_values = ready_item
        failures: list[str] = []
        table_xy = np.asarray(table_xy, dtype=float)
        box_xy = np.asarray(box_xy, dtype=float)
        cube_xy = np.asarray(cube_xy, dtype=float)
        if not _inside_table(box_xy, (0.07, 0.07), (0.30, 0.20), margin):
            failures.append("box_outside_table")
        source_footprint = np.asarray(cfg["cube_source_half_range_xy_m"], dtype=float) + 0.02
        if not _inside_table(cube_xy, source_footprint, (0.30, 0.20), margin):
            failures.append("cube_source_region_outside_table")
        if np.all(np.abs(box_xy - cube_xy) <= np.asarray([0.11, 0.11])):
            failures.append("source_target_overlap")

        model.body_pos[table_body, :2] = table_xy
        model.body_pos[box_body, :2] = table_xy + box_xy
        model.body_pos[cube_body, :2] = table_xy + cube_xy
        data.qpos[:] = base_qpos
        data.qpos[mapping.cube_qpos : mapping.cube_qpos + 2] = table_xy + cube_xy
        data.qpos[mapping.right_qpos] = np.asarray(ready_values)
        mujoco.mj_forward(model, data)
        # Legacy expert's yaw=pi/2 square symmetry, expressed explicitly as
        # Rz(yaw) @ Ry(pi/2), keeps the wrist above the object/box.
        target_orientation = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

        q = np.asarray(ready_values, dtype=float)
        min_clearance = float(data.site_xpos[mapping.grasp_site, 2] - 0.40)
        min_joint_margin = normalized_joint_margin(model, mapping, q)
        max_error = 0.0
        cube_world = table_xy + cube_xy
        box_world = table_xy + box_xy
        # Source approach/lift and a separately seeded placement branch. The
        # joint-space connection is collision checked at fine resolution.
        targets = [
            np.r_[cube_world, 0.546],
            np.r_[cube_world, 0.456],
            np.r_[cube_world, 0.521],
        ]
        selected_box_path = None
        if not failures:
            stages = ("pregrasp", "grasp", "lift")
            for stage, target in zip(stages, targets):
                ik = solve_site_pose(
                    model, data, mapping, target, target_orientation, q,
                    tolerance_m=float(cfg["ik_position_tolerance_m"]),
                    iterations=int(cfg["ik_iterations"]), margin_rad=float(cfg["joint_margin_rad"]),
                )
                max_error = max(max_error, ik.position_error_m)
                if not ik.success:
                    failures.append(f"{stage}_{ik.failure}")
                    break
                path_failures, clearance, joint_margin = check_joint_path(
                    model, data, mapping, q, ik.q, step_rad=float(cfg["path_joint_step_rad"])
                )
                failures.extend(f"{stage}_{name}" for name in sorted(path_failures))
                min_clearance = min(min_clearance, clearance)
                min_joint_margin = min(min_joint_margin, joint_margin)
                if clearance < float(cfg["tcp_table_clearance_m"]):
                    failures.append(f"{stage}_tcp_table_clearance")
                if failures:
                    break
                q = ik.q
            if not failures:
                try:
                    selected_box_path = plan_box_path(model, data, mapping, q, box_world, cfg)
                except BoxPathPlanningError as exc:
                    failures.append(str(exc))
                else:
                    q = selected_box_path.retreat_q
                    min_clearance = min(min_clearance, selected_box_path.minimum_clearance_m)
                    min_joint_margin = min(min_joint_margin, selected_box_path.minimum_joint_margin_normalized)
                    max_error = max(max_error, selected_box_path.maximum_ik_error_m)
            if not failures:
                path_failures, clearance, joint_margin = check_joint_path(
                    model, data, mapping, q, np.asarray(ready_values), step_rad=float(cfg["path_joint_step_rad"])
                )
                failures.extend(f"return_{name}" for name in sorted(path_failures))
                min_clearance = min(min_clearance, clearance)
                min_joint_margin = min(min_joint_margin, joint_margin)
        results.append(LayoutResult(
            index=index, table_x=float(table_xy[0]), table_y=float(table_xy[1]),
            box_table_x=float(box_xy[0]), box_table_y=float(box_xy[1]),
            cube_table_x=float(cube_xy[0]), cube_table_y=float(cube_xy[1]), ready_index=int(ready_index),
            success=not failures, failure="PASS" if not failures else "|".join(dict.fromkeys(failures)),
            minimum_clearance_m=float(min_clearance), minimum_joint_margin_normalized=float(min_joint_margin),
            maximum_ik_error_m=float(max_error),
        ))
        if not failures and selected_box_path is not None:
            box_paths[index] = selected_box_path.as_dict()
    distribution = Counter(reason for item in results for reason in item.failure.split("|") if reason != "PASS")
    successes = [item for item in results if item.success]
    configured_nominal = next(
        (
            item for item in successes
            if np.allclose([item.table_x, item.table_y], [0.43, 0.0])
            and np.allclose([item.box_table_x, item.box_table_y], [-0.05, -0.08])
            and np.allclose([item.cube_table_x, item.cube_table_y], [-0.21, -0.15])
            and item.ready_index == 1
        ),
        successes[0] if successes else None,
    )
    summary = {
        "schema_version": 1,
        "seed": int(cfg["seed"]),
        "scene": "real_table_v1",
        "provisional": True,
        "candidate_count": len(results),
        "success_count": len(successes),
        "failure_distribution": dict(sorted(distribution.items())),
        "source_hashes": {
            "layout_scan_yaml_sha256": sha256(config_path("layout_scan.yaml")),
            "scene_xml_sha256": sha256(workspace_path("sim", "scenes", "real_table_v1.xml")),
        },
        "nominal": asdict(configured_nominal) if configured_nominal else None,
        "nominal_box_path": box_paths.get(configured_nominal.index) if configured_nominal else None,
    }
    return results, summary


def write_results(results: list[LayoutResult], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "layout_scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
    (output_dir / "layout_scan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Small dependency-free 2D source-region heatmap (green pass, red fail).
    cells = {}
    for item in results:
        cells.setdefault((item.cube_table_x, item.cube_table_y), []).append(item.success)
    xs = sorted({key[0] for key in cells}); ys = sorted({key[1] for key in cells}, reverse=True)
    size = 70
    rects = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            values = cells.get((x, y), [])
            ratio = sum(values) / len(values) if values else 0.0
            red, green = int(220 * (1-ratio)), int(180 * ratio)
            rects.append(f'<rect x="{60+ix*size}" y="{30+iy*size}" width="{size-2}" height="{size-2}" fill="rgb({red},{green},60)"/><text x="{65+ix*size}" y="{67+iy*size}" font-size="12">{ratio:.0%}</text>')
    width, height = 80 + len(xs)*size, 60 + len(ys)*size
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="8" y="16" font-size="13">source-region feasibility</text>{"".join(rects)}</svg>\n'
    (output_dir / "layout_heatmap.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--output", type=Path, default=workspace_path("results", "layout"))
    args = parser.parse_args()
    results, summary = scan_layouts(args.max_candidates)
    write_results(results, summary, args.output)
    print(json.dumps(summary, indent=2))
    if not summary["success_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
