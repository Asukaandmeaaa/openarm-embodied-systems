"""Shared collision-checked above-box, place, and retreat planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from ..config import load_yaml
from ..scene.model import ModelMap
from .kinematics import IKResult, check_joint_path, joint_limits, solve_site_position


class BoxPathPlanningError(RuntimeError):
    """Raised when a box-path stage has no collision-safe IK solution."""


@dataclass(frozen=True)
class BoxPathPlan:
    above_box_q: np.ndarray
    place_q: np.ndarray
    retreat_q: np.ndarray
    above_box_xyz: np.ndarray
    place_xyz: np.ndarray
    retreat_xyz: np.ndarray
    minimum_clearance_m: float
    minimum_joint_margin_normalized: float
    maximum_ik_error_m: float
    plan_hash: str

    def as_dict(self) -> dict:
        return {
            "stages": {
                "above_box": {"target_xyz_m": self.above_box_xyz.tolist(), "q": self.above_box_q.tolist()},
                "place": {"target_xyz_m": self.place_xyz.tolist(), "q": self.place_q.tolist()},
                "retreat": {"target_xyz_m": self.retreat_xyz.tolist(), "q": self.retreat_q.tolist()},
            },
            "minimum_clearance_m": self.minimum_clearance_m,
            "minimum_joint_margin_normalized": self.minimum_joint_margin_normalized,
            "maximum_ik_error_m": self.maximum_ik_error_m,
            "plan_hash": self.plan_hash,
        }


def _select_stage(
    model,
    data,
    mapping: ModelMap,
    stage: str,
    target: np.ndarray,
    start_q: np.ndarray,
    seeds: list[np.ndarray],
    cfg: dict,
) -> tuple[IKResult, float, float]:
    solutions = [
        solve_site_position(
            model,
            data,
            mapping,
            target,
            seed,
            tolerance_m=float(cfg["ik_position_tolerance_m"]),
            iterations=int(cfg["ik_iterations"]),
            margin_rad=float(cfg["joint_margin_rad"]),
        )
        for seed in seeds
    ]
    valid = [solution for solution in solutions if solution.success]
    safe: list[tuple[IKResult, float, float]] = []
    collision_failures: set[str] = set()
    for solution in valid:
        failures, clearance, margin = check_joint_path(
            model,
            data,
            mapping,
            start_q,
            solution.q,
            step_rad=float(cfg["path_joint_step_rad"]),
        )
        collision_failures.update(failures)
        if not failures and clearance >= float(cfg["tcp_table_clearance_m"]):
            safe.append((solution, clearance, margin))
    if safe:
        return safe[0]
    if valid:
        reason = sorted(collision_failures)[0] if collision_failures else "tcp_table_clearance"
        raise BoxPathPlanningError(f"{stage}_{reason}")
    best = min(solutions, key=lambda solution: solution.position_error_m)
    raise BoxPathPlanningError(f"{stage}_{best.failure}")


def plan_box_path(
    model,
    data,
    mapping: ModelMap,
    start_q: np.ndarray,
    box_world_xy: np.ndarray,
    cfg: dict | None = None,
) -> BoxPathPlan:
    cfg = cfg or load_yaml("layout_scan.yaml")
    box_cfg = cfg["box_path"]
    xy = np.asarray(box_world_xy, dtype=float)
    targets = {
        "above_box": np.r_[xy, float(box_cfg["above_z_world_m"])],
        "place": np.r_[xy, float(box_cfg["place_z_world_m"])],
        "retreat": np.r_[xy, float(box_cfg["retreat_z_world_m"])],
    }
    lower, upper = joint_limits(model, mapping, float(cfg["joint_margin_rad"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    random_seeds = [lower + (upper - lower) * rng.random(7) for _ in range(6)]
    above_seeds = [np.asarray(seed, dtype=float) for seed in cfg["above_box_branch_seeds"]]
    above_seeds += [np.asarray(start_q, dtype=float), *random_seeds]
    above, clearance_a, margin_a = _select_stage(
        model, data, mapping, "above_box", targets["above_box"], start_q, above_seeds, cfg
    )
    place_seeds = [np.asarray(seed, dtype=float) for seed in cfg["placement_branch_seeds"]]
    place_seeds += [above.q, *random_seeds]
    place, clearance_p, margin_p = _select_stage(
        model, data, mapping, "place", targets["place"], above.q, place_seeds, cfg
    )
    retreat_seeds = [above.q]
    retreat_seeds += [np.asarray(seed, dtype=float) for seed in cfg["above_box_branch_seeds"]]
    retreat_seeds += [place.q, *random_seeds]
    retreat, clearance_r, margin_r = _select_stage(
        model, data, mapping, "retreat", targets["retreat"], place.q, retreat_seeds, cfg
    )
    payload = {
        "seed": int(cfg["seed"]),
        "targets": {name: value.tolist() for name, value in targets.items()},
        "q": {"above_box": above.q.tolist(), "place": place.q.tolist(), "retreat": retreat.q.tolist()},
    }
    plan_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return BoxPathPlan(
        above.q, place.q, retreat.q,
        targets["above_box"], targets["place"], targets["retreat"],
        min(clearance_a, clearance_p, clearance_r), min(margin_a, margin_p, margin_r),
        max(above.position_error_m, place.position_error_m, retreat.position_error_m), plan_hash,
    )
