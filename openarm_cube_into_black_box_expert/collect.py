"""Collector entry point for the fixed middle black-box task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openarm_cube_pick_expert import collect as legacy
from openarm_cube_pick_expert.expert import MujocoGraspPlanner, rank_candidates

from .scene import move_black_box_to_middle

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "configs/collection/openarm_cube_into_black_box_expert_params.yaml"
METADATA = ROOT / "configs/collection/openarm_cube_into_black_box_metadata.yaml"


class Collector(legacy.Collector):
    def __init__(self, output: Path, base_seed: int, render=True):
        legacy.PARAMS = PARAMS
        legacy.METADATA = METADATA
        super().__init__(output, base_seed, render)
        move_black_box_to_middle(self.model)

    def _ranked_candidate_indices(self, index: int) -> list[int]:
        """Plan once at reset and return every geometrically valid branch."""
        seed = self.base_seed + index
        yaw, _ = self.reset(seed)
        reset_state = self.state().copy()
        cube = self.data.qpos[self.cube_qadr:self.cube_qadr + 7].copy()
        planner = MujocoGraspPlanner(
            self.model,
            self.mapper,
            self.cfg,
            seed,
            runtime_data=self.data,
        )
        _, candidates = planner.plan(cube[:3], yaw, reset_state[:7])
        return [candidate.index for candidate in rank_candidates(candidates)]

    @staticmethod
    def _preflight_summary(details: dict[str, Any]) -> dict[str, Any]:
        selected = details.get("selected_candidate") or {}
        return {
            "candidate_index": selected.get("index"),
            "success": details.get("success_reason") is not None,
            "fail_reason": details.get("fail_reason"),
            "final_cube_pose": details.get("final_cube_pose"),
            "place_error_m": details.get("place_error_m"),
            "place_height_error_m": details.get("place_height_error_m"),
        }

    def run_episode(self, index: int, record: bool = True, candidate_index=None):
        """Select a candidate by deterministic, contact-aware full rollout.

        Each preflight starts from the same seeded reset and uses the exact
        executor used for recording.  Failed preflights never reach the
        recorder.  The first branch that grasps, lifts, transports, places,
        releases, returns home, and passes the static tail is replayed once
        more from reset for the committed demonstration.
        """
        if candidate_index is not None or not self.cfg.get("full_candidate_preflight", True):
            return super().run_episode(
                index, record=record, candidate_index=candidate_index
            )

        candidate_indices = self._ranked_candidate_indices(index)
        preflights: list[dict[str, Any]] = []
        selected_index: int | None = None
        last_details: dict[str, Any] | None = None
        for candidate in candidate_indices:
            ok, details = super().run_episode(
                index, record=False, candidate_index=candidate
            )
            preflights.append(self._preflight_summary(details))
            last_details = details
            if ok:
                selected_index = candidate
                break

        if selected_index is None:
            if last_details is None:
                ok, last_details = super().run_episode(
                    index, record=False, candidate_index=-1
                )
                assert not ok
            last_details["candidate_preflights"] = preflights
            last_details["preflight_selected_candidate"] = None
            underlying = last_details.get("fail_reason")
            last_details["preflight_terminal_reason"] = underlying
            last_details["fail_reason"] = "all_candidate_preflights_failed"
            return False, last_details

        context = {
            "candidate_preflights": preflights,
            "preflight_selected_candidate": selected_index,
        }
        self._episode_context = context
        try:
            ok, details = super().run_episode(
                index, record=record, candidate_index=selected_index
            )
        finally:
            del self._episode_context
        if not ok:
            details["preflight_terminal_reason"] = "formal_rollout_diverged"
        return ok, details


def main() -> None:
    legacy.Collector = Collector
    legacy.PARAMS = PARAMS
    legacy.METADATA = METADATA
    legacy.main()


if __name__ == "__main__":
    main()
