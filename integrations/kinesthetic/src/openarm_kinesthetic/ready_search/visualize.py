"""Visual comparison of the Top-N candidates.

Two views over the same machinery:

* ``render_candidates`` writes one captioned two-panel PNG per candidate, so the
  comparison survives without a display and can be attached to a review;
* ``view_candidates`` opens the interactive MuJoCo viewer and switches between the
  Top-N with the number keys, which is how an operator judges "is this pose relaxed,
  is the wrist camera cable direction sensible, is the height above the table right".

Both reuse the scene's existing operator-only mocap handle (group 5, hidden from
recorded camera images) as the TCP marker: it is moved to the candidate TCP with the
candidate's orientation, so the orange sphere and its RGB axes show the tool frame the
search generated. The marker is what makes "where does the tool point" visible; a list
of joint angles does not convey it.

The second panel uses a free camera aimed at the Teaching Ready anchor rather than the
scene's ceiling camera: from directly overhead the arm occupies little of the frame and
successive candidates look almost identical. Nothing in this module touches hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..scene.model import name_id, reset_task_ready
from .config import ReadySearchConfig, load_ready_search_config, report_output_path
from .tool_frame import TOOL_SITE, derive_approach_axis

MARKER_BODY = "right_handle_target"
# MuJoCo's default render option enables geom groups 0-2 only. The operator TCP marker
# lives in group 5 (the group that is also hidden from recorded camera images), so
# without this it would never be drawn and "where does the tool point" would not be
# visible in either the screenshots or the interactive viewer.
MARKER_GEOM_GROUP = 5


def scene_option(marker_geom_group: int = MARKER_GEOM_GROUP):
    """Render option that keeps the default groups and adds the operator marker group."""
    import mujoco

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[int(marker_geom_group)] = 1
    return option


def apply_candidate(
    model,
    data,
    mapping,
    q: Sequence[float],
    *,
    marker_body: str = MARKER_BODY,
    marker_offset_along_approach_m: float = 0.0,
) -> None:
    """Install one candidate configuration and move the operator marker onto its TCP.

    The marker keeps the TCP's own orientation, so its RGB axes are the tool frame. It is
    optionally shifted along the approach axis: the grasp TCP sits *between the fingers*,
    so a marker drawn exactly there is hidden by the gripper in most poses and the tool
    direction - the thing a visual comparison is for - would not be visible.
    """
    import mujoco

    data.qpos[mapping.right_qpos] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    data.ctrl[mapping.right_actuators] = np.asarray(q, dtype=float)
    mujoco.mj_forward(model, data)

    body = name_id(model, mujoco.mjtObj.mjOBJ_BODY, marker_body)
    mocap = int(model.body_mocapid[body])
    if mocap < 0:
        raise ValueError(f"{marker_body} is not a mocap body")
    site = name_id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)
    site_rotation = np.asarray(data.site_xmat[site], dtype=float).reshape(3, 3)
    position = np.asarray(data.site_xpos[site], dtype=float).copy()
    if marker_offset_along_approach_m:
        approach_axis, _, _ = derive_approach_axis(model, site)
        position = position + float(marker_offset_along_approach_m) * (site_rotation @ approach_axis)
    data.mocap_pos[mocap] = position
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, site_rotation.reshape(9))
    data.mocap_quat[mocap] = quaternion
    mujoco.mj_forward(model, data)


def build_scene_for_report(config: ReadySearchConfig):
    from .search import build_search_scene

    return build_search_scene(config)


def candidate_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(report.get("top_n", []))


@dataclass(frozen=True)
class RenderView:
    """One render view: either a named scene camera or a free camera aimed at a point."""

    name: str
    camera: str | None = None
    lookat: tuple[float, float, float] | None = None
    azimuth_deg: float = 140.0
    elevation_deg: float = -25.0
    distance_m: float = 0.9


def render_views(config: ReadySearchConfig, report: Mapping[str, Any]) -> list[RenderView]:
    anchor = tuple(float(value) for value in report.get("anchor", {}).get("world_m", (0.35, 0.0, 0.7)))
    block = dict(config.visualization)
    views = [RenderView(name="ceiling", camera=str(block.get("camera", "ceiling")))]
    if bool(block.get("overview_panel", True)):
        overview = dict(block.get("overview", {}) or {})
        configured_lookat = overview.get("lookat_world_m")  # null means "use the anchor"
        lookat = tuple(float(value) for value in (configured_lookat or anchor))
        views.append(
            RenderView(
                name="oblique",
                lookat=lookat,
                azimuth_deg=float(overview.get("azimuth_deg", 140.0)),
                elevation_deg=float(overview.get("elevation_deg", -25.0)),
                distance_m=float(overview.get("distance_m", 0.9)),
            )
        )
    return views


def _capture(renderer, data, view: RenderView, option=None):
    import mujoco

    option = option if option is not None else scene_option()
    if view.camera is not None:
        renderer.update_scene(data, camera=view.camera, scene_option=option)
        return renderer.render()
    camera = mujoco.MjvCamera()
    camera.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    camera.fixedcamid = -1
    camera.lookat[:] = np.asarray(view.lookat, dtype=float)
    camera.distance = float(view.distance_m)
    camera.azimuth = float(view.azimuth_deg)
    camera.elevation = float(view.elevation_deg)
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


def render_candidates(
    report: Mapping[str, Any],
    *,
    config: ReadySearchConfig | None = None,
    output_dir: Path | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Path]:
    """Render one captioned panel image per Top-N candidate. Needs an offscreen GL backend."""
    import mujoco

    config = config or load_ready_search_config()
    render_width = int(width or config.visualization.get("render_width", 640))
    render_height = int(height or config.visualization.get("render_height", 480))
    directory = output_dir or report_output_path(config).parent
    directory.mkdir(parents=True, exist_ok=True)

    model, data, mapping = build_scene_for_report(config)
    views = render_views(config, report)
    marker = str(config.visualization.get("marker_body", MARKER_BODY))
    marker_offset = float(config.visualization.get("marker_offset_along_approach_m", 0.0))
    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    option = scene_option()
    written: list[Path] = []
    # The shortlist changes between runs, so a previous run's `top1_<other label>.png`
    # would otherwise stay on disk next to this run's `top1_...`, which is worse than
    # having no image at all: it looks like both are the current ranking.
    stale = sorted(directory.glob("top*_*.png")) + sorted(
        directory.glob("j4_pi2_*.png")
    )
    try:
        for rank, item in enumerate(candidate_rows(report), start=1):
            apply_candidate(
                model,
                data,
                mapping,
                item["q_rad"],
                marker_body=marker,
                marker_offset_along_approach_m=marker_offset,
            )
            panels = [np.asarray(_capture(renderer, data, view, option), dtype=np.uint8) for view in views]
            path = directory / f"top{rank}_{item['label']}.png"
            _write_composite(path, panels, _caption(rank, item, views))
            written.append(path)
    finally:
        renderer.close()
    for old in stale:
        if old not in written:
            old.unlink()
    return written


def _caption(rank: int, item: Mapping[str, Any], views: Sequence[RenderView]) -> str:
    return (
        f"#{rank} {item['label']}  family={item['family']}  "
        f"tilt={item['tcp_pose']['tilt_below_horizontal_deg']:.1f}deg  "
        f"J4={item['j4_rad']:.4f} rad  J4margin={item['j4_margin_normalized']:.3f}  "
        f"tauJ4={item['static_gravity_torque_j4_nm']:+.3f} Nm  "
        f"traj max|tauJ4|={item.get('trajectory_maximum_absolute_j4_gravity_torque_nm')}  "
        f"score={item['trajectory_score']:.4f}   [{', '.join(view.name for view in views)}]"
    )


def _write_composite(path: Path, panels: Sequence[np.ndarray], caption: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("rendering requires Pillow (the project's `sim` extra)") from exc

    height = max(int(panel.shape[0]) for panel in panels)
    widths = [int(panel.shape[1]) for panel in panels]
    band = 18
    canvas = Image.new("RGB", (sum(widths), height + band), (24, 26, 30))
    offset = 0
    for panel, width in zip(panels, widths):
        canvas.paste(Image.fromarray(np.asarray(panel, dtype=np.uint8)), (offset, band))
        offset += width
    ImageDraw.Draw(canvas).text((4, 4), caption, fill=(235, 235, 235))
    canvas.save(path)


def view_candidates(
    report: Mapping[str, Any],
    *,
    config: ReadySearchConfig | None = None,
    start_index: int = 0,
) -> dict[str, Any]:
    """Interactive viewer over the Top-N candidates: number keys switch, ESC quits."""
    import mujoco
    import mujoco.viewer

    config = config or load_ready_search_config()
    rows = candidate_rows(report)
    if not rows:
        raise ValueError("the report contains no Top-N candidates to display")
    model, data, mapping = build_scene_for_report(config)
    marker = str(config.visualization.get("marker_body", MARKER_BODY))
    marker_offset = float(config.visualization.get("marker_offset_along_approach_m", 0.0))
    keys = {int(key): int(index) for key, index in config.visualization["viewer_keys"].items()}
    for index in range(min(9, len(rows))):
        keys.setdefault(index + 1, index)

    def show(index: int) -> None:
        apply_candidate(
            model,
            data,
            mapping,
            rows[index]["q_rad"],
            marker_body=marker,
            marker_offset_along_approach_m=marker_offset,
        )

    state = {"index": int(start_index)}
    show(state["index"])

    def on_key(keycode: int) -> None:
        character = chr(keycode) if 0 < keycode < 0x110000 else ""
        if character.isdigit() and int(character) in keys:
            index = keys[int(character)]
            if 0 <= index < len(rows):
                state["index"] = index
                show(index)
                print(f"[ready-search] showing #{index + 1} {rows[index]['label']}", flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        # The viewer starts with geom groups 0-2 enabled; the TCP marker is group 5.
        viewer.opt.geomgroup[MARKER_GEOM_GROUP] = 1
        for index, row in enumerate(rows, start=1):
            print(
                f"[ready-search] #{index} {row['label']}  J4={row['j4_rad']:.4f}  "
                f"tauJ4={row['static_gravity_torque_j4_nm']:.4f} Nm  score={row['trajectory_score']}",
                flush=True,
            )
        available = "/".join(str(index) for index in range(1, min(9, len(rows)) + 1))
        print(
            f"[ready-search] press {available} to switch candidate, ESC to quit",
            flush=True,
        )
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
    return {"displayed": len(rows), "last_index": state["index"]}


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_text(report: Mapping[str, Any]) -> str:
    from .report import render_comparison_table, render_family_table, render_orientation_table

    return "\n\n".join(
        (render_comparison_table(report), render_family_table(report), render_orientation_table(report))
    )


def reset_scene(config: ReadySearchConfig | None = None):
    """Compile the scene at the task-ready keyframe, for callers that only want to look."""
    import mujoco

    from ..scene.model import load_model

    config = config or load_ready_search_config()
    model = load_model(config.scene)
    data = mujoco.MjData(model)
    mapping = reset_task_ready(model, data)
    return model, data, mapping
