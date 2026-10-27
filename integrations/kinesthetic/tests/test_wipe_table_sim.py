import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openarm_kinesthetic.sim.wipe_table import (
    CLOTH_GEOM_NAME,
    CLOTH_HALF_SIZE_XY_M,
    CLOTH_PRESS_DEPTH_M,
    CLOTH_THICKNESS_M,
    CLOTH_TOP_Z,
    GRIPPER_TIP_PREFIXES,
    SCENE,
    TABLE_CLOTH_CENTER_Z,
    cloth_and_tip_geoms,
    cloth_replay,
    cloth_track,
    compose_picture_in_picture,
    load_trajectory,
    next_cloth_position,
    presses_cloth,
    sampled_positions,
    trajectory_report,
    video_source_times,
)


def _payload():
    return {
        "duration_s": 2.0,
        "joint_names": [f"joint{i}" for i in range(8)],
        "samples": [
            {"time_from_start_s": 0.0, "positions_rad": [0.0] * 8},
            {"time_from_start_s": 1.0, "positions_rad": [1.0] * 8},
            {"time_from_start_s": 2.0, "positions_rad": [3.0] * 8},
        ],
    }


def test_sampling_interpolates_original_timestamps_and_clamps() -> None:
    q = sampled_positions(_payload(), [-1.0, 0.5, 1.5, 3.0])
    assert q.shape == (4, 8)
    assert np.allclose(q[:, 0], [0.0, 0.5, 2.0, 3.0])


def test_video_timeline_has_one_second_holds() -> None:
    times = video_source_times(2.0, fps=30)
    assert len(times) == 120
    assert np.all(times[:31] == 0.0)
    assert times[-1] == 2.0


def test_report_describes_integrity_and_joint_extrema() -> None:
    report = trajectory_report(_payload())
    assert report["duration_s"] == 2.0
    assert report["sample_count"] == 3
    assert report["estimated_sample_rate_hz"] == 1.0
    assert report["strictly_increasing_time"] is True
    assert report["finite"] is True
    assert report["maximum_joint_step_rad"] == 2.0


def test_historical_wipe_file_uses_existing_recorder_contract() -> None:
    from openarm_kinesthetic.paths import workspace_path

    path = workspace_path(
        "data", "nano_snapshot", "openarm_kinesthetic_data", "recorded_motions",
        "right_motion_20260921_145809_163485294.json",
    )
    if not path.exists():
        return
    payload = load_trajectory(path)
    assert payload["sample_count"] == 1623
    assert payload.get("mode", "continuous") == "continuous"
    assert json.loads(path.read_text())["duration_s"] == payload["duration_s"]


def test_scene_uses_h2_physical_table_not_inflated_guard() -> None:
    root = ET.parse(SCENE).getroot()
    table = root.find(".//body[@name='wipe_table']")
    assert table is not None
    assert [float(value) for value in table.attrib["pos"].split()] == [
        0.382, 0.016, 0.250,
    ]
    top = table.find("geom[@name='wipe_table_top']")
    assert top is not None
    assert [float(value) for value in top.attrib["size"].split()] == [
        0.20, 0.30, 0.02,
    ]
    material = root.find(".//material[@name='table_material']")
    assert material is not None
    rgb = [float(value) for value in material.attrib["rgba"].split()[:3]]
    assert min(rgb) >= 0.90
    assert TABLE_CLOTH_CENTER_Z == pytest.approx(0.284)
    assert CLOTH_TOP_Z == pytest.approx(0.298)
    assert CLOTH_THICKNESS_M == pytest.approx(0.028)

    cloth = root.find(".//body[@name='purple_wipe_cloth']")
    assert cloth is not None
    visual = cloth.find(f"geom[@name='{CLOTH_GEOM_NAME}']")
    assert visual is not None
    size = [float(value) for value in visual.attrib["size"].split()]
    assert size[2] * 2 == pytest.approx(CLOTH_THICKNESS_M)
    assert size[:2] == list(CLOTH_HALF_SIZE_XY_M)
    # Only the slab may collide, so the gripper is felt against the cloth body
    # itself and not against the raised decoration on top of it.
    assert visual.attrib["contype"] == "1"
    assert visual.attrib["conaffinity"] == "1"
    for decoration in ("purple_wipe_cloth_top", "purple_wipe_cloth_seam_x1"):
        geom = cloth.find(f"geom[@name='{decoration}']")
        assert geom is not None
        assert geom.attrib["contype"] == "0"
        assert geom.attrib["conaffinity"] == "0"

    camera = root.find(".//camera[@name='camera_ceiling']")
    assert camera is not None
    assert camera.attrib["pos"] == "0.071134 0.006854 0.734108"
    assert camera.attrib["quat"] == (
        "0.66793047 0.16449449 -0.15898649 -0.70819047"
    )
    assert camera.attrib["fovy"] == "66"


def test_picture_in_picture_keeps_main_and_adds_white_border() -> None:
    main = np.zeros((12, 20, 3), dtype=np.uint8)
    inset = np.full((4, 5, 3), 80, dtype=np.uint8)
    result = compose_picture_in_picture(main, inset, margin=1, border=1)
    assert np.all(result[2:6, 2:7] == 80)
    assert np.all(result[1, 1:8] == 245)
    assert np.all(result[:, 10:] == 0)


def test_cloth_is_dragged_by_the_gripper_and_stays_when_released() -> None:
    cloth = np.asarray((0.30, -0.16, TABLE_CLOTH_CENTER_Z))
    moved_tcp = np.asarray((0.3005, -0.1595, 0.30))

    dragged, touching = next_cloth_position(
        cloth, moved_tcp, cloth, pressed=True, previously_pressed=True
    )
    assert touching is True
    assert np.allclose(dragged[:2], (0.3005, -0.1595))
    assert dragged[2] == TABLE_CLOTH_CENTER_Z

    # The tip left the cloth: it stays exactly where it was put down.
    parked, touching = next_cloth_position(
        dragged, np.asarray((0.36, -0.10, 0.34)), dragged,
        pressed=False, previously_pressed=True,
    )
    assert touching is False
    assert np.array_equal(parked, dragged)

    # Coming back down elsewhere must not teleport the cloth under the gripper.
    elsewhere = np.asarray((0.34, -0.11, 0.30))
    parked, touching = next_cloth_position(
        dragged, elsewhere, dragged, pressed=False, previously_pressed=False
    )
    assert touching is False
    assert np.array_equal(parked, dragged)

    # Re-entering the cloth drags it by the gripper's own increment only.
    resumed, touching = next_cloth_position(
        dragged, elsewhere, dragged, pressed=True, previously_pressed=True
    )
    assert touching is True
    assert np.allclose(resumed[:2], dragged[:2] + (elsewhere[:2] - dragged[:2]))


def test_cloth_track_seeds_the_first_press_and_parks_without_one() -> None:
    tcp = np.asarray([
        (0.20, -0.30, 0.36),
        (0.30, -0.16, 0.36),
        (0.3010, -0.1600, 0.30),
        (0.3060, -0.1560, 0.30),
        (0.3060, -0.1560, 0.34),
        (0.4000, -0.0500, 0.34),
    ])
    pressing = np.asarray([False, False, True, True, False, False])
    centers = cloth_track(tcp, pressing)
    # Seeded at the first press, so that press has no jump to make.
    assert np.allclose(centers[1], (0.3010, -0.1600))
    assert np.allclose(centers[2], (0.3010, -0.1600))
    assert np.allclose(centers[3], (0.3060, -0.1560))
    # Released and carried away: the cloth keeps the position it was left at.
    assert np.allclose(centers[4], (0.3060, -0.1560))
    assert np.allclose(centers[5], (0.3060, -0.1560))


def _recorded_payload():
    from openarm_kinesthetic.paths import workspace_path

    path = workspace_path(
        "data", "nano_snapshot", "openarm_kinesthetic_data", "recorded_motions",
        "right_motion_20260921_145809_163485294.json",
    )
    if not path.exists():
        return None, None
    payload = load_trajectory(path)
    positions = sampled_positions(
        payload, video_source_times(float(payload["duration_s"]))
    )
    return payload, positions


def _tcp_series(mujoco, model, data, resolver, positions):
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame"
    )
    tcp = np.empty((len(positions), 3), dtype=float)
    for index, q in enumerate(positions):
        resolver.set_qpos(data.qpos, q, "right")
        mujoco.mj_forward(model, data)
        tcp[index] = data.site_xpos[site]
    return tcp


def _contact_depth(mujoco, data, cloth_geom, tip_geoms) -> float:
    depth = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (contact.geom1, contact.geom2)
        if cloth_geom in pair and (pair[0] in tip_geoms or pair[1] in tip_geoms):
            depth = max(depth, -float(contact.dist))
    return depth


def test_scene_exposes_gripper_tip_and_cloth_collision_geoms() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    cloth_geom, tip_geoms = cloth_and_tip_geoms(mujoco, model)
    assert model.geom_contype[cloth_geom] == 1
    assert model.geom_conaffinity[cloth_geom] == 1
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        for geom in tip_geoms
    ]
    assert len(tip_geoms) >= 9
    assert all(name.startswith(GRIPPER_TIP_PREFIXES) for name in names)
    assert any("finger_inner_right" in name for name in names)
    assert any("finger_outer_right" in name for name in names)
    # The table and floor must stay out of collision so the kinematic replay
    # cannot generate spurious contacts.
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name in ("floor", "wipe_table_top"):
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_recorded_wipe_presses_only_while_the_tip_is_inside_the_cloth() -> None:
    """The drag must come from real penetration, not from a height guess."""
    payload, positions = _recorded_payload()
    if payload is None:
        return
    import mujoco

    source_t = video_source_times(float(payload["duration_s"]))
    (
        mujoco_, model, data, resolver, cloth_mocap, centers, pressing
    ) = cloth_replay(payload, positions)
    cloth_geom, tip_geoms = cloth_and_tip_geoms(mujoco_, model)
    tcp = _tcp_series(mujoco_, model, data, resolver, positions)

    # Every frame flagged as a press must really be a penetration of the tip.
    for index in np.flatnonzero(pressing):
        resolver.set_qpos(data.qpos, positions[index], "right")
        data.mocap_pos[cloth_mocap] = (
            centers[index][0], centers[index][1], TABLE_CLOTH_CENTER_Z
        )
        mujoco_.mj_forward(model, data)
        assert presses_cloth(mujoco_, data, cloth_geom, tip_geoms)
        assert _contact_depth(mujoco_, data, cloth_geom, tip_geoms) >= CLOTH_PRESS_DEPTH_M

    # ... and the two recorded wipe phases are exactly the pressed phases.
    def phases(mask):
        on = np.flatnonzero(mask)
        out = []
        start = previous = on[0]
        for index in on[1:]:
            if index != previous + 1:
                out.append((float(source_t[start]), float(source_t[previous])))
                start = index
            previous = index
        out.append((float(source_t[start]), float(source_t[previous])))
        return out

    spans = phases(pressing)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(8.5, abs=0.2)
    assert spans[0][1] == pytest.approx(24.7, abs=0.2)
    assert spans[1][0] == pytest.approx(26.3, abs=0.2)
    assert spans[1][1] == pytest.approx(33.8, abs=0.2)

    # The repositioning pause and the walk-away must leave the cloth untouched.
    paused = (source_t > 24.8) & (source_t < 26.2)
    assert not pressing[paused].any()
    assert not pressing[source_t > 34.0].any()


def test_recorded_wipe_never_teleports_the_cloth() -> None:
    """The reported bug: lifting and replacing the gripper snapped the cloth.

    The cloth may only ever move as far as the gripper moved it, so no frame of
    the recording can contain a jump.
    """
    payload, positions = _recorded_payload()
    if payload is None:
        return
    (
        mujoco, model, data, resolver, cloth_mocap, centers, pressing
    ) = cloth_replay(payload, positions)
    tcp = _tcp_series(mujoco, model, data, resolver, positions)
    source_t = video_source_times(float(payload["duration_s"]))

    cloth_step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    tcp_step = np.linalg.norm(np.diff(tcp[:, :2], axis=0), axis=1)
    assert cloth_step.max() <= tcp_step.max() + 1e-9
    assert cloth_step.max() < 0.005

    # Both measured lift-and-replace pauses must leave the cloth parked.
    paused = (~pressing[1:]) & (~pressing[:-1]) & (np.diff(tcp[:, 2]) != 0.0)
    assert paused.any()
    assert np.allclose(cloth_step[paused], 0.0, atol=1e-12)
    assert (source_t[1:][paused] > 24.0).any()

    # ... and the cloth has to end up on the tabletop, not hanging off it.
    corners = np.asarray(np.meshgrid([-1, 1], [-1, 1])).T.reshape(-1, 2)
    extent = centers[:, None, :] + corners[None, :, :] * CLOTH_HALF_SIZE_XY_M
    assert extent[:, :, 0].min() >= 0.182
    assert extent[:, :, 1].min() >= -0.284


def test_preview_inset_matches_the_recorded_layout() -> None:
    import mujoco

    from openarm_kinesthetic.sim.wipe_table import (
        INSET_BORDER,
        INSET_HEIGHT,
        INSET_MARGIN,
        INSET_WIDTH,
        inset_rect,
    )

    class _Viewport:
        height = 800
        width = 1200

    rect = inset_rect(mujoco, _Viewport)
    assert (rect.width, rect.height) == (INSET_WIDTH, INSET_HEIGHT)
    assert rect.left == INSET_MARGIN + INSET_BORDER
    # Top-left corner, so the inset sits where the video writes it.
    assert rect.bottom + INSET_HEIGHT + INSET_MARGIN + INSET_BORDER == 800
