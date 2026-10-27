"""Tests for the real-side Teaching Ready orientation candidates.

The real station does its own IK, so what has to be right here is the geometry: the
orientations must be built about the *registered* table frame, they must reproduce the
requested tilt exactly, the legacy reference must stay available as a control, and the
sweep around the simulation prior must stay a local correction rather than a new design.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openarm_kinesthetic.real.ready_candidates import (  # noqa: E402
    APPROACH_AXIS_IN_TARGET_FRAME,
    SIMULATION_FAMILY_HEADINGS_DEG,
    SIMULATION_FAMILY_TILT_DEG,
    ReadyCandidateError,
    candidates_from_prior,
    deduplicate_orientations,
    matrix_to_xyzw,
    orientation_for_approach,
    prior_tilts_deg,
    quaternion_to_matrix,
    table_frame,
)

ANCHOR = (0.40, 0.0, 0.40)
NORMAL = (0.0, 0.0, 1.0)
# A reference orientation whose tool axis (-Z) already points straight down: the real
# analogue of the legacy fixed orientation.
LEGACY_REFERENCE_XYZW = (0.0, 0.0, 0.0, 1.0)


def _frame():
    return table_frame(ANCHOR, NORMAL)


# --------------------------------------------------------------------------------------
# Table frame
# --------------------------------------------------------------------------------------


def test_forward_axis_is_derived_from_the_registered_anchor() -> None:
    frame = _frame()
    assert np.allclose(frame.forward, [1.0, 0.0, 0.0])
    assert "anchor" in frame.forward_source
    assert np.allclose(frame.normal, [0.0, 0.0, 1.0])
    # lateral = normal x forward, the same right-handed convention the simulator used.
    assert np.allclose(frame.lateral, np.cross(frame.normal, frame.forward))
    assert np.allclose(frame.lateral, [0.0, 1.0, 0.0])


def test_table_frame_follows_a_registration_change() -> None:
    frame = table_frame((0.30, 0.30, 0.40), NORMAL)
    assert np.allclose(frame.forward, [math.sqrt(0.5), math.sqrt(0.5), 0.0], atol=1e-12)
    frame = table_frame(ANCHOR, (0.0, 0.05, 1.0))
    assert frame.normal[1] > 0.0
    assert abs(float(frame.forward @ frame.normal)) < 1e-12


def test_vertical_anchor_needs_an_explicit_hint() -> None:
    with pytest.raises(ReadyCandidateError, match="forward axis is undefined"):
        table_frame((0.0, 0.0, 0.6), NORMAL)
    frame = table_frame((0.0, 0.0, 0.6), NORMAL, forward_hint_world=(0.0, 1.0, 0.0))
    assert np.allclose(frame.forward, [0.0, 1.0, 0.0])
    assert "hint" in frame.forward_source


def test_a_degenerate_normal_or_hint_is_refused() -> None:
    with pytest.raises(ReadyCandidateError, match="non-zero"):
        table_frame(ANCHOR, (0.0, 0.0, 0.0))
    with pytest.raises(ReadyCandidateError, match="lies along the table normal"):
        table_frame((0.0, 0.0, 0.6), NORMAL, forward_hint_world=(0.0, 0.0, 1.0))
    with pytest.raises(ReadyCandidateError, match="three finite values"):
        table_frame(ANCHOR, (0.0, 1.0))


# --------------------------------------------------------------------------------------
# Orientation construction
# --------------------------------------------------------------------------------------

# The legacy fixed orientation points the tool straight down, so its tilt is the 90 degree
# end of the same scale. Making it a member of the scale is what makes the comparison
# meaningful, and it is asserted here rather than assumed.
SCALE_CASES = (0.0, 30.0, 40.0, 50.0, 60.0, 89.3, 90.0)
TARGET_AXIS = np.asarray(APPROACH_AXIS_IN_TARGET_FRAME, dtype=float)


@pytest.mark.parametrize("tilt", SCALE_CASES)
def test_generated_orientation_reproduces_the_requested_tilt(tilt: float) -> None:
    frame = _frame()
    orientation, approach = orientation_for_approach(
        frame, quaternion_to_matrix(LEGACY_REFERENCE_XYZW),
        tilt_below_horizontal_deg=tilt, heading_deg=0.0,
    )
    assert np.allclose(orientation @ TARGET_AXIS, approach, atol=1e-12)
    assert float(-approach @ frame.normal) == pytest.approx(math.sin(math.radians(tilt)), abs=1e-12)
    assert np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(orientation)) == pytest.approx(1.0)


def test_heading_rotates_the_approach_axis_in_the_table_plane() -> None:
    frame = _frame()
    reference = quaternion_to_matrix(LEGACY_REFERENCE_XYZW)
    forward, _ = orientation_for_approach(
        frame, reference, tilt_below_horizontal_deg=40.0, heading_deg=0.0
    )
    lateral, _ = orientation_for_approach(
        frame, reference, tilt_below_horizontal_deg=40.0, heading_deg=90.0
    )
    assert np.allclose(forward @ TARGET_AXIS, [math.cos(math.radians(40)), 0.0, -math.sin(math.radians(40))], atol=1e-12)
    assert np.allclose(lateral @ TARGET_AXIS, [0.0, math.cos(math.radians(40)), -math.sin(math.radians(40))], atol=1e-12)


def test_roll_turns_about_the_approach_axis_only() -> None:
    frame = _frame()
    reference = quaternion_to_matrix(LEGACY_REFERENCE_XYZW)
    plain, approach = orientation_for_approach(
        frame, reference, tilt_below_horizontal_deg=60.0, heading_deg=0.0
    )
    rolled, rolled_approach = orientation_for_approach(
        frame, reference, tilt_below_horizontal_deg=60.0, heading_deg=0.0, roll_deg=15.0
    )
    assert np.allclose(approach, rolled_approach, atol=1e-12)
    assert not np.allclose(plain, rolled)
    # The tool axis is unchanged, so a roll offset cannot move the tool direction.
    assert np.allclose(rolled @ TARGET_AXIS, plain @ TARGET_AXIS, atol=1e-12)


def test_out_of_range_tilt_is_refused() -> None:
    with pytest.raises(ReadyCandidateError, match=r"\[-90, 90\]"):
        orientation_for_approach(
            _frame(),
            quaternion_to_matrix(LEGACY_REFERENCE_XYZW),
            tilt_below_horizontal_deg=120.0,
            heading_deg=0.0,
        )


def test_quaternion_helpers_round_trip() -> None:
    for tilt in (0.0, 37.0, 90.0):
        orientation, _ = orientation_for_approach(
            _frame(), quaternion_to_matrix(LEGACY_REFERENCE_XYZW),
            tilt_below_horizontal_deg=tilt, heading_deg=22.0, roll_deg=-7.5,
        )
        xyzw = matrix_to_xyzw(orientation)
        assert all(isinstance(value, float) for value in xyzw)
        assert float(np.linalg.norm(xyzw)) == pytest.approx(1.0)
        assert np.allclose(quaternion_to_matrix(xyzw), orientation, atol=1e-12)
    with pytest.raises(ReadyCandidateError, match="zero length"):
        quaternion_to_matrix((0.0, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------------------
# The sweep around the simulation prior
# --------------------------------------------------------------------------------------


def test_prior_tilts_are_a_local_window_around_the_simulation_families() -> None:
    tilts = prior_tilts_deg(window_deg=10.0, step_deg=5.0)
    assert set(tilts) == set(SIMULATION_FAMILY_TILT_DEG)
    for family, nominal in SIMULATION_FAMILY_TILT_DEG.items():
        values = tilts[family]
        assert nominal in values
        assert min(values) == nominal - 10.0
        assert max(values) == nominal + 10.0
        assert len(values) == 5
    # The window is a correction, not a re-sweep: nothing reaches the far end of the scale.
    assert max(max(values) for values in tilts.values()) < 90.0


def test_prior_tilts_reject_bad_settings() -> None:
    with pytest.raises(ReadyCandidateError, match="step must be positive"):
        prior_tilts_deg(window_deg=10.0, step_deg=0.0)
    with pytest.raises(ReadyCandidateError, match="non-negative"):
        prior_tilts_deg(window_deg=-1.0, step_deg=5.0)
    with pytest.raises(ReadyCandidateError, match="unknown simulation family"):
        prior_tilts_deg(window_deg=5.0, step_deg=5.0, families=("oblique_15",))


def test_candidate_sweep_shape_and_priority() -> None:
    candidates = candidates_from_prior(
        _frame(),
        LEGACY_REFERENCE_XYZW,
        tilts_deg=prior_tilts_deg(window_deg=10.0, step_deg=5.0),
        headings_deg=SIMULATION_FAMILY_HEADINGS_DEG,
    )
    labels = [item.label for item in candidates]
    assert labels[0] == "legacy_reference"
    assert len(labels) == len(set(labels))
    # oblique_60 is the simulation's best J4 trade-off, so it comes first.
    families = [item.family for item in candidates[1:]]
    assert families[0] == "oblique_60"
    assert families.index("oblique_40") < families.index("side_horizontal")
    expected = 1 + 5 + 5 + 5 * 3  # legacy + three families, side_horizontal has three headings
    assert len(candidates) == expected

    legacy = candidates[0]
    assert legacy.tilt_below_horizontal_deg == pytest.approx(90.0, abs=1e-9)
    assert np.allclose(
        legacy.orientation_matrix @ np.asarray(APPROACH_AXIS_IN_TARGET_FRAME),
        [0.0, 0.0, -1.0],
        atol=1e-12,
    )
    for item in candidates:
        # Every candidate is a proper rotation whose tool axis is exactly where it claims.
        assert np.allclose(
            item.orientation_matrix @ np.asarray(APPROACH_AXIS_IN_TARGET_FRAME),
            item.approach_axis_world,
            atol=1e-12,
        )


def test_reference_can_be_omitted() -> None:
    candidates = candidates_from_prior(
        _frame(),
        LEGACY_REFERENCE_XYZW,
        tilts_deg=prior_tilts_deg(window_deg=0.0, step_deg=5.0),
        headings_deg=SIMULATION_FAMILY_HEADINGS_DEG,
        include_reference=False,
    )
    assert all(item.label != "legacy_reference" for item in candidates)
    assert {item.tilt_below_horizontal_deg for item in candidates} == {60.0, 40.0, 0.0}


def test_near_identical_orientations_are_collapsed() -> None:
    frame = _frame()
    candidates = candidates_from_prior(
        frame,
        LEGACY_REFERENCE_XYZW,
        tilts_deg=prior_tilts_deg(window_deg=10.0, step_deg=5.0),
        headings_deg=SIMULATION_FAMILY_HEADINGS_DEG,
        rolls_deg=(0.0, 1.0),
    )
    kept, dropped = deduplicate_orientations(candidates, minimum_angle_deg=5.0)
    # A 1 degree roll leaves the approach axis alone, so those duplicates disappear while
    # every genuinely different direction survives.
    assert dropped
    assert all(
        math.degrees(
            math.acos(float(np.clip(item.approach_axis_world @ other.approach_axis_world, -1.0, 1.0)))
        )
        >= 5.0
        for index, item in enumerate(kept)
        for other in kept[index + 1 :]
    )
    assert len(kept) < len(candidates)
    # Rolls are still available when they are the point of the sweep.
    only_rolls = candidates_from_prior(
        frame, LEGACY_REFERENCE_XYZW,
        tilts_deg={"oblique_60": [60.0]},
        headings_deg={"oblique_60": [0.0]},
        rolls_deg=(-10.0, 0.0, 10.0),
        include_reference=False,
    )
    assert len(only_rolls) == 3
