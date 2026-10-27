import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vision_grasp_baseline.waypoint_place_plan import intermediate_poses


def fixtures():
    start=np.eye(4);start[:3,3]=[.338,-.172,.395]
    start[:3,:3]=Rotation.from_euler('xyz',[0.,np.pi/2,1.45]).as_matrix()
    release=start.copy();release[:3,3]=[.40,.03,.40]
    release[:3,:3]=Rotation.from_rotvec([.15,0,0]).as_matrix()@start[:3,:3]
    return start,release


def test_orientation_changes_only_at_pickup_clearance():
    start,release=fixtures();steps=intermediate_poses(start,release)
    assert len(steps)==13
    for name,t,rotating in steps:
        if rotating:
            assert np.allclose(t[:2,3],start[:2,3])
            assert t[2,3]==pytest.approx(.43)
        elif name!='vertical_clearance':
            assert np.allclose(t[:3,:3],release[:3,:3])


def test_translation_intermediates_and_return_are_horizontal():
    start,release=fixtures();steps=intermediate_poses(start,release)
    forward=[t for name,t,_ in steps if name.startswith('transfer_')]
    back=[t for name,t,_ in steps if name.startswith('return_')]
    assert len(forward)==len(back)==4
    assert all(t[2,3]==pytest.approx(.43) for t in forward+back)
    assert np.allclose(forward[-1][:2,3],release[:2,3])
    assert np.allclose(back[-1][:2,3],start[:2,3])
    assert np.allclose(steps[6][1],release)


def test_invalid_segment_count_is_rejected():
    with pytest.raises(ValueError):
        intermediate_poses(*fixtures(),segments=0)
