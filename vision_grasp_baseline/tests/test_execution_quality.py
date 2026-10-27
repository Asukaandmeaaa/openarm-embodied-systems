import numpy as np
import pytest
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from vision_grasp_baseline.execution_quality import endpoint_quality,task_joint4_bounds


def pose(xyz,pitch=.7):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',pitch).as_matrix();t[:3,3]=xyz
    return t


def test_reported_side_clearance_miss_is_rejected():
    target=pose([.28,-.20,.43])
    actual=pose([.251922572,-.253546250,.316621606])
    result=endpoint_quality(target,actual)
    assert not result['accepted']
    assert result['position_error_m']==pytest.approx(.128492035,abs=1e-8)


def test_recorded_normal_release_tracking_remains_accepted():
    target=pose([.375716417,.030003230,.398153187])
    actual=pose([.377045641,.024066358,.398551858],.711)
    assert endpoint_quality(target,actual)['accepted']


def test_invalid_feedback_never_counts_as_arrival():
    bad=np.eye(4);bad[2,3]=np.nan
    with pytest.raises(ValueError,match='Invalid TCP'):
        endpoint_quality(np.eye(4),bad)


def test_startup_extension_cannot_be_used_by_task_trajectory():
    names=['other','openarm_right_joint4']
    task_joint4_bounds(names,[SimpleNamespace(positions=[-1,-.02])])
    with pytest.raises(ValueError):
        task_joint4_bounds(names,[SimpleNamespace(positions=[0,-.058])])
