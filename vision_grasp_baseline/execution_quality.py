"""Check actual TCP arrival separately from a controller's success result."""
import numpy as np
from scipy.spatial.transform import Rotation


def task_joint4_bounds(joint_names,points):
    index=list(joint_names).index('openarm_right_joint4')
    if any(float(p.positions[index])<-.020001 for p in points):
        raise ValueError('Task path enters startup-only right elbow extension; replan within original limits')


def endpoint_quality(target,actual,*,position_tolerance_m=.020,rotation_tolerance_rad=.10):
    target=np.asarray(target,float);actual=np.asarray(actual,float)
    if target.shape!=(4,4) or actual.shape!=(4,4) or not np.isfinite([target,actual]).all():
        raise ValueError('Invalid TCP transform')
    position=float(np.linalg.norm(target[:3,3]-actual[:3,3]))
    rotation=float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())
    return dict(accepted=position<=position_tolerance_m and rotation<=rotation_tolerance_rad,
                position_error_m=position,orientation_error_rad=rotation,
                position_tolerance_m=position_tolerance_m,orientation_tolerance_rad=rotation_tolerance_rad,
                limitation='Arrival check only; does not certify collision clearance or scene registration')
