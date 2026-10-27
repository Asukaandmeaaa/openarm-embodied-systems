"""A fixed conservative table box, with explicit admission of fresh detections.

Build from all captured table boxes, including the original 3cm edge and 3mm
surface padding. Never substitute a smaller latest-frame box during replay.
Coordinates here are calibration-world coordinates, before local registration.
"""
from copy import deepcopy
from itertools import product
import math

import numpy as np


def table_box(table):
    n=np.asarray(table['normal'],dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>.001 or n[2]<.9:
        raise ValueError('Invalid table normal')
    x=np.array([math.cos(table['yaw']),math.sin(table['yaw']),0.])
    x-=n*(x@n);x/=np.linalg.norm(x)
    rotation=np.column_stack((x,np.cross(n,x),n))
    center=np.r_[table['center_xy'],table['top_z']]-n*(.35/2-.003)
    size=np.r_[np.maximum(table['size_xy'],[.4,.6])+.06,.35]
    if not np.isfinite(np.r_[center,size]).all() or np.any(size<=0):raise ValueError('Invalid table extent')
    return center,rotation,size


def corners(center,rotation,size):
    return np.asarray(list(product([-1.,1.],repeat=3)))*np.asarray(size)/2@rotation.T+center


def detections(scene):
    return [scene['measured']['table'],*[row['table'] for row in scene.get('frames',[])]]


def build_envelope(scene):
    reference,rotation,_=table_box(scene['measured']['table'])
    points=np.concatenate([corners(*table_box(table)) for table in detections(scene)])
    local=(points-reference)@rotation
    # Reserve 10mm for subsequent edge estimates, 1mm along the normal.
    # The fit already contains all ten original inflated table observations.
    margin=np.array([.010,.010,.001])
    lo,hi=local.min(axis=0)-margin,local.max(axis=0)+margin
    return dict(schema=1,center=(reference+rotation@((lo+hi)/2)).tolist(),rotation=rotation.tolist(),
                size=(hi-lo).tolist(),uncertainty_margin_m=margin.tolist(),
                source_frames=len(scene.get('frames',[])),base_edge_padding_m=.03,base_top_padding_m=.003)


def admit_scene(plan_scene,fresh_scene):
    envelope=plan_scene.get('table_collision_envelope')
    if envelope is None:
        raise ValueError('Legacy trajectory has no stable table envelope; regenerate with collect without --replay-plan')
    center=np.asarray(envelope['center']);rotation=np.asarray(envelope['rotation']);size=np.asarray(envelope['size'])
    if (center.shape!=(3,) or rotation.shape!=(3,3) or size.shape!=(3,)
            or not np.isfinite(np.r_[center,rotation.ravel(),size]).all() or np.any(size<=0)
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or np.linalg.det(rotation)<.999999):
        raise ValueError('Invalid saved table envelope')
    points=np.concatenate([corners(*table_box(table)) for table in detections(fresh_scene)])
    exceed=np.maximum(abs((points-center)@rotation)-size/2,0.).max(axis=0)
    if max(exceed)>1e-7:
        raise ValueError('Fresh table is outside planned envelope (m): '+str(exceed.tolist())+'; regenerate the full plan before motion')
    result=deepcopy(fresh_scene);result['table_collision_envelope']=deepcopy(envelope)
    return result


def planning_scene(scene):
    result=deepcopy(scene);result['table_collision_envelope']=build_envelope(scene)
    return result


def with_top_clearance(scene,height):
    result=deepcopy(scene)
    envelope=result['table_collision_envelope']
    normal=np.asarray(envelope['rotation'])[:,2]
    envelope['center']=(np.asarray(envelope['center'])+normal*height/2).tolist()
    envelope['size'][2]+=height
    return result


def with_edge_clearance(scene,margin):
    """Reserve lateral tracking room during planning, preserving table height."""
    if not math.isfinite(margin) or margin<0:
        raise ValueError('Table edge clearance must be finite and nonnegative')
    result=deepcopy(scene)
    envelope=result['table_collision_envelope']
    envelope['size'][:2]=(np.asarray(envelope['size'][:2])+2*margin).tolist()
    return result
