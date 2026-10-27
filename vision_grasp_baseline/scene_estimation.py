"""Robust fixed-object estimates; retain raw frames and reject moving batches."""
from copy import deepcopy
import math

import numpy as np

from .collection_io import check_fixed_scene


class UnstableSceneError(ValueError):
    """A fresh stationary batch may recover; never accept this batch for motion."""


def angle_difference(a,b,period):
    return (np.asarray(a)-b+period/2)%period-period/2


def circular_median(values,period):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Non-finite detected orientation')
    differences=angle_difference(values[:,None],values[None,:],period)
    anchor=values[np.argmin(np.sum(abs(differences),axis=0))]
    return float(angle_difference(anchor+np.median(angle_difference(values,anchor,period)),0.,period))


def stabilize_scene(scene,*,stationary=None):
    """Use a consensus, including recent frames, instead of the last noisy pose.

    Table observations are deliberately untouched: the conservative table
    envelope must continue to contain every original observation.
    """
    if stationary is None:
        stationary=scene.get('pose_estimation',{}).get('stationary_fixture',False)
    rows=scene.get('frames',[])
    if len(rows)<5:raise ValueError('At least 5 RGB-D frames required for stable fixture estimation')
    result=deepcopy(scene)
    # Perception initially aliases measured to frames[-1]. deepcopy preserves
    # that alias: replacing measured objects would otherwise overwrite a raw
    # frame and change the consensus on the next validation pass.
    result['measured']=deepcopy(scene['measured'])
    quality={}
    for name,center_key,period in [('cube','center',math.pi/2),('bin','center_xy',math.pi)]:
        objects=[row[name] for row in rows]
        centers=np.asarray([obj[center_key] for obj in objects],dtype=float)
        if not np.isfinite(centers).all():raise ValueError('Non-finite detected '+name+' position')
        center=np.median(centers,axis=0)
        angles=np.asarray([obj['yaw'] for obj in objects])
        yaw=circular_median(angles,period)
        residual=abs(angle_difference(angles,yaw,period))
        inliers=(residual<=.08)&(np.linalg.norm(centers-center,axis=1)<=.005)
        # A recent sustained change is not an outlier, even if the older frames
        # are still the majority. One noisy final frame alone is tolerated.
        if inliers.sum()<math.ceil(.7*len(rows)) or (not stationary and inliers[-3:].sum()<2):
            raise UnstableSceneError(f'Unstable {name} detection: {inliers.sum()}/{len(rows)} agreeing frames; '
                             f'latest 3: {inliers[-3:].sum()}/3. Keep fixture still and recapture')
        estimated=deepcopy(objects[-1])
        for key in estimated:
            values=np.asarray([obj[key] for obj in objects],dtype=float)
            if not np.isfinite(values).all():raise ValueError('Non-finite detected '+name+' '+key)
            estimated[key]=np.median(values[inliers],axis=0).tolist()
        estimated['yaw']=circular_median(angles[inliers],period)
        result['measured'][name]=estimated
        quality[name]=dict(inliers=int(inliers.sum()),frames=len(rows),
                           yaw_rad=estimated['yaw'],yaw_max_residual_rad=float(residual.max()),
                           latest_inliers=int(inliers[-3:].sum()))
    result['pose_estimation']=dict(method='temporal_consensus_median_v1',objects=quality)
    if stationary:result['pose_estimation']['stationary_fixture']=True
    return result


def check_replay_fixture(scene,plan_scene):
    fresh=stabilize_scene(scene)
    check_fixed_scene(fresh,plan_scene,cube_tolerance=.005,bin_tolerance=.005,yaw_tolerance=.08)
    delta=abs(float(angle_difference(fresh['measured']['cube']['yaw'],plan_scene['measured']['cube']['yaw'],math.pi/2)))
    if delta>.08:
        raise ValueError(f'Cube orientation changed by {math.degrees(delta):.2f} deg across stable frames; regenerate complete plan')
    return fresh
