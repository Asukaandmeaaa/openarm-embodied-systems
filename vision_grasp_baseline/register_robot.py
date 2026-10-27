"""Diagnostic translation-only ICP, with symmetric residuals and holdout poses.

It never edits calibration and never sends a trajectory. A model/depth fit
is evidence about alignment, not proof of robot-base displacement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def voxel(points,size=.002):
    _,indices=np.unique(np.floor(points/size).astype('int64'),axis=0,return_index=True)
    return points[indices]


def residual(source,target,delta):
    forward=cKDTree(target).query(source+delta)[0]
    reverse=cKDTree(source+delta).query(target)[0]
    return dict(model_to_observed_median_m=float(np.median(forward)),
                observed_to_model_median_m=float(np.median(reverse)),
                model_to_observed_p90_m=float(np.percentile(forward,90)),
                observed_to_model_p90_m=float(np.percentile(reverse,90)))


def fit_translation(source,target):
    tree=cKDTree(target);delta=np.zeros(3)
    for _ in range(80):
        distances,ids=tree.query(source+delta)
        keep=distances<min(.04,np.percentile(distances,65))
        if keep.sum()<200:raise ValueError('Insufficient surface correspondences')
        update=np.median(target[ids[keep]]-source[keep],axis=0)-delta
        delta+=update
        if np.linalg.norm(update)<1e-5:break
    return delta


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows=[];clouds=[]
    for p in a.runs:
        data=np.load(p/'projection_clouds.npz');source=voxel(data['model_camera']);target=voxel(data['observed_camera'])
        delta=fit_translation(source,target)
        wc=np.array(json.loads((p/'plan.json').read_text())['scene']['world_from_color_optical'])
        rows.append(dict(run=str(p),translation_model_to_observation_camera_m=delta.tolist(),
                         translation_in_old_world_m=(wc[:3,:3]@delta).tolist(),
                         before=residual(source,target,np.zeros(3)),after=residual(source,target,delta)))
        clouds.append((source,target,delta))
    for row,(source,target,_) in zip(rows,clouds):
        row['cross_pose_residuals']=[residual(source,target,delta) for _,_,delta in clouds]
    out=dict(diagnostic_only=True,calibration_changed=False,hardware_motion=False,poses=rows,
             limitation='Translation-only collision-mesh ICP; partial views, CAD and depth bias remain confounders.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
