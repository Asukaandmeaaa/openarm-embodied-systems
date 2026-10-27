#!/usr/bin/env python3
"""Offline only: derived candidates for manual replay, never imports ROS nodes.

TOPPRA retimes a lightly smoothed path. A finite-support Gaussian convolution
and a C4 quintic spline smooth its acceleration transitions. Exact polynomial
extrema are used to stretch time to existing velocity/acceleration/jerk limits.
The runtime still sends positions only: these bounds describe the reference,
not a claim about controller interpolation or measured robot acceleration.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.interpolate import make_smoothing_spline, make_interp_spline, PPoly
from scipy.ndimage import gaussian_filter1d
import toppra as ta
import yaml

WS = Path('/home/lushilin/openarm_vla_ws')
NATIVE = Path('/home/lushilin/openarm_kinesthetic_ws/ros2_ws/src/openarm_kinesthetic_recorder')
sys.path.insert(0, str(NATIVE))
from openarm_kinesthetic_recorder.recorded_motion import validate_collection_trajectory
LIMITS = Path('/home/lushilin/openarm_ws/src/openarm_ros2/openarm_bimanual_moveit_config/config/openarm_v2.0/joint_limits.yaml')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extrema(spline, order):
    """Exact extrema of each polynomial piece, including both sides of knots."""
    lo, hi = [], []
    for j in range(7):
        pp = PPoly.from_spline((spline.t, spline.c[:, j], spline.k)).derivative(order)
        roots = pp.derivative().roots(extrapolate=False)
        x = np.r_[pp.x, roots[np.isfinite(roots)]]
        x = x[(x >= spline.t[spline.k]) & (x <= spline.t[-spline.k-1])]
        y = pp(x)
        lo.append(float(np.min(y))); hi.append(float(np.max(y)))
    return np.array(lo), np.array(hi)


def maxima(spline, order):
    lo, hi = extrema(spline, order)
    return np.maximum(abs(lo), abs(hi))


def path_distance(queries, path):
    delta=np.diff(path,axis=0); denominator=np.maximum(np.sum(delta*delta,axis=1),1e-30)
    largest=0.
    for start in range(0,len(queries),128):
        relative=queries[start:start+128,None,:]-path[None,:-1,:]
        u=np.clip(np.einsum('ijk,jk->ij',relative,delta)/denominator[None,:],0,1)
        errors=np.max(abs(relative-u[:,:,None]*delta[None,:,:]),axis=2)
        largest=max(largest,float(np.max(np.min(errors,axis=1))))
    return largest


def path_parameter(tr, query):
    # Reconstruct constant-acceleration scalar retiming from public problem data.
    ss, speed = tr.problem_data.gridpoints, tr.problem_data.sd_vec
    dt = 2*np.diff(ss)/(speed[:-1]+speed[1:])
    ts = np.r_[0, np.cumsum(dt)]
    acc = np.diff(speed**2)/(2*np.diff(ss))
    idx = np.clip(np.searchsorted(ts, query, side='right')-1, 0, len(acc)-1)
    elapsed = query-ts[idx]
    return np.clip(ss[idx]+speed[idx]*elapsed+.5*acc[idx]*elapsed**2, ss[0], ss[-1])


def smooth_segment(t, q, velocity, acceleration, jerk, *, preserve_duration):
    """Keep both endpoints exactly; reject excessive reference-path deformation."""
    duration = float(t[-1])
    for lam in [1e-3, 3e-4, 1e-4, 3e-5]:
        u = t/duration
        h = 10*u**3-15*u**4+6*u**5
        fit = np.column_stack([make_smoothing_spline(t, q[:, j], lam=lam)(t) for j in range(7)])
        fit += (1-h)[:, None]*(q[0]-fit[0])+h[:, None]*(q[-1]-fit[-1])
        fit[0], fit[-1] = q[0], q[-1]
        # Drop near-duplicate geometry, not endpoints. Otherwise long pauses
        # make path-speed constraints numerically ill-conditioned.
        keep = [0]
        for i in range(1, len(fit)-1):
            if np.max(abs(fit[i]-fit[keep[-1]])) > 5e-4:
                keep.append(i)
        if len(keep)>1 and np.max(abs(fit[-1]-fit[keep[-1]])) < 5e-4:
            keep.pop()
        keep.append(len(fit)-1)
        knots = fit[keep]
        ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
        if np.any(np.diff(ss)<=0):
            raise ValueError('stationary/degenerate segment requires separate handling')
        path = ta.SplineInterpolator(ss, knots)
        algorithm = ta.algorithm.TOPPRA([
            ta.constraint.JointVelocityConstraint(velocity),
            ta.constraint.JointAccelerationConstraint(acceleration),
        ], path, gridpoints=np.linspace(0, ss[-1], 3001), parametrizer='ParametrizeConstAccel')
        trajectory = algorithm.compute_trajectory(0, 0)
        if trajectory is None:
            continue
        dt, pad, sigma = .01, .4, .1
        times = np.arange(0, trajectory.duration+2*pad+dt, dt)
        clipped = np.clip(times-pad, 0, trajectory.duration)
        values = gaussian_filter1d(trajectory(clipped), sigma/dt, axis=0, mode='nearest', truncate=4)
        reference_t = np.interp(path_parameter(algorithm, clipped), ss, t[keep])
        reference_t = gaussian_filter1d(reference_t, sigma/dt, mode='nearest', truncate=4)
        reference_t[0], reference_t[-1] = 0, duration
        u = times/times[-1]; h = 10*u**3-15*u**4+6*u**5
        values += (1-h)[:, None]*(q[0]-values[0])+h[:, None]*(q[-1]-values[-1])
        values[0], values[-1] = q[0], q[-1]
        bc = [(1, np.zeros(7)), (2, np.zeros(7))]
        spline = make_interp_spline(times, values, k=5, bc_type=(bc, bc))
        v, a, j = [maxima(spline, k) for k in (1, 2, 3)]
        scale = max(1., float(max(v/velocity)), float(np.sqrt(max(a/acceleration))), float(np.cbrt(max(j/jerk))))
        scale *= 1.001  # numerical margin by slowing, never relaxing a bound
        if preserve_duration:
            scale = max(scale, duration/times[-1])
        final_duration = float(times[-1]*scale)
        out_t = np.linspace(0, final_duration, int(np.ceil(final_duration/.02))+1)
        out_q = spline(out_t/scale)
        out_q[0], out_q[-1] = q[0], q[-1]
        source_t = np.interp(out_t/scale, times, reference_t)
        deviation = max(path_distance(out_q,q),path_distance(q,out_q))
        if deviation > .01:
            continue  # processing fidelity budget, not a robot safety threshold
        return out_t, out_q, source_t, {
            'original_duration_s': duration, 'processed_duration_s': final_duration,
            'lambda': lam, 'max_reference_difference_rad': deviation,
            'reference_difference_metric':'bidirectional nearest polyline distance in joint max norm',
            'max_velocity_rad_s': (v/scale).tolist(),
            'max_acceleration_rad_s2': (a/scale**2).tolist(),
            'max_jerk_rad_s3': (j/scale**3).tolist(),
            'position_min_rad': extrema(spline, 0)[0].tolist(),
            'position_max_rad': extrema(spline, 0)[1].tolist(),
            'endpoint_velocity_max': float(np.max(abs(spline([0,times[-1]],1)))/scale),
            'endpoint_acceleration_max': float(np.max(abs(spline([0,times[-1]],2)))/scale**2),
        }
    raise ValueError('no candidate meets smoothing/reference fidelity checks')


def process(source, limits):
    original = json.loads(source.read_text())
    validate_collection_trajectory(original)
    t = np.array([s['time_from_start_s'] for s in original['samples']])
    q = np.array([s['positions_rad'] for s in original['samples']])
    names = original['joint_names'][:7]
    v = np.array([limits[n]['max_velocity'] for n in names])
    a = np.array([limits[n]['max_acceleration'] for n in names])
    j = np.array([limits[n]['max_jerk'] for n in names])
    if not all(limits[n].get('has_acceleration_limits') and limits[n].get('has_jerk_limits') for n in names):
        raise ValueError('expected enabled acceleration and jerk limits')
    # Do not exceed the source trajectory's per-joint peak velocity either.
    v = np.minimum(v, np.max(abs(np.diff(q[:,:7],axis=0)/np.diff(t)[:,None]),axis=0))
    if np.any(v<=0):
        raise ValueError('zero-motion joint needs separate retiming handling')
    indices = [0]+[m['sample_index'] for m in original['markers']]+[len(t)-1]
    result = copy.deepcopy(original)
    result['samples'], result['markers'] = [], []
    segments, offset = [], 0.
    for seg, (start, end) in enumerate(zip(indices, indices[1:])):
        st, sq, ref, report = smooth_segment(t[start:end+1]-t[start], q[start:end+1,:7], v,a,j,preserve_duration=seg<2)
        # Existing planning position bounds, if specified, also apply offline.
        for k, name in enumerate(names):
            cfg = limits[name]
            if cfg.get('has_position_limits') and (report['position_min_rad'][k]<cfg['min_position'] or report['position_max_rad'][k]>cfg['max_position']):
                raise ValueError(f'{name} outside configured position limits')
        segments.append(report)
        for i in range(0 if seg==0 else 1, len(st)):
            elapsed = float(offset+st[i]); raw_time = float(t[start]+ref[i])
            gripper = float(np.interp(raw_time,t,q[:,7]))
            result['samples'].append({
                'sample_index': len(result['samples']), 'time_from_start_s': elapsed,
                'positions_rad': sq[i].tolist()+[gripper],
                # Compatibility field is explicitly synthetic; original clocks
                # and sample quality remain in immutable source/provenance.
                'source_timestamp_s': original['samples'][0]['source_timestamp_s']+elapsed,
                'original_time_from_start_s': raw_time,
            })
        offset += float(st[-1])
        if seg<2:
            marker = copy.deepcopy(original['markers'][seg])
            marker.update(sample_index=len(result['samples'])-1,time_from_start_s=offset,
                          source_timestamp_s=result['samples'][-1]['source_timestamp_s'],binding_error_s=0.)
            result['markers'].append(marker)
    result['duration_s'] = offset
    result['sample_period_target_s'] = .02
    gaps = np.diff([s['time_from_start_s'] for s in result['samples']])
    result['quality'] = {key:0 for key in ['rejected_incomplete_count','rejected_nonfinite_count','rejected_invalid_timestamp_count','timestamp_regression_count','estimated_dropped_sample_count']}
    result['quality'].update(accepted_sample_count=len(result['samples']), maximum_sample_gap_s=float(max(gaps)), scope='generated_reference_only_not_sensor_quality')
    result['processing'] = {
        'version':1, 'method':'bounded_smoothing_toppra_gaussian_quintic',
        'source_path': str(source), 'source_sha256':sha(source),
        'source_trial_id':original['trial_id'], 'source_quality':original['quality'],
        'source_markers':original['markers'], 'source_duration_s':original['duration_s'],
        'timestamp_semantics':'source_timestamp_s is synthetic replay time anchored to original start; original_time_from_start_s maps to immutable source',
        'limits_path':str(LIMITS), 'limits_sha256':sha(LIMITS),
        'geometry_min_step_rad':.0005, 'gaussian_sigma_s':.1, 'fit_grid_period_s':.01, 'output_period_max_s':.02,
        'reference_fidelity_budget_rad':.01, 'segments':segments,
        'controller_interface':'positions_only; derivative bounds are for offline smooth reference',
        'physical_replay_status':'pending', 'collision_check_status':'requires_live_scene_plan_only',
    }
    # Invariants: all arm endpoints/markers, semantic events, raw source IDs.
    for old_idx,new_idx in zip(indices, [0]+[m['sample_index'] for m in result['markers']]+[len(result['samples'])-1]):
        assert result['samples'][new_idx]['positions_rad'] == original['samples'][old_idx]['positions_rad']
    validate_collection_trajectory(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',type=Path,default=WS/'openarm_kinesthetic_data/collections_real14_20261016/notes/replay14_files.txt')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing candidate batch')
    paths=[Path(s).resolve(strict=True) for s in args.list.read_text().splitlines() if s.strip()]
    limits=yaml.safe_load(LIMITS.read_text())['joint_limits']
    # Complete all processing before publishing a usable selection list.
    results=[]; blocked=[]
    for source in paths:
        try:
            value=process(source,limits)
        except ValueError as exc:
            blocked.append(dict(source_path=str(source),source_sha256=sha(source),reason=str(exc)))
            print(source.name, 'BLOCKED:', exc, flush=True)
            continue
        results.append((source,value))
        print(source.name, f"{value['processing']['source_duration_s']:.2f} -> {value['duration_s']:.2f} s",flush=True)
    output=args.output.resolve(); (output/'trajectories').mkdir(parents=True)
    (output/'notes').mkdir(); files=[]; rows=[]
    for source,value in results:
        dest=output/'trajectories'/source.name
        dest.write_text(json.dumps(value,indent=2)+'\n'); files.append(dest)
        rows.append(dict(file=str(dest),sha256=sha(dest),source_sha256=sha(source),processing=value['processing']))
    (output/'notes/replay14_files.txt').write_text(''.join(str(p)+'\n' for p in files))
    (output/'notes/replay14_files.sha256').write_text(''.join(sha(p)+'  '+str(p)+'\n' for p in files))
    (output/'notes/processing_report.json').write_text(json.dumps({'script_sha256':sha(__file__),'numpy':np.__version__,'scipy':scipy.__version__,'toppra':'0.6.10','rows':rows,'blocked':blocked},indent=2)+'\n')

if __name__=='__main__': main()
