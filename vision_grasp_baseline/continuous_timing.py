"""Offline timing of complete expert phases; no ROS or motor access.

TOPPRA sets path speed, a short Gaussian transition removes acceleration
discontinuities, and an exact Bernstein audit bounds the controller's quintic
interpolant. The resulting geometry MUST be collision checked before dispatch.
"""
from copy import deepcopy
from .gripper_settings import (OPEN_POSITION,CLOSE_POSITION,COLLECTION_POLICY,COLLECTION_EVENTS,
                               TRANSIT_POSITION,TRANSIT_SETTLE_S)

import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BPoly, make_interp_spline
from scipy.ndimage import gaussian_filter1d


def controller_bounds(t, q, v, a):
    bounds = [np.zeros(q.shape[1]) for _ in range(3)]
    lower, upper = q.min(axis=0), q.max(axis=0)
    for i in range(len(t)-1):
        poly = BPoly.from_derivatives(t[i:i+2], [[q[i], v[i], a[i]], [q[i+1], v[i+1], a[i+1]]])
        edges=np.linspace(t[i],t[i+1],max(1,int(np.ceil((t[i+1]-t[i])/.05)))+1)
        for left,right in zip(edges,edges[1:]):
            part=BPoly.from_derivatives([left,right],[[poly(x),poly(x,1),poly(x,2)] for x in (left,right)])
            lower = np.minimum(lower, part.c.min(axis=(0, 1)))
            upper = np.maximum(upper, part.c.max(axis=(0, 1)))
            for k in range(1, 4):
                bounds[k-1] = np.maximum(bounds[k-1], abs(part.derivative(k).c).max(axis=(0, 1)))
    return bounds, lower, upper


def path_knots(plans):
    """Discard timing and rest knots; preserve every geometric segment endpoint."""
    rows = []
    for plan in plans:
        points = np.asarray([p['positions'] for p in plan['trajectory']['joint_trajectory']['points']])
        if rows and np.max(abs(rows[-1]-points[0])) > 1e-5:
            raise ValueError('Discontinuous virtual planning start: '+plan['name'])
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distance)>1e-9]
        points, distance = points[keep], distance[keep]
        if len(points)<2:
            continue
        # Geometry sampled independently of the old conservative time scaling.
        grid = np.linspace(0, distance[-1], max(3, int(np.ceil(distance[-1]/.025))+1))
        sampled = np.column_stack([np.interp(grid, distance, points[:,j]) for j in range(7)])
        rows.extend(sampled[1:] if rows else sampled)
    q = np.asarray(rows)
    if len(q)<3 or not np.isfinite(q).all():
        raise ValueError('Invalid complete phase geometry')
    keep = np.r_[True, np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-8]
    return q[keep]


def retime(plans, limits, smoothing_sigma=.12):
    import toppra as ta
    q = path_knots(plans)
    ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    path = ta.SplineInterpolator(ss, q)
    velocity, acceleration, jerk = [np.asarray(limits[k]) for k in ('velocity','acceleration','jerk')]
    if any(x.shape!=(7,) or not np.isfinite(x).all() or (x<=0).any() for x in (velocity,acceleration,jerk)):
        raise ValueError('Missing positive motion limits')
    algorithm = ta.algorithm.TOPPRA([
        ta.constraint.JointVelocityConstraint(velocity),
        ta.constraint.JointAccelerationConstraint(acceleration),
    ], path, gridpoints=np.linspace(0,ss[-1],3001), parametrizer='ParametrizeConstAccel')
    trajectory = algorithm.compute_trajectory(0,0)
    if trajectory is None:
        raise ValueError('Complete phase cannot be time parameterized')
    dt, sigma = .02, smoothing_sigma
    pad=4*sigma
    times = np.arange(0, trajectory.duration+2*pad+dt, dt)
    values = gaussian_filter1d(trajectory(np.clip(times-pad,0,trajectory.duration)),
                               sigma/dt,axis=0,mode='nearest',truncate=4)
    u = times/times[-1]; h = 10*u**3-15*u**4+6*u**5
    values += (1-h)[:,None]*(q[0]-values[0])+h[:,None]*(q[-1]-values[-1])
    # The upstream LIN path can touch a planning position bound exactly.
    # Smoothly round it inside the bound before full path revalidation;
    # never clamp an already running command stream.
    unclipped=values.copy()
    width=.002
    lower_cap=np.asarray(limits['lower'])+.0002
    upper_cap=np.asarray(limits['upper'])-.0002
    values=lower_cap+width*np.logaddexp(0.,(values-lower_cap)/width)
    values=upper_cap-width*np.logaddexp(0.,(upper_cap-values)/width)
    projection=float(np.max(abs(values-unclipped)))
    if projection>.005:raise ValueError('Smoothing needs excessive joint-limit projection')
    # Restore endpoints over the whole phase with a C2 correction. Replacing
    # just the endpoint after soft projection creates a tiny 20ms kink whose
    # artificial jerk used to stretch otherwise normal phases by 3x or more.
    values+=(1-h)[:,None]*(q[0]-values[0])+h[:,None]*(q[-1]-values[-1])
    values[0],values[-1]=q[0],q[-1]
    bc=[(1,np.zeros(7)),(2,np.zeros(7))]
    spline=make_interp_spline(times,values,k=5,bc_type=(bc,bc))
    positions, velocities, accelerations = [spline(times,k) for k in range(3)]
    velocities[[0,-1]]=0.;accelerations[[0,-1]]=0.
    bounds,lower,upper=controller_bounds(times,positions,velocities,accelerations)
    scale=max(1.,float(max(bounds[0]/velocity)),float(np.sqrt(max(bounds[1]/acceleration))),
              float(np.cbrt(max(bounds[2]/jerk))))*1.002
    times*=scale;velocities/=scale;accelerations/=scale**2
    if times[-1]>180 or not np.isfinite(times).all():
        raise ValueError('Unreasonable timed phase duration')
    # Bernstein convex hull also bounds positions BETWEEN controller knots.
    if np.any(lower<np.asarray(limits['lower'])-1e-7) or np.any(upper>np.asarray(limits['upper'])+1e-7):
        raise ValueError('Smoothed phase exceeds joint position bounds: '+str(dict(stages=[p['name'] for p in plans],
                         lower=lower.tolist(),upper=upper.tolist(),limits=limits)))
    return dict(times=times.tolist(),positions=positions.tolist(),velocities=velocities.tolist(),
                accelerations=accelerations.tolist(),duration_s=float(times[-1]),
                audit=dict(method='TOPPRA + C2 quintic; Bernstein bounds of JTC interpolation',
                           velocity_bound=(bounds[0]/scale).tolist(),acceleration_bound=(bounds[1]/scale**2).tolist(),
                           jerk_bound=(bounds[2]/scale**3).tolist(),time_scale_for_jerk=scale,
                           joint_limit_projection_rad=projection,
                           transition_sigma_s=sigma,
                           source_stages=[p['name'] for p in plans]))


def gripper_duration(start,end,limits):
    # Exact extrema for rest-to-rest quintic: v=1.875*d/T,
    # a=(10/sqrt(3))*d/T^2, j=60*d/T^3.
    distance=abs(end-start)
    return max(.2,1.875*distance/limits['velocity'],
               np.sqrt(10/np.sqrt(3)*distance/limits['acceleration']),
               np.cbrt(60*distance/limits['jerk']))*1.01


def assemble(phases,gripper_limits,gripper_policy='legacy_open_transit_v1'):
    if gripper_policy==COLLECTION_POLICY:
        return assemble_closed_transit(phases,gripper_limits)
    if gripper_policy!='legacy_open_transit_v1' or len(phases)!=3:
        raise ValueError('Unknown gripper policy/phase layout')
    rows=[];events=[];offset=0.;finger=OPEN_POSITION
    for index,phase in enumerate(phases):
        events.append(dict(name=('reach','carry','return')[index],time_s=offset))
        for i,(t,q,v,a) in enumerate(zip(phase['times'],phase['positions'],phase['velocities'],phase['accelerations'])):
            if rows and i==0:continue
            rows.append(dict(time_s=offset+t,positions=q+[finger],velocities=v+[0.],accelerations=a+[0.]))
        offset=rows[-1]['time_s']
        if index<2:
            target=CLOSE_POSITION if index==0 else OPEN_POSITION
            name='close' if index==0 else 'open'
            events.append(dict(name=name,time_s=offset))
            offset+=gripper_duration(finger,target,gripper_limits)
            row=deepcopy(rows[-1]);row.update(time_s=offset)
            row['positions'][-1]=target;rows.append(row);finger=target
            # A physical grasp/drop settling interval, not software planning wait.
            offset+=.15;row=deepcopy(rows[-1]);row['time_s']=offset;rows.append(row)
    return dict(samples=rows,events=events,duration_s=offset)


def assemble_closed_transit(phases,gripper_limits):
    if len(phases)!=5:
        raise ValueError('Closed transit requires pregrasp/approach/carry/retreat/zero phases')
    rows=[];events=[];offset=0.;finger=TRANSIT_POSITION
    names=('reach','approach','carry','return','return_closed')
    transitions=(('pregrasp_open',OPEN_POSITION),('close',CLOSE_POSITION),
                 ('open',OPEN_POSITION),('transit_close',TRANSIT_POSITION))
    for index,phase in enumerate(phases):
        events.append(dict(name=names[index],time_s=offset))
        for i,(t,q,v,a) in enumerate(zip(phase['times'],phase['positions'],phase['velocities'],phase['accelerations'])):
            if rows and i==0:
                if np.max(abs(np.asarray(q)-rows[-1]['positions'][:7]))>1e-5:
                    raise ValueError('Discontinuous closed-transit arm phase')
                continue
            rows.append(dict(time_s=offset+t,positions=q+[finger],velocities=v+[0.],accelerations=a+[0.]))
        offset=rows[-1]['time_s']
        if index<4:
            name,target=transitions[index]
            events.append(dict(name=name,time_s=offset))
            offset+=gripper_duration(finger,target,gripper_limits)
            row=deepcopy(rows[-1]);row.update(time_s=offset)
            row['positions'][-1]=target;rows.append(row);finger=target
            offset+=TRANSIT_SETTLE_S if name=='transit_close' else .15
            row=deepcopy(rows[-1]);row['time_s']=offset;rows.append(row)
    assert tuple(e['name'] for e in events)==COLLECTION_EVENTS
    return dict(samples=rows,events=events,duration_s=offset,gripper_policy=COLLECTION_POLICY)


def main():
    source=json.loads(Path(sys.argv[1]).read_text())
    phases=[retime(p,source['limits'],source.get('smoothing_sigma_s',.12)) for p in source['phase_plans']]
    result=dict(phases=phases,**assemble(phases,source['gripper_limits'],
                                       source.get('gripper_policy','legacy_open_transit_v1')))
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False)+'\n')


if __name__=='__main__':main()
