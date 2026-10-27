"""Collision validation and Dora replay in MuJoCo, with no hardware node."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from core import NAMES, check_start, dense_samples, digest, validate_bundle, write_json
from validation_contract import aligned_pose, replay_arrays, ros_limits, simulation_output

ROOT = Path(__file__).resolve().parents[2]
KIN = ROOT.parent/'openarm_kinesthetic_ws'


def model_hashes(kin):
    paths = [kin/'sim/scenes/fixed_home_guard_v1.xml']
    paths += sorted((kin/'vendor/openarm_mujoco/v2').rglob('*.xml'))
    paths += sorted((kin/'vendor/openarm_mujoco/v2/assets').rglob('*.stl'))
    return {str(p.relative_to(kin)): digest(p) for p in paths}


class Scene:
    def __init__(self, bundle, kin=KIN, *, position_only=False):
        import mujoco
        self.mujoco = mujoco
        # ACT needs positions/contact geometry, not acceleration or constraint
        # forces. Keep the legacy full-forward path as the default/reference.
        self.position_only = position_only
        self.model = mujoco.MjModel.from_xml_path(str(kin/'sim/scenes/fixed_home_guard_v1.xml'))
        self.data = mujoco.MjData(self.model)
        collision = bundle['home_config']['static_collision']
        if collision['frame_id'] != 'world':
            raise ValueError('table frame must be world')
        center = np.array(collision['center_xyz_m'], dtype=float)
        size = np.array(collision['dimensions_xyz_m'], dtype=float)
        quat = np.array(collision['orientation_xyzw'], dtype=float)
        if (center.shape != (3,) or size.shape != (3,) or quat.shape != (4,)
                or not np.all(np.isfinite(np.r_[center,size,quat]))
                or np.any(size <= 0) or abs(np.linalg.norm(quat)-1)>1e-6):
            raise ValueError('invalid table geometry')
        body = self.model.body('fixed_inflated_table_guard').id
        self.model.body_pos[body] = center
        self.model.body_quat[body] = quat[[3,0,1,2]]
        self.model.geom_size[self.model.geom('table_top_collision').id] = size/2
        self.indices = [int(self.model.joint(name).qposadr[0]) for name in NAMES]
        self.mimic = [int(self.model.joint(f'openarm_{side}_finger_joint2').qposadr[0])
                      for side in ('right','left')]
        self.joint_ids = [self.model.joint(name).id for name in NAMES]
        sys.path.insert(0, str(kin/'src'))
        from openarm_kinesthetic.initialization.collision import penetrating_contact_records
        self.contacts = penetrating_contact_records

    def put(self, q):
        self.data.qpos[self.indices] = q
        self.data.qpos[self.mimic] = q[[7,15]]
        self.data.qvel[:] = 0
        if self.position_only:
            self.mujoco.mj_fwdPosition(self.model, self.data)
        else:
            self.mujoco.mj_forward(self.model, self.data)
        # Existing collision policy excludes only coupled finger/internal pairs;
        # includes table, pedestal, inter-arm and non-adjacent self collisions.
        return self.contacts(self.model, self.data, tolerance_m=0.0)


def validate(path, output, kin=KIN, live_start=None, scope='all'):
    bundle = json.loads(Path(path).read_text())
    t, q, active = replay_arrays(bundle, scope)
    scene = Scene(bundle, kin)
    failures = []
    try:
        check_start(bundle, bundle['measured_start_rad'], active_indices=active)
        start_alignment = {'passed': True, 'tolerance_rad': .03}
    except ValueError as exc:
        start_alignment = {'passed': False, 'tolerance_rad': .03, 'error': str(exc)}
        failures.append('start_mismatch')
    if live_start is not None:
        check_start(bundle, live_start, active_indices=active)
        # Left context still has to match when a current scene state is supplied.
        if scope == 'right' and np.max(np.abs(np.asarray(live_start)[8:]-q[0, 8:])) > .03:
            raise ValueError('left collision context start mismatch')
    model_limits = scene.model.jnt_range[scene.joint_ids]
    driver = bundle['driver_config']
    driver_limits = np.array(driver['joint_limits']['right_arm']+driver['joint_limits']['left_arm'])
    legacy_limits = np.c_[np.maximum(model_limits[:,0], driver_limits[:,0]),
                          np.minimum(model_limits[:,1], driver_limits[:,1])]
    limits, ros_velocity, sources, limit_policy = ros_limits(
        bundle, legacy_limits, driver['joint_velocity_limits']*2, required=scope == 'right')
    lower, upper = limits.T
    # One CAN position quantization bin, not a relaxed physical joint limit.
    quantization = 25/65535
    limit_mask = (q < lower-quantization) | (q > upper+quantization)
    limit_violations = []
    for i in np.flatnonzero(np.any(limit_mask, axis=0)):
        indices = np.flatnonzero(limit_mask[:, i])
        limit_violations.append({
            'joint_name': NAMES[i],
            'recorded_range_rad': [float(q[:, i].min()), float(q[:, i].max())],
            'model_limits_rad': model_limits[i].tolist(),
            'driver_limits_rad': driver_limits[i].tolist(),
            'effective_limits_rad': [float(lower[i]), float(upper[i])],
            'role': 'replayed_joint' if i in active else 'observed_collision_context_only',
            'blocks_simulation': bool(i in active),
            'violating_samples': len(indices),
            'first_time_s': float(t[indices[0]]), 'last_time_s': float(t[indices[-1]]),
        })
    peak_velocity = np.max(np.abs(np.diff(q, axis=0)/np.diff(t)[:,None]), axis=0)
    velocity_limits = np.minimum(ros_velocity, np.array(driver['joint_velocity_limits']*2))
    if any(v['blocks_simulation'] for v in limit_violations):
        failures.append('joint_limits')
    if np.any(peak_velocity[active] > velocity_limits[active]):
        failures.append('interpolated_velocity_limits')
    contacts, minimum, count = [], 0., 0
    traces = []
    scene_bundle = dict(bundle, positions_rad=q.tolist())
    for stamp, pose in dense_samples(scene_bundle):
        count += 1
        if count > 1000000:
            raise ValueError('validation exceeds one million samples')
        hits = scene.put(pose)
        if hits:
            minimum = min(minimum, min(h['distance_m'] for h in hits))
            if len(contacts) < 100:
                contacts.append({'time_s': stamp, 'contacts': hits})
        if count % 10 == 1 or stamp == t[-1]:
            traces.append([stamp, *scene.data.xpos[scene.model.body('openarm_right_ee_base_link').id]])
    if contacts:
        failures.append('collision')
    measured_end = np.array(bundle['measured_end_rad'])
    final_error = float(np.max(np.abs(q[-1, active]-measured_end[active])))
    if final_error > .03:
        failures.append('recorded_final_tracking_error')
    h2 = bundle.get('h2_execution')
    if bundle.get('phase') != 'H2' or not h2 or not h2.get('success'):
        failures.append('no_successful_H2_provenance')
    sources.update({str(p): digest(p) for p in
                    [Path(__file__), Path(__file__).with_name('core.py'),
                     Path(__file__).with_name('validation_contract.py'),
                     kin/'src/openarm_kinesthetic/initialization/collision.py']})
    report = {'schema': 'openarm_home_sim_validation_v2', 'trajectory_sha256': digest(path),
              'geometric_checks_passed': not failures, 'failures': failures,
              'scope': scope, 'replayed_joint_names': [NAMES[i] for i in active],
              'left_role': 'observed_collision_context_no_command_output' if scope == 'right' else 'replayed',
              'start_alignment': start_alignment,
              'start_tolerance_basis': 'conservative replay-entry guard; not a ROS gripper tracking tolerance',
              'physical_replay_authorized': False,
              'physical_blockers': ['Fresh real starting-state check has not been performed.',
                                    'Direct-CAN gravity/control equivalence is not implemented by this simulation.',
                                    'Fresh physical scene check remains required; operator confirmed recorded HOME previously.',
                                    'Physical graph actively holds the left arm; simulation-only scope does not authorize that graph.',
                                    'Dynamic tracking and stopping behavior are not certified by a kinematic replay.'],
              'duration_s': float(t[-1]), 'original_samples': len(t), 'checked_samples': count,
              'interpolation': bundle['interpolation'], 'max_dt_s': .005, 'max_dq_rad': .002,
              'collision_policy': 'existing initialization pair exclusions; reject any negative contact distance',
              'joint_limit_violations': limit_violations,
              'joint_limit_policy': limit_policy,
              'effective_position_limits_rad': limits.tolist(),
              'effective_velocity_limits_rad_s': velocity_limits.tolist(),
              'validation_source_hashes': sources,
              'model_driver_limit_differences': [
                  {'joint_name': NAMES[i], 'ros_effective_limits_rad': limits[i].tolist(),
                   'model_limits_rad': model_limits[i].tolist(), 'driver_limits_rad': driver_limits[i].tolist()}
                  for i in range(16) if not (np.allclose(limits[i], model_limits[i]) and
                                             np.allclose(limits[i], driver_limits[i]))],
              'minimum_contact_distance_m': minimum, 'first_100_collision_samples': contacts,
              'peak_interpolated_velocity_rad_s': peak_velocity.tolist(),
              'final_command_measured_error_rad': final_error,
              'post_H2_comparison': bundle.get('post_H2_comparison'),
              'effective_table': bundle['home_config']['static_collision'],
              'model_hashes': model_hashes(kin),
              'limitations': ['Kinematic MuJoCo test, not a dynamic or hardware replay.',
                              'Sampled sweep at <=5 ms and <=0.002 rad, not continuous collision proof.',
                              'Scene contains table/robot/pedestal/floor; added physical obstacles are not modeled.']}
    write_json(output, report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11,8))
    for i in range(8):
        axes[0].plot(t, q[:,i], label=f'J{i+1}')
    axes[0].set(xlabel='Recorded time (s)', ylabel='Right joint position (rad)')
    axes[0].legend(ncol=8)
    trace = np.array(traces)
    axes[1].plot(trace[:,1],trace[:,3])
    axes[1].set(xlabel='World x (m)', ylabel='Right EE-base z (m)', title='Recorded path projection')
    fig.suptitle(f'Kinematic replay: {"PASS" if not failures else "FAIL"}; physical authorization remains false')
    fig.tight_layout()
    fig.savefig(str(output)+'.png', dpi=140)
    plt.close(fig)
    return report


def run_dora(path, report_path, log_path, kin=KIN, scope='all'):
    """Timer-driven, real-time kinematic replay in a single simulator process."""
    import dora
    import pyarrow as pa
    import time
    bundle = json.loads(Path(path).read_text())
    report = json.loads(Path(report_path).read_text())
    if (report.get('schema') != 'openarm_home_sim_validation_v2'
            or not report.get('geometric_checks_passed')
            or report.get('scope') != scope
            or report['trajectory_sha256'] != digest(path)
            or report['model_hashes'] != model_hashes(kin)
            or not report.get('validation_source_hashes')
            or any(digest(p) != sha for p, sha in report['validation_source_hashes'].items())):
        raise ValueError('matching successful geometric validation required before Dora simulation')
    t, q, active = replay_arrays(bundle, scope)
    scene = Scene(bundle, kin)
    # Set simulated start explicitly; this never represents a live hardware check.
    simulated_start = np.array(bundle['measured_start_rad'])
    check_start(bundle, simulated_start, active_indices=active)
    if scene.put(simulated_start):
        raise ValueError('collision in simulated starting state')
    node = dora.Node()
    started = None
    last_elapsed = 0.
    with Path(log_path).open('x') as stream:
        for event in node:
            if event['type'] == 'STOP':
                break
            if event['type'] != 'INPUT' or event['id'] != 'tick':
                continue
            now = time.monotonic()
            if started is None:
                started = now
            elapsed = min(now-started, float(t[-1]))
            # Validate the sweep over every crossed original knot, including a
            # delayed tick; checking only the newly displayed pose misses hazards.
            knots = np.r_[last_elapsed, t[(t>last_elapsed)&(t<elapsed)], elapsed]
            for s in knots:
                pose = aligned_pose(t, q, s)
                if scene.put(pose):
                    raise RuntimeError('collision during Dora simulation')
            last_elapsed = elapsed
            output_name, output_pose = simulation_output(pose, scope)
            stream.write(json.dumps({'elapsed_s':elapsed, 'positions_rad':output_pose.tolist(),
                                     'joint_names':[NAMES[i] for i in active],
                                     'left_collision_context_rad':pose[8:].tolist() if scope == 'right' else None,
                                     'simulation_only':True, 'scope':scope})+'\n')
            node.send_output(output_name, pa.array(output_pose), {'simulation_only':True})
            if elapsed >= t[-1]:
                node.send_output('status', pa.array(['SIM_REPLAY_COMPLETE']))
                print('SIM_REPLAY_COMPLETE; no physical replay authorization', flush=True)
                break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', default=os.getenv('HOME_REPLAY_TRAJECTORY'))
    p.add_argument('--report', default=os.getenv('HOME_REPLAY_REPORT'))
    p.add_argument('--kinesthetic-workspace', type=Path, default=KIN)
    p.add_argument('--start-state', help='optional JSON array of all 16 joints for offline mismatch check')
    p.add_argument('--dora', action='store_true')
    p.add_argument('--scope', choices=['all','right'], default=os.getenv('HOME_REPLAY_SCOPE', 'all'))
    p.add_argument('--log', default=os.getenv('HOME_REPLAY_LOG'))
    a = p.parse_args()
    if not a.trajectory or not a.report:
        p.error('--trajectory and --report (or HOME_REPLAY_* environment) required')
    if a.dora:
        if not a.log:
            p.error('--log or HOME_REPLAY_LOG required')
        run_dora(a.trajectory, a.report, a.log, a.kinesthetic_workspace, scope=a.scope)
    else:
        report = validate(a.trajectory, a.report, a.kinesthetic_workspace,
                          json.loads(Path(a.start_state).read_text()) if a.start_state else None, scope=a.scope)
        print(json.dumps({k:v for k,v in report.items() if k not in ('model_hashes','first_100_collision_samples')}, indent=2))
        sys.exit(0 if report['geometric_checks_passed'] else 2)
