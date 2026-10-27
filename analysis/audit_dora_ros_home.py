"""Offline HOME audit. No ROS, Dora runtime, CAN imports, or motor commands.

Run: .venv-real-act/bin/python analysis/audit_dora_ros_home.py
Outputs: results/dora_ros_home_audit_20261014/{audit.json,paths.png,paths.npz}
"""
from pathlib import Path
import hashlib
import json
import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
KIN = ROOT.parent / 'openarm_kinesthetic_ws'
EVAL = ROOT / 'official/dora-openarm-evaluation'
OUT = ROOT / 'results/dora_ros_home_audit_20261014'


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read(EVAL / 'guarded-home-right.json')
    ros_path = KIN / 'ros2_ws/src/openarm_kinesthetic_recorder/config/home_initialization.yaml'
    ros = yaml.safe_load(ros_path.read_text())
    source = EVAL / 'nodes/dora-openarm/src/dora_openarm/guarded_home.py'
    spec = importlib.util.spec_from_file_location('offline_guard', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # These three samples are transcribed from the user's 16:20 run, not simulated.
    initial = np.array([-.17223620965896025, -.03604943923094517, .10585946440833105,
                        .3900587472343009, .20656900892652708, .09098191805905209,
                        .087167162584878, -.05283436331731117])
    verified_zero = np.array([.0005722133211261138, -.005531395437552433,
                             .004386968795300206, .005912870984969842,
                             .014305333028152845, .014305333028152845,
                             .007438773174639479, initial[-1]])
    stopped = np.array([-.1287479972533756, .24395361257343318, .13637750820172378,
                        1.1675059128709844, .2817196917677567, -.18139162279697807,
                        .11768520637827073, initial[-1]])
    guard = module.Guard(cfg, initial, 0.0)
    home = np.array(cfg['home'])
    u = np.linspace(0, 1, 1001)
    blend = u*u*(3-2*u)
    paths = {}
    for label, start, end, fraction in [
        ('H1_recovery', initial, guard.targets[0], u),
        ('Dora_stage_A', verified_zero, guard.targets[1], blend),
        ('Dora_stage_B_hypothetical', guard.targets[1], guard.targets[2], blend),
        ('Dora_J8_open_hypothetical', guard.targets[2], home, blend),
    ]:
        paths[label] = start + fraction[:, None] * (end-start)
    paths['Dora_stage_A_open_gripper_sensitivity'] = paths['Dora_stage_A'].copy()
    paths['Dora_stage_A_open_gripper_sensitivity'][:, 7] = -.35

    old = read(KIN / 'results/home_commissioning/lifted_supported_home_10cm_moveit.json')
    historical = np.array(old['plan']['path_quality']['q_rad'])
    paths['historical_plan_only_NOT_executed_H2'] = np.column_stack(
        [historical, np.full(len(historical), -.342374)])
    paths['logged_stop'] = stopped[None, :]
    paths['configured_HOME'] = home[None, :]

    model = mujoco.MjModel.from_xml_path(str(KIN / 'sim/scenes/fixed_home_guard_v1.xml'))
    data = mujoco.MjData(model)
    table_body = model.body('fixed_inflated_table_guard').id
    table = model.geom('table_top_collision').id
    original_center = model.body_pos[table_body].copy()
    # XML has stale z=.090 (top .340); explicitly use H2 YAML's .020 (top .270).
    model.body_pos[table_body] = ros['static_collision']['center_xyz_m']
    indices = [int(model.joint(f'openarm_right_joint{i}').qposadr[0]) for i in range(1, 8)]
    fingers = [int(model.joint(f'openarm_right_finger_joint{i}').qposadr[0]) for i in (1, 2)]
    right_geoms = [i for i in range(model.ngeom)
                   if model.body(int(model.geom_bodyid[i])).name.startswith('openarm_right')
                   and (model.geom_contype[i] or model.geom_conaffinity[i])]
    ee = model.body('openarm_right_ee_base_link').id

    def scan(qpath):
        distances, xyz, contact_depths, pairs = [], [], [], []
        for q in qpath:
            data.qpos[indices] = q[:7]
            data.qpos[fingers] = q[7]
            mujoco.mj_forward(model, data)
            ds = [float(mujoco.mj_geomDistance(model, data, g, table, .5, None)) for g in right_geoms]
            which = int(np.argmin(ds))
            distances.append(ds[which])
            pairs.append(model.geom(right_geoms[which]).name)
            xyz.append(data.xpos[ee].copy())
            contacts = [float(c.dist) for c in data.contact
                        if (c.geom1 == table and c.geom2 in right_geoms)
                        or (c.geom2 == table and c.geom1 in right_geoms)]
            contact_depths.append(min(contacts, default=0.0))
        distances = np.array(distances)
        contact_depths = np.array(contact_depths)
        worst = int(np.argmin(distances))
        penetrating = np.flatnonzero(contact_depths < -.001)
        return {'minimum_geom_distance_m': float(distances[worst]),
                'worst_sample': worst, 'worst_geom': pairs[worst],
                'worst_q_rad': qpath[worst].tolist(),
                'minimum_contact_distance_m': float(contact_depths.min()),
                'contact_samples_deeper_than_1mm': len(penetrating),
                'first_contact_sample_deeper_than_1mm': int(penetrating[0]) if len(penetrating) else None,
                'sample_count': len(qpath)}, np.array(xyz), distances, contact_depths

    scans, arrays = {}, {}
    for scene, dimensions in [('inflated_H2_guard', ros['static_collision']['dimensions_xyz_m']),
                              ('approximate_physical_table', [.4, .6, .5])]:
        model.geom_size[table] = np.array(dimensions) / 2
        scans[scene] = {}
        for label, qpath in paths.items():
            summary, xyz, distance, contact_depth = scan(qpath)
            scans[scene][label] = summary
            arrays[label+'_q'] = qpath
            arrays[label+'_xyz'] = xyz
            arrays[scene+'_'+label+'_distance'] = distance
            arrays[scene+'_'+label+'_contact_depth'] = contact_depth
    fk_error = float(np.max(np.linalg.norm(arrays['historical_plan_only_NOT_executed_H2_xyz'] -
                                         np.array(old['plan']['path_quality']['tcp_xyz_m']), axis=1)))

    episodes = []
    for path in sorted((ROOT/'openarm_kinesthetic_data/collections/episodes_pending_review').glob('*/metadata.json')):
        meta = read(path)
        init = meta.get('home_initialization', {})
        goal = init.get('home_goal_positions_rad')
        if goal is None:
            continue
        episodes.append({'episode': path.parent.name,
                         'goal_max_error_rad': float(np.max(np.abs(np.array(goal)-home[:7]))),
                         'initial_arm_home_error_rad': float(np.max(np.abs(np.array(meta['initial_positions'][:7])-home[:7]))),
                         'initialization_trajectory_in_training_data': meta.get('initialization_trajectory_in_training_data'),
                         'home_report_path': init.get('path'),
                         'home_report_sha256': init.get('sha256'),
                         'home_trajectory_sha256': init.get('trajectory_sha256'),
                         'home_config_sha256': init.get('config_sha256'),
                         'gripper_accepted': init.get('home_gripper', {}).get('accepted'),
                         'end_home_error_rad': meta.get('episode_end_home_max_error_rad')})
    motions = []
    for path in sorted((ROOT/'openarm_kinesthetic_data/recorded_motions').glob('*.json')):
        motion = read(path)
        samples = motion.get('samples', [])
        if not samples:
            continue
        motions.append({'file': str(path.relative_to(ROOT)), 'sample_count': len(samples),
                        'first_arm_home_error_rad': float(np.max(np.abs(np.array(samples[0]['positions_rad'][:7])-home[:7]))),
                        'last_arm_home_error_rad': float(np.max(np.abs(np.array(samples[-1]['positions_rad'][:7])-home[:7])))})

    report = {
        'mode': 'OFFLINE_KINEMATIC_AUDIT_NO_HARDWARE',
        'limitations': ['Reconstructed nominal paths are not a measured per-tick Dora trace.',
                        'Stage B and gripper opening were not reached in the supplied run.',
                        'Historical MoveIt samples are plan-only, not the executed H2 trajectory.',
                        'Geometry uses the recorded 2026-09-20 registration, not a new table measurement.',
                        'MuJoCo mesh hull contacts are model evidence, not measured physical penetration.',
                        'Only right-arm/table collision is scanned; no complete safety certification.'],
        'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, ros_path, EVAL/'guarded-home-right.json']},
        'home': home.tolist(),
        'ros_dora_home_max_difference_rad': float(np.max(np.abs(home[:7]-ros['right_home_joint_positions_rad']))),
        'ros_dora_side_max_difference_rad': float(np.max(np.abs(np.array(cfg['side_prehome'][:7])-ros['side_prehome']['selected_joint_positions_rad']))),
        'ros_final_tolerance_rad': ros['final_acceptance']['joint_tolerance_rad'],
        'dora_final_tolerance_rad': cfg['tolerance'],
        'logged_stop': stopped.tolist(),
        'logged_stop_max_side_error_rad': float(np.max(np.abs(stopped-guard.targets[1]))),
        'logged_stop_max_arm_home_error_rad': float(np.max(np.abs(stopped[:7]-home[:7]))),
        'nominal_stage_A_duration_s': guard._phase_duration(verified_zero, guard.targets[1]),
        'scene_original_center': original_center.tolist(),
        'effective_center': ros['static_collision']['center_xyz_m'],
        'historical_moveit_vs_mujoco_ee_fk_max_error_m': fk_error,
        'scans': scans, 'episodes': episodes, 'recorded_motions': motions,
    }
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(OUT/'paths.npz', **arrays)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['H1_recovery', 'Dora_stage_A', 'Dora_stage_B_hypothetical',
              'historical_plan_only_NOT_executed_H2']
    for label in labels:
        xyz = arrays[label+'_xyz']
        axes[0].plot(xyz[:,0], xyz[:,2], label=label)
        axes[1].plot(xyz[:,0], xyz[:,1], label=label)
    for label in labels[:3]:
        dist = arrays['inflated_H2_guard_'+label+'_contact_depth']
        axes[2].plot(np.linspace(0, 1, len(dist)), dist*1000, label=label)
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((.132,-.23), .5,.5, color='red', alpha=.1))
    axes[1].add_patch(Rectangle((.132,-.334), .5,.7, color='red', alpha=.1))
    axes[0].set(xlabel='World x (m)', ylabel='EE base z (m)', title='Side projection (shading ignores y)')
    axes[1].set(xlabel='World x (m)', ylabel='EE base y (m)', title='Top projection (shading ignores z)')
    axes[2].axhline(0, color='black', linewidth=.8)
    axes[2].set(xlabel='Normalized stage time', ylabel='Minimum contact distance (mm; 0 = no contact)',
                title='All right-arm collision geoms vs H2 guard')
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    fig.suptitle('Offline reconstruction: endpoints match; paths do not establish ROS H2 equivalence')
    fig.tight_layout()
    fig.savefig(OUT/'paths.png', dpi=160)
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','recorded_motions','hashes')}, indent=2))
    print(f'Episodes checked: {len(episodes)}; motions checked: {len(motions)}; outputs: {OUT}')


if __name__ == '__main__':
    main()
