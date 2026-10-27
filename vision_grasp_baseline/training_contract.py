"""Read-only provenance audit for closed-transit ACT supervision; no ROS/CAN."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from .collection_io import ORDER, TARGET_STREAMS
from .gripper_settings import COLLECTION_POLICY

INTERFACES = {'right_joint_trajectory_controller': ['position', 'velocity'],
              'left_joint_trajectory_controller': ['position'],
              'right_gripper_controller': ['position']}
PART_NAMES = {'right_arm': ORDER[:7], 'right_gripper': ORDER[7:8], 'left_arm': ORDER[8:15]}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ordered(names, positions, expected):
    if len(names) != len(positions) or len(set(names)) != len(names) or not set(expected) <= set(names):
        raise ValueError('Invalid joint names/position dimensions')
    values = np.asarray([positions[names.index(n)] for n in expected], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite joint positions')
    return values


def audit_action_contract(path):
    """Prove state<-JointState, action<-causal JTC output, without changing data.

    Tracking error is diagnostic, not a reason to replace expert command labels
    with measured poses. Matching the deployment controller is a separate gate.
    """
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text())
    if meta.get('gripper_policy') != COLLECTION_POLICY:
        raise ValueError('Action-contract audit expects '+COLLECTION_POLICY)
    if meta.get('joint_order') != ORDER or not meta.get('action_source', '').startswith('JTC output.positions'):
        raise ValueError('Unexpected observation/action contract in metadata')
    actual_interfaces = meta.get('controller_command_interfaces', {})
    if any(actual_interfaces.get(k) != v for k, v in INTERFACES.items()):
        raise ValueError('Collection controller interfaces missing/mismatched; do not assume old ACT compatibility')
    targets = {key: read_rows(path/f'target_{key}.jsonl') for key in TARGET_STREAMS}
    times = {key: [r['timestamp_s'] for r in rows] for key, rows in targets.items()}
    for key, ts in times.items():
        if not ts or not np.isfinite(ts).all() or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError('Missing/non-monotonic target stream: '+key)
        for row in targets[key]:
            ordered(PART_NAMES[key], row['positions'], PART_NAMES[key])
    # Independent raw ROS evidence, not just the label's human-readable name.
    raw = {key: {} for key in TARGET_STREAMS}
    for row in read_rows(path/'controller_raw.jsonl'):
        key = row['controller']
        if key in raw:
            stamp = row['timestamp_s']
            if stamp in raw[key]:
                raise ValueError('Duplicate raw controller timestamp: '+key)
            raw[key][stamp] = row['message']
    for key, rows in targets.items():
        for row in rows:
            message = raw[key].get(row['timestamp_s'])
            if message is None:
                raise ValueError('Missing raw controller output evidence: '+key)
            q = ordered(message['joint_names'], message['output']['positions'], PART_NAMES[key])
            if not np.array_equal(q, row['positions']):
                raise ValueError('Target is not raw JTC output.positions: '+key)
            if key == 'right_arm':
                velocity = message['output'].get('velocities', [])
                ordered(message['joint_names'], velocity, PART_NAMES[key])
    states = read_rows(path/'joint_observation.jsonl')
    state_times = [r['timestamp_s'] for r in states]
    if (not state_times or not np.isfinite(state_times).all()
            or any(b <= a for a, b in zip(state_times, state_times[1:]))):
        raise ValueError('Missing/non-monotonic measured state')
    measured = []
    for row in states:
        q = ordered(row['order'], row['positions'], ORDER)
        message = row.get('ros_message')
        if message is None or not np.array_equal(q, ordered(message['name'], message['position'], ORDER)):
            raise ValueError('Observation is not original JointState.position feedback')
        measured.append(q)
    actions = read_rows(path/'demonstration_action.jsonl')
    action_times = [r['timestamp_s'] for r in actions]
    if not action_times or any(b <= a for a, b in zip(action_times, action_times[1:])):
        raise ValueError('Missing/non-monotonic actions')
    first_join = max(ts[0] for ts in times.values())
    if action_times != [t for t in times['right_arm'] if t >= first_join]:
        raise ValueError('Actions do not cover every causally joinable right-arm output sample')
    errors = []
    max_state_age = 0.
    for row in actions:
        stamp = row['timestamp_s']
        if not np.isfinite(stamp):
            raise ValueError('Non-finite action timestamp')
        ids = {k: bisect_right(ts, stamp)-1 for k, ts in times.items()}
        if min(ids.values()) < 0:
            raise ValueError('Action uses missing/future controller target')
        component_times = {k: times[k][i] for k, i in ids.items()}
        if (component_times != row.get('component_timestamp_s') or component_times['right_arm'] != stamp
                or max(stamp-t for t in component_times.values()) > .1):
            raise ValueError('Action component timestamps are not causal/fresh')
        expected = sum((targets[k][ids[k]]['positions'] for k in TARGET_STREAMS), [])
        # TARGET_STREAMS is right arm, right finger, left arm. Left finger is passive.
        expected += [meta['initial_state'][-1]]
        action = ordered(row['order'], row['positions'], ORDER)
        if not np.array_equal(action, expected):
            raise ValueError('Action labels differ from causal controller output; do not train')
        i = bisect_right(state_times, stamp)-1
        if i >= 0:
            age = stamp-state_times[i]
            if age > .05:
                raise ValueError('Measured state is stale relative to action')
            max_state_age = max(max_state_age, age)
            errors.append(measured[i]-action)
    if len(errors) < 2:
        raise ValueError('Insufficient overlapping state/action evidence')
    errors = np.asarray(errors)
    return dict(passed=True, episode=str(path.resolve()), checked_actions=len(actions),
                checked_states=len(states), max_state_age_s=max_state_age,
                contract=dict(schema_version=1, state='JointState.position measured feedback',
                    action='absolute joint position: JTC output.positions, not measured feedback',
                    units='rad', joint_order=ORDER, fps=30, gripper_policy=COLLECTION_POLICY,
                    controller_command_interfaces=INTERFACES,
                    arm_velocity_evidence='controller_raw.jsonl: right_arm.message.output.velocities',
                    passive_left_finger='initial measured hold, not commanded',
                    deployment_requirement='match position+velocity control and validate timing/filtering; legacy position-only ACT is NOT compatible'),
                tracking_diagnostic=dict(samples=len(errors),
                    measured_minus_command_mean_rad=errors.mean(0).tolist(),
                    absolute_error_p95_rad=np.quantile(abs(errors), .95, axis=0).tolist(),
                    note='Same-source-time diagnostic, not external TCP accuracy or an action-label correction'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([audit_action_contract(path) for path in args.episodes], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
