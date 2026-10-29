"""Diagnostics must expose missed events, exclude padding, and leave inference unchanged."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="Optional ACT dependency; CPU collection tests do not require it")

from experiments.act_sim_mix.metrics import (
    PredictionMetrics, event_metrics, phase_labels, recovery_success,
)
from experiments.act_sim_mix.evaluate_session import summarize_session
from experiments.act_sim_mix.train import evaluate


def simulated_dataset(tmp_path, n=24, prefix=3):
    raw = tmp_path / 'raw'
    raw.mkdir()
    phases = np.array(['FAILED_LIFT'] * prefix + ['PREGRASP'] * 8 + ['CLOSE'] * 8 + ['TRANSPORT'] * (n - 16))
    np.savez(raw / 'numeric.npz', phase=phases, valid=np.arange(n + prefix) >= prefix)
    action = np.zeros((n, 16), dtype=np.float32)
    action[:, 7] = np.r_[np.linspace(-.02, -.45, 8), np.linspace(-.45, -.02, 8), np.full(n - 16, -.02)]
    entry = {'collection_root': str(tmp_path), 'path': 'raw', 'converted_start_frame': prefix}
    episode = {'number': 0, 'length': n, 'a': action, 'domain': 'recovery', 'recovery_type': 'bad_height'}
    return SimpleNamespace(manifest={'episodes': [entry]}, episodes=[episode])


def oracle_predictions(a, horizon):
    return a[np.minimum(np.arange(len(a))[:, None] + np.arange(horizon), len(a) - 1)]


def test_phase_errors_align_to_target_time_and_ignore_padding(tmp_path):
    ds = simulated_dataset(tmp_path)
    pred = oracle_predictions(ds.episodes[0]['a'], 100)
    # Garbage in padded targets cannot affect errors.
    for i in range(len(pred)):
        pred[i, len(pred) - i:] = 999
    metrics = PredictionMetrics(ds, 100)
    metrics.update(pred, np.zeros(len(pred), int), np.arange(len(pred)))
    report = metrics.report()
    assert report['overall']['0']['frames'] == 24
    assert report['overall']['10']['frames'] == 14
    assert '25' not in report['overall']
    assert report['overall']['10']['arm_mae_deg'] == 0
    assert report['by_phase']['pregrasp_open']['0']['frames'] == 8
    assert '10' not in report['by_phase']['pregrasp_open']
    assert report['by_phase']['carry']['10']['frames'] == 8
    assert report['by_recovery_type']['bad_height']['0']['frames'] == 24
    assert report['by_recovery_type']['closed_approach'] == {}


def test_missing_opening_is_counted_and_known_delay_is_measured():
    reference = np.r_[np.full(30, -.02), np.linspace(-.02, -.45, 30), np.full(30, -.45)]
    labels = np.array(['reach'] * 30 + ['pregrasp_open'] * 60)
    missed = event_metrics(np.full(90, -.02), reference, labels)
    row = missed['pregrasp_open']['by_fraction']['0.5']
    assert row['reference_events'] == row['missing_events'] == 1
    assert row['missing_fraction'] == 1
    assert row['delay_s']['count'] == 0 and row['delay_s']['mean'] is None
    assert missed['failure_to_open']['fraction'] == 1
    delayed = event_metrics(np.r_[np.full(6, reference[0]), reference[:-6]], reference, labels)
    assert delayed['pregrasp_open']['by_fraction']['0.5']['delay_s']['mean'] == pytest.approx(.2)
    assert delayed['release']['eligible_events'] == 0


def test_incomplete_diagnostics_rejected(tmp_path):
    metrics = PredictionMetrics(simulated_dataset(tmp_path), 100)
    with pytest.raises(ValueError, match='coverage'):
        metrics.report()


def test_phase_sources_are_verified_and_exclude_failed_prefix(tmp_path):
    ds = simulated_dataset(tmp_path)
    labels, _ = phase_labels(ds, ds.episodes[0])
    assert len(labels) == 24 and labels[0] == 'pregrasp_open'
    ds.manifest['episodes'][0]['sha256_numeric'] = 'tampered'
    with pytest.raises(ValueError, match='hash mismatch'):
        phase_labels(ds, ds.episodes[0])


def test_real_phase_timestamps_use_episode_start(tmp_path):
    events = [{'phase': 'timeline_event', 'timestamp_s': t, 'detail': {'name': name}}
              for t, name in ((100.1, 'pregrasp_open'), (100.2, 'approach'), (100.3, 'close'))]
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    ds = SimpleNamespace(manifest={'episodes': [{'source_episode_path': str(tmp_path), 'start_timestamp_s': 100.}]})
    labels, _ = phase_labels(ds, {'number': 0, 'length': 15})
    assert labels[0] == 'other' and labels[4] == 'pregrasp_open'
    assert labels[7] == 'approach' and labels[10] == 'close'


def test_recovery_rates_keep_denominators_and_missing_types():
    rows = [{'scene': {'recovery_type': 'bad_height', 'group_id': 'same_family'},
             'success': flag, 'failure_type': None if flag else 'not_released'} for flag in (True, False)]
    report = recovery_success(rows)
    assert report['bad_height']['scenes'] == 2 and report['bad_height']['families'] == 1
    assert report['bad_height']['success_rate'] == .5
    assert report['bad_height']['failure_types'] == {'not_released': 1}
    assert report['closed_approach']['success_rate'] is None


def test_training_diagnostics_reuse_one_native_forward_and_restore_mode(tmp_path):
    ds = simulated_dataset(tmp_path)
    class Dataset:
        episodes, manifest = ds.episodes, ds.manifest
        def __len__(self): return len(ds.episodes[0]['a'])
        def __getitem__(self, i):
            a = oracle_predictions(ds.episodes[0]['a'], 100)[i]
            return {'action': torch.tensor(a), 'observation.state': torch.zeros(16),
                    'action_is_pad': torch.tensor(np.arange(100) + i >= len(self)),
                    'episode_index': torch.tensor(0), 'frame_index': torch.tensor(i),
                    **{f'observation.images.{role}': torch.zeros(3, 2, 2) for role in ('ceiling', 'wrist_right')}}
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def forward(self, batch):
            self.calls += 1
            return batch['action'], (None, None)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = Model(); self.config = SimpleNamespace(chunk_size=100)
        def forward(self, batch):
            pred, _ = self.model(batch)
            return (pred - batch['action']).abs().mean(), {}
    policy = Policy()
    rng = torch.get_rng_state().clone()
    def pre(batch): return {**batch, 'action': batch['action'] / 2}
    report = evaluate(policy, pre, lambda a: a * 2, {'sim_validation': Dataset()})['sim_validation']
    assert report['loss'] == 0 and report['diagnostics']['overall']['0']['gripper_mae_rad'] == 0
    assert policy.model.calls == 3 and policy.training
    assert not policy.model._forward_hooks
    assert torch.equal(rng, torch.get_rng_state())


def write_records(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_session_reports_actual_tracking_boundaries_and_separate_latency(tmp_path):
    positions = np.zeros((100, 16)).tolist()
    positions[0][0] = .2
    raw, executor, driver = [], [], []
    for i, base in enumerate((1_000_000_000, 2_000_000_000)):
        md = {'chunk_id': str(i), 'n_action_steps': 50, 'observation_monotonic_ns': base,
              'requested_monotonic_ns': base + 10_000_000, 'replied_monotonic_ns': base + 30_000_000,
              'execution_mode': 'chunk100_exec50'}
        raw.append({'event': 'raw_policy_chunk', 'positions': positions, 'metadata': md})
        executor.append({'event': 'executor_output', 'output': 'move_position_right',
                         'monotonic_ns': base + 40_000_000, 'metadata': md})
        driver.append({'event': 'mit_command', 'monotonic_ns': base + 45_000_000, 'reference': md,
                       'q_ref': [.1 * i] * 8, 'q_feedback': [.1 * i - .01] * 8, 'held': False})
    driver.append({**driver[-1], 'held': True})
    for name, rows in (('policy_client', raw), ('executor', executor), ('feedforward_right', driver)):
        write_records(tmp_path / (name + '.jsonl'), rows)
    report = summarize_session(tmp_path)
    assert report['active_driver_samples'] == 2 and report['held_driver_samples_excluded'] == 1
    assert report['tracking']['gripper_abs_error_rad']['mean'] == pytest.approx(.01)
    assert report['timing_ms']['inference_round_trip']['mean'] == 20
    assert report['timing_ms']['observation_to_executor_output']['mean'] == 40
    assert report['timing_ms']['observation_to_driver_log']['mean'] == 45
    assert report['boundaries'][0]['actual_reference_to_previous_reference'] is None
    assert report['filtered_boundary_max_arm_deg']['mean'] == pytest.approx(.1 * 180 / np.pi)


def test_missing_physical_evidence_is_null_not_zero(tmp_path):
    report = summarize_session(tmp_path)
    assert len(report['missing_files']) == 3
    assert report['tracking']['samples'] == 0
    assert report['timing_ms']['observation_to_driver_log']['mean'] is None
