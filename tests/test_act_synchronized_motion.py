import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.act_trial import motion_generator as motion
from scripts.act_trial.guard import Guard
from test_act_trial_guard import setup, Scene, start_guard  # shared synthetic limit fixture


def generator():
    return motion.SynchronizedGenerator(np.full(8, .25), .25,
                                        np.full(8, -2.), np.full(8, 2.), True)


def test_phase_sync_keeps_straight_path_for_static_target_from_rest():
    g = generator()
    q, v = np.zeros(8), np.zeros(8)
    target = np.arange(1, 9)*.01
    for _ in range(200):
        previous_v = v.copy()
        q, v, _, _ = g.propose(q, v, target, .01)
        # All joints share path progress for compatible rest-to-rest inputs.
        np.testing.assert_allclose(q/target, np.full(8, q[0]/target[0]), atol=1e-8)
        assert np.max(np.abs(v-previous_v)) <= .0025+1e-10
    np.testing.assert_allclose(q, target, atol=1e-10)


def test_reversals_preserve_velocity_bound_not_reset():
    g = generator()
    q, v = np.zeros(8), np.zeros(8)
    for step in range(120):
        target = np.full(8, .3 if step < 60 else -.3)
        prev_v = v.copy()
        q, v, _, _ = g.propose(q, v, target, .01)
        assert np.max(np.abs(v-prev_v)) <= .0025+1e-10
        assert np.max(np.abs(v)) <= .25+1e-10
        if step == 60:
            assert np.min(v) > .10  # Cannot instantly reverse or zero velocity.


def test_braking_reserve_satisfies_existing_acceleration_bound():
    q, v = np.zeros(8), np.linspace(-.25, .25, 8)
    end, duration = motion.braking_reserve(q, v, .25)
    assert duration == 1.
    assert np.max(np.abs(v/duration)) <= .25
    np.testing.assert_allclose(end, v/2)
    np.testing.assert_allclose(motion.braking_reserve(q, np.zeros(8), .25)[0], q)


def test_recorded_implementation_metadata_does_not_block_repeated_testing(monkeypatch):
    c = {"motion_generator": motion.contract("synchronized_v1")}
    assert motion.from_config(c) == "synchronized_v1"
    c["motion_generator"]["source_sha256"] = {"guard.py": "previous-version"}
    c["motion_generator"]["library_sha256"] = "previous-recorded-metadata"
    c["motion_generator"]["collision"] = "previous-description"
    # Reading a stored selection no longer computes/compares a current contract.
    monkeypatch.setattr(motion, 'contract', lambda *_: pytest.fail('metadata comparison'))
    assert motion.from_config(c) == "synchronized_v1"
    assert motion.from_config({"motion_generator": {"name": "synchronized_v1"}}) == "synchronized_v1"
    assert motion.from_config({}) == "legacy"
    with pytest.raises(ValueError):
        motion.from_config({"motion_generator": {"name": "typo"}})


def test_prepare_binds_mode_without_hardware(monkeypatch, tmp_path):
    from scripts.act_trial import manual_experiment as manual
    monkeypatch.setattr(manual, "ROOT", tmp_path)
    (tmp_path/"results").mkdir()
    monkeypatch.setattr(manual, "environment", lambda *args: {})
    monkeypatch.setattr(manual, "offline_preflight",
                        lambda *args, **kw: ({"offline_checks_passed": True}, {}))
    manual.prepare(profile="task", checkpoint=tmp_path/"model",
                   motion_generator="synchronized_v1")
    config_path, = (tmp_path/"results").glob("act_manual_*/act/trial_config.json")
    assert motion.from_config(json.loads(config_path.read_text())) == "synchronized_v1"


@pytest.mark.parametrize('stale_contract', [False, True])
def test_next_trial_preserves_selected_mode(monkeypatch, tmp_path, stale_contract):
    from scripts.act_trial import manual_experiment as manual
    from test_act_manual_experiment import motion_attempt
    state = {"status": {"state": "idle"}, "synchronized": True, "drain_ready": True,
             "pending": [], "hold_pending": [], "raw_active": {"arm": []}}
    trial = motion_attempt(monkeypatch, tmp_path, state)
    old_contract = motion.contract("synchronized_v1")
    if stale_contract:
        old_contract['source_sha256'] = {'previous_guard.py': 'old-source-hash'}
        old_contract['collision'] = 'previous reserve rejection policy'
    (trial/"trial_config.json").write_text(json.dumps({"motion_generator": old_contract}))
    manual.next_trial(tmp_path, trial)
    assert motion.from_config(json.loads((trial/"trial_config.json").read_text())) == "synchronized_v1"
    archive, = tmp_path.glob('act_completed_attempt_*')
    assert json.loads((archive/'trial_config.json').read_text())['motion_generator'] == old_contract
    recovery = json.loads((trial/'recovery.json').read_text())
    assert recovery['motion_contract_refreshed'] is stale_contract
    assert recovery['automatic_START'] is False


@pytest.mark.parametrize('stored', [None, {}, {'name': 'unknown'}])
def test_restart_rejects_invalid_mode_instead_of_falling_back(stored):
    with pytest.raises(ValueError, match='selection'):
        motion.selected_name({'motion_generator': stored})


def test_default_guard_unchanged_and_opt_in_keeps_commissioning_bounds(setup):
    initial, bundle, _ = setup
    legacy = Guard(initial, bundle, now=0., scene=Scene())
    assert legacy.generator is None
    g = Guard(initial, bundle, now=0., scene=Scene(), motion_generator="synchronized_v1")
    start_guard(g)
    measured = initial.copy()
    for i in range(1, 301):
        request = initial[:8]+.03
        d = g.tick(request, measured, .01+i*.01, .001, .01)
        assert d["stage"] in ("active", "complete"), d["errors"]
        measured[:8] = d["dispatched_proposal"]
        assert abs(measured[7]-initial[7]) < 1e-12
        assert np.max(np.abs(measured[:7]-initial[:7])) <= .04+1e-10
    assert d["stage"] == "complete"


def test_commissioning_collision_still_stops_without_reserve_rejection(setup):
    initial, bundle, _ = setup
    obstacle = initial[0]+.02
    scene = Scene(lambda q: q[0] > obstacle)
    g = Guard(initial, bundle, now=0., scene=scene, profile="commissioning",
              motion_generator="synchronized_v1")
    start_guard(g)
    measured = initial.copy()
    request = initial[:8].copy()
    request[0] += .03
    reserve_crossed_while_active = False
    for i in range(1, 201):
        d = g.tick(request, measured, .01+i*.01, .001, .01)
        measured[:8] = d["dispatched_proposal"]
        if d["stage"] == "fault":
            break
        if d['motion_generation']['braking_reserve_endpoint_rad'][0] > obstacle:
            reserve_crossed_while_active = True
            assert d['generated_proposal_rad'][0] <= obstacle
    assert reserve_crossed_while_active
    assert d["stage"] == "fault"
    assert d["errors"] == ["collision in right command proposal sweep"]
    assert d["dispatched_proposal"][0] < obstacle
    assert d["generated_proposal_rad"][0] > obstacle


def test_recorded_grasp_boundary_roundoff_does_not_reject_command():
    # Last accepted state and next target from the 2026-10-21 grasp attempt.
    q = np.array([.05959244485268065, .049717309319780786, .06195833447848587,
                  .7566371933670971, .10467709493712336, -.7597657657212681,
                  .07519552198343346, -.10564286683314895])
    v = np.array([.007919310790439324, -.004215131499115011, .006371262565857761,
                  .015348987111246005, -.0035689517186661885, -.05103942892984852,
                  .017190733943903542, .053117166872626936])
    target = np.array([.06105447933077812, .048869624733924866, .06325633078813553,
                       .7593691349029541, .10392650961875916, -.7654,
                       .07821894437074661, -.1])
    lower = np.array([-1.3963, -.17453, -1.5708, -.02, -1.5708, -.7654, -1.5508, -.35])
    upper = np.array([3.4907, 3.3161, 1.5708, 2.4435, 1.5708, .7654, 1.5508, -.1])
    g = motion.SynchronizedGenerator(np.full(8, .25), .25, lower, upper, True)
    candidate, velocity, _, _ = g.propose(q, v, target, .01)
    assert np.all(candidate >= lower) and np.all(candidate <= upper)
    assert np.max(np.abs(velocity-v)) <= .0025+1e-10
    assert g.diagnostics['braking_reserve_policy'] == 'diagnostic_only'
    assert 1e-10 < g.diagnostics['braking_reserve_limit_excess_rad'][7] < 1e-8


@pytest.mark.parametrize("kind", ["freshness", "watchdog", "measured_limits", "action_age"])
def test_synchronized_mode_does_not_bypass_existing_gates(setup, kind):
    initial, bundle, _ = setup
    g = Guard(initial, bundle, now=0., scene=Scene(), motion_generator="synchronized_v1")
    start_guard(g)
    measured = initial.copy()
    if kind == "measured_limits":
        measured[5] = -.8
    d = g.tick(initial[:8], measured, .2 if kind == "watchdog" else .02,
               .11 if kind == "freshness" else .001, .51 if kind == "action_age" else .01)
    assert d["stage"] == "fault"


def test_no_nonfinite_or_invalid_period():
    g = generator()
    with pytest.raises(ValueError):
        g.propose(np.zeros(8), np.zeros(8), np.full(8, np.nan), .01)
    with pytest.raises(ValueError):
        g.propose(np.zeros(8), np.zeros(8), np.zeros(8), .02)


def test_live_training_guard_does_not_signal_processes(monkeypatch):
    monkeypatch.setattr(motion, "training_processes", lambda: [123])
    with pytest.raises(ValueError, match="training is running"):
        motion.check_live_host("synchronized_v1")
    motion.check_live_host("legacy")


def test_training_scan_matches_arguments_not_embedded_shell_text(tmp_path):
    one = tmp_path/"101"; one.mkdir()
    (one/"cmdline").write_bytes(b"python\0scripts/train_act_real7030.py\0--config=x\0")
    two = tmp_path/"102"; two.mkdir()
    (two/"cmdline").write_bytes(b"bash\0-c\0echo scripts/train_act_real7030.py\0")
    assert motion.training_processes(tmp_path) == [101]
