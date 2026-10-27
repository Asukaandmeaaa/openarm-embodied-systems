"""Synthetic rolling trajectories and fake transports only. No hardware."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import yaml

from scripts.act_trial.rolling_actions import (
    RollingActions, Chunk, SCHEDULE, INTERVAL_NS, MAX_AGE_NS, EXPIRY_NS, contract,
)


def meta(origin, identifier=None):
    return dict(observation_monotonic_ns=int(origin), source_age_ns=5_000_000,
                source_skew_ns=2_000_000, interval=INTERVAL_NS,
                action_schedule=SCHEDULE, queue_chunk_id=str(origin) if identifier is None else identifier,
                reset=False)


def line(origin, intercept=0., velocity=.3):
    t=(origin/1e9)+np.arange(50)*INTERVAL_NS/1e9
    q=np.zeros((50,16)); q[:,0]=intercept+velocity*t; q[:,7]=-.02
    return q


def started():
    r=RollingActions(); seed=np.zeros(8);seed[7]=-.02
    r.begin(seed, 0)
    return r


def test_full_chunk_time_origin_and_analytic_hermite_derivative():
    c=Chunk(line(1_000_000_000),meta(1_000_000_000),1_050_000_000)
    q,v=c.sample(1_250_000_000)
    assert q[0] == pytest.approx(.375)
    assert v[0] == pytest.approx(.3)
    assert c.q.shape == (50,8)


def test_start_discards_dry_run_and_waits_for_post_start_observation():
    r=RollingActions()
    r.add(line(0),meta(0),50_000_000)
    r.begin(np.zeros(8),100_000_000)
    assert r.chunks == []
    assert not r.add(line(90_000_000),meta(90_000_000),110_000_000)
    assert r.sample(120_000_000) is None
    r.add(line(130_000_000),meta(130_000_000),180_000_000)
    assert r.sample(180_000_000) is None
    q,v,_=r.sample(180_000_001)
    np.testing.assert_allclose(q,0.,atol=1e-12)
    np.testing.assert_allclose(v,0.,atol=1e-7)


def test_new_chunk_does_not_restart_at_its_first_point():
    r=started(); r.add(line(0),meta(0),50_000_000)
    q0,v0,_=r.sample(270_000_000)
    r.add(line(220_000_000,intercept=-.08),meta(220_000_000),270_000_000)
    q1,v1,_=r.sample(270_000_000)
    np.testing.assert_array_equal(q0,q1)
    np.testing.assert_array_equal(v0,v1)
    q,v,m=r.sample(350_000_000)
    assert q[0] == pytest.approx(.105)  # continuing OLD tail past raw index 10
    assert v[0] == pytest.approx(.3)
    assert m['contributors'][0]['raw_step'] > 10


def test_position_and_velocity_continuous_through_admission_and_expiry():
    r=started()
    r.add(line(0),meta(0),40_000_000)
    r.add(line(220_000_000,intercept=-.08),meta(220_000_000),260_000_000)
    for stamp in (260_000_000, 360_000_000, EXPIRY_NS-100_000_000, EXPIRY_NS):
        a=r.sample(stamp-100); b=r.sample(stamp+100)
        np.testing.assert_allclose(a[0],b[0],atol=1e-6)
        np.testing.assert_allclose(a[1],b[1],atol=2e-5)
    for stamp in (280_000_000,400_000_000,430_000_000,470_000_000,500_000_000):
        q,v,m=r.sample(stamp)
        derivative=(r.sample(stamp+1000)[0]-r.sample(stamp-1000)[0])/(2e-6)
        np.testing.assert_allclose(v,derivative,atol=1e-6)
        assert stamp-m['observation_monotonic_ns'] < MAX_AGE_NS


def test_no_deadline_refresh_when_old_and_new_predictions_blend():
    r=started();r.add(line(0),meta(0),40_000_000)
    r.add(line(200_000_000),meta(200_000_000),240_000_000)
    _,_,m=r.sample(430_000_000)
    assert m['observation_monotonic_ns'] == 0
    assert len(m['contributors']) == 2
    _,_,m=r.sample(EXPIRY_NS)
    assert m['observation_monotonic_ns'] == 200_000_000
    with pytest.raises(ValueError,match='no fresh'):
        r.sample(200_000_000+EXPIRY_NS)


def test_missing_first_post_start_prediction_times_out():
    r=started()
    assert r.sample(MAX_AGE_NS-1) is None
    with pytest.raises(ValueError,match='no fresh'):
        r.sample(MAX_AGE_NS)


def test_stop_latches_and_never_accepts_later_predictions():
    r=started();r.add(line(0),meta(0),40_000_000);r.stop()
    assert not r.add(line(100_000_000),meta(100_000_000),140_000_000)
    with pytest.raises(ValueError,match='stopped'):r.sample(150_000_000)
    with pytest.raises(ValueError,match='restart'):r.begin(np.zeros(8),160_000_000)


@pytest.mark.parametrize('bad', ['nan','shape','interval','future','stale','reset'])
def test_invalid_predictions_rejected(bad):
    r=started();q=line(0);m=meta(0);now=40_000_000
    if bad=='nan':q[20,2]=np.nan
    if bad=='shape':q=q[:10]
    if bad=='interval':m['interval']=10_000_000
    if bad=='future':m['observation_monotonic_ns']=now+1
    if bad=='stale':now=MAX_AGE_NS
    if bad=='reset':m['reset']=True
    with pytest.raises(ValueError):r.add(q,m,now)


def test_reordered_or_duplicate_observation_rejected():
    r=started();r.add(line(0),meta(0),40_000_000)
    with pytest.raises(ValueError,match='duplicate'):r.add(line(0),meta(0),50_000_000)


def test_rolling_graph_has_no_completion_barrier_or_second_execution_clock():
    root=Path(__file__).resolve().parents[1]
    graph=yaml.safe_load((root/'official/dora-openarm-evaluation/dataflow-act-visual32-ros-host.yaml').read_text())
    nodes={n['id']:n for n in graph['nodes']}
    assert 'actions-executor' not in nodes
    assert nodes['trial-bridge']['inputs']['policy_actions']=='policy-server/actions'
    assert 'chunk_complete' not in nodes['policy-server']['inputs']
    assert nodes['policy-server']['inputs']['execution_stop']=='trial-bridge/execution_stop'


def test_client_sends_full_fifty_and_fresh_requests_without_ack(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts/act_trial'))
    sys.path.insert(0,str(root/'official/dora-openarm-evaluation/src'))
    import queued_policy_client as client
    clock=SimpleNamespace(now=1_000_000_000,monotonic_ns=lambda:clock.now)
    monkeypatch.setattr(client,'time',clock)
    class Node:
        outputs=[]
        def __iter__(self):
            for i,t in enumerate([1_000_000_000,1_033_000_000,1_120_000_000]):
                clock.now=t
                yield dict(type='INPUT',id='observation',metadata=meta(t),
                           value=pa.array([{'id':i,'position':[0.]*16}]))
            yield dict(type='INPUT',id='execution_stop')
        def send_output(self,*args):self.outputs.append(args)
    class IO:
        requests=[]
        def write(self,s):self.requests.append(json.loads(s))
        def flush(self):pass
        def readline(self):return json.dumps(dict(positions=line(0).tolist(),interval=INTERVAL_NS))
    log=SimpleNamespace(run_id='synthetic',write=lambda *a,**k:None)
    node,sock=Node(),IO()
    client.run(node,sock,str(tmp_path),log,rolling=True)
    assert len(sock.requests)==2
    assert len(node.outputs)==2
    for name,value,m in node.outputs:
        assert name=='actions'
        assert np.array(value.to_pylist()).shape==(50,16)
        assert m['action_schedule']==SCHEDULE


def test_synthetic_backtrack_prefix_does_not_repeat_with_zero_net_progress():
    # Synthetic policy, not a replay of demonstration data or a robot model:
    # first point is q-.08 and point 9 returns exactly to q. A hard ten-step
    # prefix has zero net progress. The rolling fresh tail may reach later steps.
    r=started();q=np.zeros(8);q[7]=-.02;pending=[]
    for i in range(401):
        stamp=i*10_000_000
        if i%10==0:
            p=np.zeros((50,16));p[:,7]=-.02
            p[:,0]=q[0]-.08+(.08/9)*np.arange(50)
            assert p[9,0] == pytest.approx(q[0])
            pending.append((stamp+40_000_000,p,meta(stamp)))
        while pending and pending[0][0]<=stamp:
            arrival,p,m=pending.pop(0);r.add(p,m,arrival)
        sample=r.sample(stamp)
        if sample is not None:q=sample[0]
    assert q[0] > .05
