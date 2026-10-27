"""Bridge lifecycle messages never enter action execution; no hardware I/O."""
import sys
from pathlib import Path
from types import SimpleNamespace
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/act_trial'))
from logged_executor import LoggedNode


def test_stop_cancels_receive_loop_and_latches_outputs():
    records, sends = [], []
    event = {'type': 'INPUT', 'id': 'execution_stop',
             'value': SimpleNamespace(to_pylist=lambda: ['measured speed fault'])}
    raw = SimpleNamespace(next=lambda: event, send_output=lambda *args: sends.append(args))
    log = SimpleNamespace(write=lambda *args, **kwargs: records.append((args, kwargs)))
    node = LoggedNode(raw, log, {})
    assert node.next() == {'type': 'STOP'}
    node.send_output('move_position_right', None)
    node.send_output('timing_fault', None)
    assert not sends
    assert records[0][0] == ('executor_stopped_by_bridge',)


def test_actions_are_unmodified():
    event = {'type': 'INPUT', 'id': 'actions', 'value': object()}
    node = LoggedNode(SimpleNamespace(next=lambda: event), None, {})
    assert node.next() is event
    assert not node.bridge_stopped


def test_graph_wires_stop_only_to_executor():
    graph = yaml.safe_load(Path('official/dora-openarm-evaluation/dataflow-act-trial-ros-host.yaml').read_text())
    nodes = {n['id']: n for n in graph['nodes']}
    assert 'execution_stop' in nodes['trial-bridge']['outputs']
    assert nodes['actions-executor']['inputs']['execution_stop'] == 'trial-bridge/execution_stop'
