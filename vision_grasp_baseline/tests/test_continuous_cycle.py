import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from vision_grasp_baseline import cycle


@pytest.mark.parametrize('check_passed',[True,False])
@pytest.mark.parametrize('stationary',[True,False])
def test_default_continuous_cycle_plans_everything_before_executor_owns_recording(tmp_path,monkeypatch,check_passed,stationary):
    base=tmp_path/'baseline';base.mkdir()
    monkeypatch.setattr(cycle,'BASE',base);monkeypatch.setattr(cycle,'WORKSPACE',tmp_path)
    monkeypatch.setenv('VISION_CAPTURE_SERVICE','http://test')
    out=tmp_path/'run';episode=tmp_path/'episode';calls=[]
    monkeypatch.setattr('sys.argv',['cycle','--continuous','--output',str(out),'--record-episode',str(episode)]+
                        (['--stationary-fixture'] if stationary else []))
    state=dict(status=dict(state='idle',fault=None),pending=[],synchronized=True,hold_pending=[])
    monkeypatch.setattr(cycle.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=json.dumps(state)))
    scene=dict(measured=dict(cube=dict(center=[.3,.2,.1]),bin=dict(center_xy=[0.,0.],yaw=0.,outer_size_xy=[.1,.1])))
    class Process:
        returncode=0
        def __init__(self,cmd,**kwargs):
            stage=cmd[1];calls.append(stage)
            def argument(key):return Path(cmd[cmd.index(key)+1])
            if stage=='capture':argument('--output').mkdir()
            elif stage=='detect':(argument('--capture')/'scene.json').write_text(json.dumps(scene))
            elif stage=='zero-test':argument('--output').write_text('{"success":true}')
            elif stage=='observe':
                argument('--output').mkdir();(argument('--output')/'robot.urdf').write_text('model')
            elif stage=='continuous-plan':
                argument('--output').mkdir();(argument('--output')/'trajectory.json').write_text('{"duration_s":33.0}')
            elif stage=='continuous-check':
                assert '--record-episode' not in cmd
                argument('--output').mkdir();(argument('--output')/'result.json').write_text('{"success":true}')
                if not check_passed:self.returncode=1
            elif stage=='continuous-execute':
                assert argument('--record-episode')==episode
                assert argument('--plan').exists()
                assert argument('--validation-proof').exists()
                argument('--output').mkdir()
                (argument('--output')/'result.json').write_text(json.dumps(dict(success=True,returned_mechanical_zero=True,recording_quality=dict(passed=True))))
            else:raise AssertionError(stage)
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(cycle.subprocess,'Popen',Process)
    def unexpected_rpc(*args,**kwargs):raise AssertionError('Parent must not begin recording before full plan')
    monkeypatch.setattr('vision_grasp_baseline.collection_io.rpc',unexpected_rpc)
    if not check_passed:
        with pytest.raises(RuntimeError,match='continuous-check failed'):cycle.main()
        assert calls[-1]=='continuous-check' and 'continuous-execute' not in calls
        assert not episode.exists()
        return
    cycle.main()
    assert calls==['capture','detect','zero-test','capture','detect','observe','continuous-plan']+\
                  ([] if stationary else ['capture','detect'])+['continuous-check','continuous-execute']
    report=json.loads((out/'cycle_result.json').read_text())
    assert report['success'] and report['returned_mechanical_zero']
    assert report['recording_quality']['passed']
