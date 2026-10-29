"""A simulation pilot must not require private real-robot data to audit."""
from argparse import Namespace

from experiments.act_sim_mix import prepare_dataset as prep


def test_pilot_audit_does_not_read_real_data_unless_requested(tmp_path,monkeypatch):
    report={'rows':[{}], 'episodes':1,'normal':1,'recovery':0,'recovery_counts':{},
            'normal_frames':2,'recovery_valid_frames':0,'complete':False}
    monkeypatch.setattr(prep,'combined_audit',lambda _:dict(report))
    def forbidden():raise AssertionError('Real data was not requested')
    monkeypatch.setattr(prep,'real_audit',forbidden)
    args=Namespace(collection=[tmp_path],report=tmp_path/'audit.json',pilot=True,real_root=None)
    assert prep.cmd_audit(args)==0
    assert not (tmp_path/'real_audit.json').exists()
