import json

import pytest

from vision_grasp_baseline.prepare_training import selection


def populate(root):
    for tag,count in [('nearby',10),('spread',16),('rotation',6)]:
        for index in range(count):
            path=root/tag/'episodes'/f'episode_{index:04d}'
            path.mkdir(parents=True)
            (path/'metadata.json').write_text(json.dumps(dict(review_status='approved',terminal_label='returned_zero',
                operator_marked_success=True,automatic_task_contract_passed=True,data_quality_passed=True,
                source_trajectory_trial_id=f'{tag}_{index}')))


def test_split_is_reproducible_stratified_and_disjoint_by_physical_episode(tmp_path):
    populate(tmp_path);result=selection(tmp_path)
    assert result==selection(tmp_path)
    train=result['splits']['train'];val=result['splits']['validation']
    assert len(train)==26 and len(val)==6 and not set(train)&set(val)
    assert result['counts']==dict(nearby=dict(train=8,validation=2),spread=dict(train=13,validation=3),rotation=dict(train=5,validation=1))


def test_changed_or_unapproved_dataset_cannot_silently_change_frozen_recipe(tmp_path):
    populate(tmp_path)
    file=tmp_path/'nearby/episodes/episode_0000/metadata.json'
    data=json.loads(file.read_text());data['operator_marked_success']=False;file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='expected frozen 10'):selection(tmp_path)


def test_duplicate_physical_source_is_rejected_even_with_different_folder_names(tmp_path):
    populate(tmp_path)
    for file in (tmp_path/'nearby/episodes').glob('*/metadata.json'):
        data=json.loads(file.read_text());data['source_trajectory_trial_id']='one_recording';file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Duplicate|leakage'):selection(tmp_path)
