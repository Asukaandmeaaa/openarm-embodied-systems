import json

import pytest

from vision_grasp_baseline.export_act import collection_policy
from vision_grasp_baseline.gripper_settings import COLLECTION_POLICY


def test_export_refuses_mixed_new_and_legacy_demonstrations(tmp_path):
    new=tmp_path/'new';old=tmp_path/'old';new.mkdir();old.mkdir()
    (new/'metadata.json').write_text(json.dumps(dict(gripper_policy=COLLECTION_POLICY)))
    (old/'metadata.json').write_text('{}')
    assert collection_policy([new],COLLECTION_POLICY)==COLLECTION_POLICY
    assert collection_policy([old])=='legacy_open_transit_v1'
    with pytest.raises(ValueError,match='mix'):collection_policy([new,old])
    with pytest.raises(ValueError,match='Unexpected'):collection_policy([old],COLLECTION_POLICY)
