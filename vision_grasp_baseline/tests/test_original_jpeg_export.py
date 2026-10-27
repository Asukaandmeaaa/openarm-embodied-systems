import hashlib
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from vision_grasp_baseline.collection_io import ORDER,CAMERAS
from vision_grasp_baseline.export_act import save_original_episode


def test_original_images_are_embedded_byte_exact_and_sources_untouched(tmp_path):
    pytest.importorskip('lerobot')
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    source=tmp_path/'source';source.mkdir()
    streams={};original={}
    for k,key in enumerate(CAMERAS):
        streams[key]=[]
        for i in range(3):
            file=source/f'{key}_{i}.jpg'
            image=np.full((480,640,3),[20+i,50+k,150-i],dtype=np.uint8)
            assert cv2.imwrite(str(file),image)
            streams[key].append(dict(path=file.name));original[file]=file.read_bytes()
    q={'joint_observation':np.zeros((3,16),np.float32),
       'demonstration_action':np.arange(48,dtype=np.float32).reshape(3,16)*.001+.02}
    ids={k:np.arange(3) for k in (*CAMERAS,*q)}
    features={key:dict(dtype='float32',shape=(16,),names=ORDER) for key in ('observation.state','action')}
    features.update({f'observation.images.{k}':dict(dtype='image',shape=(480,640,3),names=['height','width','channels']) for k in CAMERAS})
    root=tmp_path/'dataset'
    ds=LeRobotDataset.create(repo_id='local/test',root=root,fps=30,features=features,
                            robot_type='openarm',use_videos=False)
    save_original_episode(ds,source,(streams,q,ids,dict(frame_count=3)),'full closed task')
    assert ds.meta.total_episodes==1
    # A second episode proves this public API advances episode/global indices.
    save_original_episode(ds,source,(streams,q,ids,dict(frame_count=3)),'full closed task')
    ds.finalize()
    table=pq.read_table(sorted((root/'data').rglob('*.parquet')))
    assert table['episode_index'].to_pylist()==[0]*3+[1]*3
    assert table['timestamp'].to_pylist()==pytest.approx([0.,1/30,2/30]*2)
    assert np.allclose(np.array(table['action'].to_pylist()),np.tile(q['demonstration_action'],(2,1)))
    for key in CAMERAS:
        rows=table[f'observation.images.{key}'].to_pylist()
        for i,row in enumerate(rows):
            file=source/streams[key][i%3]['path']
            assert row['bytes']==original[file]
            assert Image.open(io.BytesIO(row['bytes'])).size==(640,480)
    assert all(f.read_bytes()==data for f,data in original.items())
    # Embedded data must remain usable without source files at read time.
    source.rename(tmp_path/'moved_source')
    loaded=LeRobotDataset('local/test',root=root,return_uint8=True)
    assert loaded[0]['observation.images.ceiling'].shape==(3,480,640)
    from vision_grasp_baseline.selected_column_reader import enable_selected_columns
    import torch
    from torch.utils.data import DataLoader
    windows=LeRobotDataset('local/test',root=root,return_uint8=True,
                          delta_timestamps={'action':[i/30 for i in range(50)]})
    baseline=[windows[i] for i in (0,2,3,5)]
    enable_selected_columns(windows)
    for i,original in zip((0,2,3,5),baseline):
        actual=windows[i]
        assert actual.keys()==original.keys()
        for key,value in original.items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,actual[key]),key
            else:assert value==actual[key]
    # Spawn is the production worker mode, including reader serialization.
    loader=DataLoader(windows,batch_size=2,num_workers=2,multiprocessing_context='spawn',prefetch_factor=1)
    for batch in loader:
        assert batch['action'].shape==(2,50,16)
    subset=LeRobotDataset('local/test',root=root,episodes=[1],return_uint8=True,
                         delta_timestamps={'action':[i/30 for i in range(50)]})
    baseline=[subset[i] for i in (0,2)]
    enable_selected_columns(subset)
    for i,original in zip((0,2),baseline):
        for key in ('action','action_is_pad','observation.state'):
            assert torch.equal(subset[i][key],original[key])
