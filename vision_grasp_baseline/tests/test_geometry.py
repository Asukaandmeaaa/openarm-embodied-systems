"""Geometry checks that matter before a camera target can command a robot."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
import pytest
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vision_grasp_baseline.perception import camera_transform,fit_plane,contour,rectangle,cube_pixels

ROOT=Path(__file__).resolve().parents[2]


def test_color_depth_offset_agrees_with_sdk():
    m=json.loads((ROOT/'vision_grasp_baseline/runs/current_rgbd_20261021/capture.json').read_text())
    path=ROOT/'sim/scenes/act_fixed_20261015/source_snapshot/global_camera_20261015.calib'
    cal=yaml.safe_load(path.read_text());q=cal['transform']['rotation'];t=cal['transform']['translation']
    ex=rs.extrinsics();ex.rotation=m['depth_to_color_extrinsics']['rotation_column_major'];ex.translation=m['depth_to_color_extrinsics']['translation_m']
    point_depth=[.13,-.07,.61]
    point_color=rs.rs2_transform_point_to_point(ex,point_depth)
    expected=Rotation.from_quat([q[k] for k in 'xyzw']).apply([point_depth[2],-point_depth[0],-point_depth[1]])+np.array([t[k] for k in 'xyz'])
    actual=(camera_transform(path,m) @ np.r_[point_color,1])[:3]
    assert np.allclose(expected,actual,atol=1e-7)


def test_support_plane_survives_depth_outliers():
    rng=np.random.default_rng(21)
    xy=rng.uniform(-.3,.3,(2000,2));z=.27+.02*xy[:,0]+rng.normal(0,.0007,2000)
    plane=np.column_stack((xy,z));outliers=rng.uniform(-.4,.4,(400,3))
    n,d,residual=fit_plane(np.concatenate((plane,outliers)))
    expected=np.array([-.02,0,1]);expected/=np.linalg.norm(expected)
    assert n @ expected > .9999
    assert abs(d+.27/np.sqrt(1+.02**2))<.001
    assert np.percentile(residual,95)<.003


def test_multiple_objects_are_not_silently_selected():
    mask=np.zeros((120,160),dtype='uint8')
    cv2.rectangle(mask,(10,10),(45,45),255,-1);cv2.rectangle(mask,(80,60),(115,95),255,-1)
    with pytest.raises(ValueError,match='got 2'):
        contour(mask,500)


def test_square_axis_does_not_jump_by_ninety_degrees():
    grid=np.stack(np.meshgrid(np.linspace(-.02,.02,40),np.linspace(-.02,.02,40)),axis=-1).reshape(-1,2)
    angles=[]
    for yaw in [.2,.2+np.pi/2,.2+np.pi]:
        xy=grid @ Rotation.from_euler('z',yaw).as_matrix()[:2,:2].T+[.33,-.18]
        center,size,angle=rectangle(np.column_stack((xy,np.zeros(len(xy)))))
        assert np.linalg.norm(center-[.33,-.18])<1e-5
        angles.append(angle)
    assert np.ptp(angles)<1e-5


def test_wrist_cable_does_not_split_one_cube_or_supply_its_depth():
    image=np.zeros((150,180,3),dtype='uint8')
    image[35:110,45:125]=[110,240,180]
    image[30:120,80:87]=[0,0,15]
    mask,_=cube_pixels(image)
    assert mask[60,60] and mask[60,110]
    assert not mask[:,80:87].any()
