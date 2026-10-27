"""Compare live-FK collision-mesh projections to the stationary RGB-D arm.

This diagnostic never corrects calibration or commands a robot. Collision
meshes and depth-edge errors limit accuracy; a single pose cannot certify TF.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation


def stl_triangles(path):
    data=Path(path).read_bytes();count=int.from_bytes(data[80:84],'little')
    if len(data)!=84+50*count:
        raise ValueError('Expected binary STL: '+str(path))
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    return np.frombuffer(data,offset=84,count=count,dtype=dtype)['vertices'].astype(float)


def sample_triangles(triangles,count,rng):
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    ts=triangles[rng.choice(len(triangles),count,p=areas/areas.sum())]
    uv=rng.random((count,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
    return ts[:,0]+uv[:,:1]*(ts[:,1]-ts[:,0])+uv[:,1:]*(ts[:,2]-ts[:,0])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();p=a.run
    live=json.loads((p/'live_fk/state.json').read_text());plan=json.loads((p/'plan.json').read_text())
    metadata=json.loads((p/'after/capture.json').read_text());i=metadata['color_intrinsics']
    cw=np.linalg.inv(np.array(plan['scene']['world_from_color_optical']))
    root=ET.parse(p/'live_fk/robot.urdf').getroot();rng=np.random.default_rng(21)
    package=Path('/home/lushilin/openarm_ws/install/openarm_description/share/openarm_description')
    depths=np.full((480,640),np.inf);cloud=[]
    for link in root.findall('link'):
        name=link.attrib['name']
        if name not in live['world_from_link']:continue
        for collision in link.findall('collision'):
            mesh=collision.find('geometry/mesh')
            if mesh is None:continue
            filename=mesh.attrib['filename']
            if not filename.startswith('package://openarm_description/'):
                raise ValueError('Unexpected mesh package')
            triangles=stl_triangles(package/filename.removeprefix('package://openarm_description/'))
            points=sample_triangles(triangles,60000,rng)
            points*=np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
            origin=collision.find('origin');visual=np.eye(4)
            if origin is not None:
                visual[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
                visual[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
            t=cw @ np.array(live['world_from_link'][name]) @ visual
            cp=points @ t[:3,:3].T+t[:3,3];cp=cp[cp[:,2]>.05];cloud.append(cp)
            u=np.rint(cp[:,0]/cp[:,2]*i['fx']+i['ppx']).astype(int)
            v=np.rint(cp[:,1]/cp[:,2]*i['fy']+i['ppy']).astype(int)
            valid=(u>=0)&(u<640)&(v>=0)&(v<480)
            np.minimum.at(depths,(v[valid],u[valid]),cp[valid,2])
    depths=minimum_filter(depths,size=3)
    image=cv2.imread(str(p/'after/color_009.png'));after=np.load(p/'after/depth_009.npy')*metadata['depth_scale_m']
    # Use the initial empty-arm view for all poses, so the old arm position
    # cannot erase part of the new arm's foreground mask.
    empty=Path(__file__).resolve().parent/'runs/hover_20261021_01/before_v2/depth_009.npy'
    before=np.load(empty)*metadata['depth_scale_m']
    observed=(before-after>.04)&(after>.1)&(before>.1)
    observed=cv2.morphologyEx(observed.astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),'uint8'))>0
    # Background depth flicker is not the robot. Depth holes can split the
    # real arm into several components, so retain all substantial components
    # overlapping the projected robot neighborhood (including the top edge).
    count,labels,stats,_=cv2.connectedComponentsWithStats(observed.astype('uint8'))
    neighborhood=cv2.dilate(np.isfinite(depths).astype('uint8'),np.ones((41,41),'uint8'))>0
    candidates=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>1500
                and np.mean(neighborhood[labels==k])>.5]
    if not candidates:
        raise ValueError('No unambiguous robot foreground components')
    observed=np.isin(labels,candidates)&neighborhood
    predicted=np.isfinite(depths)&(depths<before-.04)
    interior=cv2.erode((observed&predicted).astype('uint8'),np.ones((5,5),'uint8'))>0
    residual=after[interior]-depths[interior]
    overlay=image.copy();overlay[predicted]=(overlay[predicted]*.5+np.array([255,255,0])*.5).astype('uint8')
    actual_contours,_=cv2.findContours(observed.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,actual_contours,-1,(0,0,255),1)
    cv2.putText(overlay,'CYAN: FK model; RED: observed arm boundary',(5,460),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    cv2.imwrite(str(p/'after/model_overlay.png'),overlay)
    union=(observed|predicted).sum();iou=float((observed&predicted).sum()/max(union,1))
    report=dict(pose_count=1,predicted_pixels=int(predicted.sum()),observed_pixels=int(observed.sum()),
                mask_iou=iou,interior_compared_pixels=len(residual),
                depth_residual_median_m=float(np.median(residual)) if len(residual) else None,
                depth_residual_p95_abs_m=float(np.percentile(abs(residual),95)) if len(residual) else None,
                calibration_verified=False,
                limitation='Single-pose, partially cropped collision-mesh projection; not a calibration solve.')
    (p/'projection_check.json').write_text(json.dumps(report,indent=2)+'\n')
    cp=np.concatenate(cloud)
    u=np.rint(cp[:,0]/cp[:,2]*i['fx']+i['ppx']).astype(int)
    v=np.rint(cp[:,1]/cp[:,2]*i['fy']+i['ppy']).astype(int)
    keep=(u>=0)&(u<640)&(v>=0)&(v<480)
    cp,u,v=cp[keep],u[keep],v[keep]
    visible=cp[:,2]<depths[v,u]+.003
    v_ob,u_ob=np.where(observed)
    d_ob=after[v_ob,u_ob]
    observed_cloud=np.column_stack(((u_ob-i['ppx'])*d_ob/i['fx'],(v_ob-i['ppy'])*d_ob/i['fy'],d_ob))
    np.savez_compressed(p/'projection_clouds.npz',model_camera=cp[visible],observed_camera=observed_cloud)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
