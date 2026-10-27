"""Render an offline TCP path and robot FK snapshots; never connects to ROS."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.run
    plan=json.loads((root/'original_bin_waypoints_02.json').read_text())
    frames=json.loads((root/'waypoint_link_frames.json').read_text())
    base=Path(__file__).resolve().parent
    scene=json.loads((base/'runs/cycle_repeat_20261021_01/zero_scene/scene.json').read_text())
    correction=np.array(json.loads((base/'config/session_registration_20261021.json').read_text())['translation_world_m'])
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(13,6.5),layout='constrained');ax=fig.add_subplot(121,projection='3d');top=fig.add_subplot(122)
    targets=np.array([s['target_tcp_world'] for s in plan['stages']]);xyz=targets[:,:3,3]
    ax.plot(*xyz[:7].T,color='#0072B2',marker='o',label='抬高、调整姿态、四段平移、下放')
    ax.plot(*xyz[6:].T,color='#D55E00',linestyle='--',marker='.',label='上撤、四段原路返回、恢复姿态')
    for i,color in [(0,'#56B4E9'),(5,'#009E73'),(6,'#CC79A7')]:
        positions=frames[i]['positions']
        links=[f'openarm_right_link{j}' for j in range(8)]+['openarm_right_ee_base_link','openarm_right_grasp_frame']
        points=np.array([positions[k] for k in links if k in positions])
        if len(points)>1:ax.plot(*points.T,color=color,alpha=.6,linewidth=2)
    b=scene['measured']['bin'];center=np.array(b['center_xy'])+correction[:2]
    half=np.asarray(b['outer_size_xy'])/2;r=Rotation.from_euler('z',b['yaw']).as_matrix()[:2,:2]
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*half
    corners=corners@r.T+center
    rim=b['support_z']+correction[2]+b['rim_height']
    ax.plot(corners[:,0],corners[:,1],np.full(5,rim),color='black',linewidth=2)
    ax.set(xlabel='世界 X / m',ylabel='世界 Y / m',zlabel='世界 Z / m',title='机械臂关键姿态与 TCP 路径（模型）')
    ax.view_init(elev=28,azim=-55);ax.legend(loc='upper left',fontsize=8)
    top.add_patch(Polygon(np.column_stack((-corners[:-1,1],corners[:-1,0]))*100,
                          facecolor='#555555',edgecolor='black',alpha=.25,label='当前黑盒：位置不变'))
    forward=xyz[1:6];top.plot(-forward[:,1]*100,forward[:,0]*100,'o-',color='#0072B2',label='四段等高平移')
    for i,point in enumerate(forward):
        top.annotate('起点' if i==0 else (f'中间点{i}' if i<4 else '盒上方'),
                     (-point[1]*100,point[0]*100),xytext=(6,8),textcoords='offset points',fontsize=10)
    cube=np.asarray(scene['measured']['cube']['center'])+correction
    top.scatter([-cube[1]*100],[cube[0]*100],marker='s',s=90,color='#0072B2',label='原拾取位置')
    top.set(xlabel='−世界 Y / cm（向右）',ylabel='世界 X / cm',title='俯视：3 个中间点 + 盒上方终点')
    top.set_aspect('equal',adjustable='box');top.set_xlim(-13,23);top.set_ylim(30,49)
    top.grid(alpha=.25);top.legend(loc='best',fontsize=9)
    fig.suptitle('原场景中间点方案：13 段 / 2571 个模型状态检查 / 仅离线规划',fontsize=16)
    fig.supxlabel('平移高度约 43.7 cm；在拾取位置上方调整约 20°；桌面模型仍须与实物核验。',fontsize=11)
    out=root/'waypoint_plan_preview.png';fig.savefig(out,dpi=160);plt.close(fig)
    print(out)


if __name__=='__main__':main()
