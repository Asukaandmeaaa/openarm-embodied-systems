"""URDF chain FK for comparing fresh measured and motor-command TCP poses."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


class ChainFK:
    def __init__(self,path,tip='openarm_right_grasp_frame'):
        root=ET.parse(path).getroot();parents={}
        for joint in root.findall('joint'):
            o=joint.find('origin');t=np.eye(4)
            if o is not None:
                t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            axis=joint.find('axis');v=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')
            parents[joint.find('child').get('link')]=(joint.find('parent').get('link'),joint.get('name'),joint.get('type'),t,v)
        self.chain=[]
        while tip!='world':
            if tip not in parents:raise ValueError('URDF does not connect TCP to world')
            parent,name,kind,t,v=parents[tip];self.chain.append((name,kind,t,v));tip=parent
        self.chain.reverse()

    def pose(self,values):
        t=np.eye(4)
        for name,kind,origin,axis in self.chain:
            t=t@origin
            if kind in ('revolute','continuous'):
                r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(axis/np.linalg.norm(axis)*values[name]).as_matrix();t=t@r
            elif kind=='prismatic':
                r=np.eye(4);r[:3,3]=axis*values[name];t=t@r
            elif kind!='fixed':raise ValueError('Unsupported URDF joint '+kind)
        return t
