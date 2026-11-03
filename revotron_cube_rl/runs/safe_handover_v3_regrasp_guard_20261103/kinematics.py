"""URDF-only FK/IK utilities for setup and independent controller feasibility checks."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

class RobotKinematics:
    def __init__(self):
        path=Path(__file__).parent.parent/'revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf'
        root=ET.parse(path).getroot();self.joints=root.findall('joint')
        self.roots={l.get('name') for l in root.findall('link')}-{j.find('child').get('link') for j in self.joints}
        self.data=[];self.limits={}
        for j in self.joints:
            o=j.find('origin');t=np.eye(4)
            if o is not None:
                t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if j.get('type')!='fixed' else None
            self.data.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),t,axis))
            if axis is not None:self.limits[j.get('name')]=[float(j.find('limit').get(x)) for x in ('lower','upper')]
        self.by_child={item[2]:item for item in self.data}
        # Keep IK inside the same software arm envelope used by Isaac control.
        cfg_path=Path(__file__).resolve().parent/'config.json'
        cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
        margin=float(cfg.get('arm_soft_limit_margin_rad',0.15))
        explicit=cfg.get('arm_soft_limits',{})
        for n in list(self.limits):
            if n.startswith('Joint') and n.endswith(('_L','_R')):
                lo,hi=self.limits[n]
                soft=list(map(float,explicit[n])) if n in explicit else [lo+margin,hi-margin]
                self.limits[n]=[max(lo,soft[0]),min(hi,soft[1])]
    def fk_link(self,q,link):
        chain=[]
        while link in self.by_child:
            item=self.by_child[link];chain.append(item);link=item[1]
        result=np.eye(4)
        for n,p,c,origin,axis in reversed(chain):
            t=origin.copy()
            if axis is not None:t[:3,:3]=t[:3,:3]@R.from_rotvec(axis*q.get(n,0.)).as_matrix()
            result=result@t
        return result
    def fk(self,q):
        ts={n:np.eye(4) for n in self.roots};pending=list(self.data)
        while pending:
            for item in pending[:]:
                n,p,c,origin,axis=item
                if p not in ts:continue
                t=origin.copy()
                if axis is not None:t[:3,:3]=t[:3,:3]@R.from_rotvec(axis*q.get(n,0.)).as_matrix()
                ts[c]=ts[p]@t;pending.remove(item)
        return ts
    def fit_arm(self,q,side,palm_position,rotation):
        names=[f'Joint{i}_{"L" if side=="left" else "R"}' for i in range(1,8)]
        limits=np.array([self.limits[n] for n in names]);q0=np.clip([q[n] for n in names],limits[:,0]+1e-4,limits[:,1]-1e-4)
        def residual(x):
            t=self.fk_link(q|dict(zip(names,x)),side+'_palm_link')
            return np.r_[5*(t[:3,3]-palm_position),R.from_matrix(rotation@t[:3,:3].T).as_rotvec(),.001*(x-q0)]
        fit=least_squares(residual,q0,bounds=(limits[:,0]+1e-4,limits[:,1]-1e-4),max_nfev=180)
        target=q|dict(zip(names,fit.x.tolist()))
        err=np.linalg.norm(self.fk_link(target,side+'_palm_link')[:3,3]-palm_position)
        if err>.003:raise ValueError(f'IK position error: {err}')
        return target
