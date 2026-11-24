"""Offline 90 mm grasp retarget for the frozen left-hand pickup experiment.

Use robot URDF joint limits, measured pad anchors and actual collision meshes.
All lengths in meters. Results are kinematic references, not physical success.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

from kinematics import RobotKinematics

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'opposition_grasp_v20.json'
URDF=ROOT.parent/'revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf'
FINGERS=('thumb','index','middle','ring','little')
EDGE=.090
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asjson(v):return json.dumps(v,indent=2,allow_nan=False)+'\n'

class Hand:
    def __init__(self,g):
        global coal
        import coal
        self.k=RobotKinematics()
        self.g=g
        self.names={f:[n for n in g['joint_targets'] if n.startswith('left_'+f+'_')] for f in FINGERS}
        # W maps palm vectors into the table-upright, cube-body frame used by
        # FingerArrival.target_cube; c stores cube center in the palm frame.
        self.frame=np.array([[0,0,1],[-1,0,0],[0,-1,0]])@R.from_euler('z',g['frame_roll_deg'],degrees=True).as_matrix()
        self.W=self.frame@np.array(g['cube_rotation_palm']).T
        self.c0=np.array(g['center_palm'])
        self.box={e:coal.Box(e,e,e) for e in (.057,.09)}
        self.ident=coal.Transform3s.Identity()
        self.req=coal.DistanceRequest();self.req.enable_signed_distance=True
        self.meshes={}; self.mesh_hashes={}; self.anchors=g['contact_anchors']
        kin=self.k
        for l in ET.parse(URDF).getroot().findall('link'):
            name=l.get('name')
            if not (name.startswith('left_') or name=='Link7_L'):continue
            meshes=[]
            for col in l.findall('collision'):
                m=col.find('geometry/mesh')
                if m is None:continue
                path=(URDF.parent/m.get('filename')).resolve()
                s=coal.MeshLoader().load(str(path),np.fromstring(m.get('scale','1 1 1'),sep=' '));s.buildConvexRepresentation(True);s=s.convex
                arr=np.asarray(s.points()).copy()
                o=col.find('origin')
                tf=np.eye(4)
                if o is not None:
                    tf[:3,:3]=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
                    tf[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                meshes.append((s,tf,arr@tf[:3,:3].T+tf[:3,3]));self.mesh_hashes[str(path.relative_to(URDF.parent.parent))]=sha(path)
            if meshes:self.meshes[name]=meshes
        self.group={f:[n for n in self.meshes if n.startswith('left_'+f+'_')] for f in FINGERS}
        self.group['palm']=[n for n in self.meshes if not any(n in l for l in self.group.values())]
        self.inv_palm=np.linalg.inv(kin.fk_link({},'left_palm_link'))
        # Factor out the fixed chain from root to wrist. Necessary for cheap path sampling.
        self.base={};self.chains={}
        for group,links in self.group.items():
            needed=set(links)
            if group in FINGERS:needed.add(self.anchors[group]['link'])
            chain=[]
            for name,par,ch,o,a in kin.data:
                if name not in self.names.get(group,[]):continue
                chain.append((name,par,ch,o,a));needed.update((par,ch))
            for e in chain:
                p=e[1]
                while p in kin.by_child and p not in {x[2] for x in chain}:
                    j=kin.by_child[p]
                    if j[4] is not None:break
                    p=j[1];needed.add(p)
            allnodes={}
            for link in needed:
                cn=[];cursor=link
                while cursor in kin.by_child:
                    j=kin.by_child[cursor]
                    if j[0] in self.names.get(group,[]):cn.append(j)
                    elif j[4] is None:cn.append(j)
                    else:break
                    cursor=j[1]
                self.base[cursor]=self.inv_palm@kin.fk_link({},cursor)
                for j in cn:allnodes[j[2]]=j
            # kinematic order independent of URDF declaration order.
            todo=dict(allnodes);ordered=[];done=set(self.base)-set(allnodes)
            while todo:
                ready=[j for j in todo.values() if j[1] in done]
                if not ready:raise ValueError('Cyclic finger FK')
                for j in ready:ordered.append(j);done.add(j[2]);del todo[j[2]]
            self.chains[group]=ordered

    def fk(self,f,q):
        out=self.base.copy()
        for n,p,ch,o,a in self.chains[f]:
            t=out[p]@o
            if a is not None:t[:3,:3]=t[:3,:3]@R.from_rotvec(a*q[n]).as_matrix()
            out[ch]=t
        return out

    def features(self,f,q,c,edge):
        frames=self.fk(f,q)
        distances={};zmin=1.;sgn=-1. if f=='thumb' else 1.
        for n in self.group[f]:
            t=frames[n];wr=self.W@t[:3,:3];wp=self.W@(t[:3,3]-c)
            for m,o,vertices in self.meshes[n]:
                rot=wr@o[:3,:3];pos=wp+wr@o[:3,3]
                dist=float(coal.distance(m,coal.Transform3s(rot,pos),self.box[edge],self.ident,self.req,coal.DistanceResult()))
                distances[n]=min(dist,distances.get(n,1.))
                zmin=min(zmin,float((vertices@wr[2]+wp[2]).min()+edge/2))
        if f=='palm':return dict(min_dist=min(distances.values()),min_table=zmin)
        a=self.anchors[f];t=frames[a['link']]
        pt=self.W@(t[:3,3]+t[:3,:3]@a['point_local']-c)
        return dict(anchor=pt,face_gap=sgn*pt[0]-edge/2,dist=distances[a['link']],other_dist=min(d for n,d in distances.items() if 'DIP' not in n),
                    min_dist=min(distances.values()),min_table=zmin)

    def cross_finger_gap(self,q):
        shapes={}
        for f in FINGERS:
            fk=self.fk(f,q);groups=[]
            for n in self.group[f]:
                for shape,o,verts in self.meshes[n]:
                    tf=fk[n]; pose=tf@o
                    ctr=verts.mean(axis=0);radius=float(np.linalg.norm(verts-ctr,axis=1).max())
                    groups.append((shape,coal.Transform3s(pose[:3,:3],pose[:3,3]),tf[:3,:3]@ctr+tf[:3,3],radius,n))
            shapes[f]=groups
        best=1.;pair=None
        for i,f in enumerate(FINGERS):
            for f2 in FINGERS[i+1:]:
                for m,t,c,r,n in shapes[f]:
                    for m2,t2,c2,r2,n2 in shapes[f2]:
                        if np.linalg.norm(c-c2)-r-r2>best:continue
                        d=float(coal.distance(m,t,m2,t2,self.req,coal.DistanceResult()))
                        if d<best:best=d;pair=[n,n2]
        return dict(min_gap_mm=1000*best,pair=pair)

    def measure(self,q,c,edge):
        f={name:self.features(name,q,c,edge) for name in FINGERS}
        palm=self.features('palm',q,c,edge)
        span=[v['anchor'][0]-f['thumb']['anchor'][0] for n,v in f.items() if n!='thumb']
        return dict(span_min_mm=float(min(span)*1000),span_max_mm=float(max(span)*1000),
                    table_min_mm=1000*min(palm['min_table'],*(v['min_table'] for v in f.values())),
                    palm_cube_gap_mm=1000*palm['min_dist'],phalanges_min_mm=1000*min(v['other_dist'] for v in f.values()),
                    fingers={n:dict(anchor_cube_mm=(v['anchor']*1000).tolist(),anchor_face_gap_mm=1000*v['face_gap'],
                                    distal_mesh_gap_mm=1000*v['dist'],all_mesh_gap_mm=1000*v['min_dist'],
                                    table_min_mm=1000*v['min_table']) for n,v in f.items()})

    def fit(self,opened,cstart,reference,fixed_center=False):
        c0=self.c0
        groups=[self.names[f] for f in FINGERS]; names=sum(groups,[])
        starts=np.cumsum([0]+[len(a) for a in groups]);ys0=np.array([reference[n] for n in names])
        margins=.021
        limits=np.array([self.k.limits[n] for n in names])
        center_vars=0 if fixed_center else 3
        x0=np.r_[[] if fixed_center else cstart,ys0]
        low=np.r_[[] if fixed_center else c0-.03,limits[:,0]+margins];high=np.r_[[] if fixed_center else c0+.03,limits[:,1]-margins]
        x0=np.clip(x0,low+1e-7,high-1e-7)
        target_inset=.0005
        def residual(x):
            c=cstart if fixed_center else x[:3]; y=x[center_vars:];q=dict(zip(names,y))
            parts=[]; i=0
            for f in FINGERS:
                v=self.features(f,q,c,EDGE);a=v['anchor']
                yz=np.array(self.anchors[f]['target_cube'])[1:] *1.15
                # The original empirical local anchors are often slightly IN the
                # contact hull. Prioritize an actual mesh-to-face gap over the
                # idealized anchor. Table and proximal links must stay clear.
                gap=.008 if opened else -target_inset
                rr=np.r_[200*(v['dist']-gap), 75*(v['face_gap']-gap),20*(a[1:]-yz),
                         300*min(0,v['min_table']-.004),220*min(0,v['other_dist']-(.004 if opened else .0008)),
                         300*min(0,v['min_dist']-(.005 if opened else -.0014)),
                         100*np.maximum(np.abs(a[1:])-.039,0),.023*(y[starts[i]:starts[i+1]]-ys0[starts[i]:starts[i+1]])]
                parts.append(rr);i+=1
            palm=self.features('palm',{},c,EDGE)
            parts.append(np.r_[220*min(0,palm['min_dist']-.002),200*min(0,palm['min_table']-.005),7*(c-c0)])
            return np.concatenate(parts)
        m=len(residual(x0));pat=lil_matrix((m,len(x0)),dtype=int)
        if not fixed_center:pat[:,:3]=1
        start=0
        for i,f in enumerate(FINGERS):
            n=9+len(groups[i]);pat[start:start+n,center_vars+starts[i]:center_vars+starts[i+1]]=1;start+=n
        assert start+5==m
        fit=least_squares(residual,x0,bounds=(low,high),jac_sparsity=pat.tocsr(),max_nfev=100,ftol=2e-5,xtol=1e-6,gtol=2e-5,x_scale='jac')
        c=cstart if fixed_center else fit.x[:3]
        return c,dict(zip(names,fit.x[center_vars:].tolist())),dict(cost=float(fit.cost),nfev=fit.nfev)

def build():
    g=json.loads(SOURCE.read_text());h=Hand(g)
    cnew,q,fit=h.fit(False,h.c0,g['joint_targets'])
    cop,opened,ofit=h.fit(True,cnew,q,True)
    old_closed=h.measure(g['joint_targets'],h.c0,.057);old_open=h.measure(g['open_joint_targets'],h.c0,.057)
    unadapt=h.measure(g['open_joint_targets'],h.c0,EDGE)
    closed=h.measure(q,cnew,EDGE);op=h.measure(opened,cop,EDGE)
    g2=deepcopy(g);g2.update(center_palm=cnew.tolist(),joint_targets=q,open_joint_targets=opened,
                             kind='90mm_left_grasp_retarget_kinematic',physics_verified=False)
    for f in FINGERS:
        # Use the new measured anchor's tangential location, with a 1.5mm
        # virtual arrival target inside its new face, as in the baseline.
        a=np.array(closed['fingers'][f]['anchor_cube_mm'])/1000
        g2['face_points'][f]=(h.frame.T@a).tolist()
        a[0]=(-1 if f=='thumb' else 1)*(.045-.0015)
        g2['contact_anchors'][f]['target_cube']=a.tolist()
    closure_q=[{n:(1-u)*opened[n]+u*q[n] for n in q} for u in np.linspace(0,1,21)]
    cross=min((h.cross_finger_gap(x) for x in closure_q),key=lambda v:v['min_gap_mm'])
    path=[h.measure({n:(1-u)*opened[n]+u*q[n] for n in q},cnew,EDGE) for u in np.linspace(0,1,21)]
    limits=h.k.limits
    if not (all(limits[n][0] < state[n] < limits[n][1] for state in (q,opened) for n in q)
            and op['span_min_mm'] > 100
            and min(x['all_mesh_gap_mm'] for x in op['fingers'].values()) > 2
            and all(-2 < x['distal_mesh_gap_mm'] < .2 for x in closed['fingers'].values())
            and min(v['table_min_mm'] for v in path)>2 and min(v['phalanges_min_mm'] for v in path) > 0
            and min(v['palm_cube_gap_mm'] for v in path)>2 and cross['min_gap_mm']>1):
        raise ValueError('Rejected left-hand retarget: insufficient physical clearance or reach')
    for stale in ('collision_distance_min_m','inter_hand_clearance_m','preshape_path_clearance_m','closing_path_clearance_m','elapsed_s','optimization_evaluations'):
        g2.pop(stale,None)
    g2['table_clearance_min_m']=min(v['table_min_mm'] for v in path)/1000
    g2['cube_contact_distances_m']={f:v['distal_mesh_gap_mm']/1000 for f,v in closed['fingers'].items()}
    return g2,dict(schema='round15_size_retarget_geometry_v1',source_grasp_sha256=sha(SOURCE),urdf_sha256=sha(URDF),
                   mesh_sha256=h.mesh_hashes,edge_m=EDGE,source_edge_m=.057,scope='left fingers, palm XYZ, quasi-static box envelope; geometry only, not simulated evidence',
                   objective='widen lateral opening, solve independently for open and closed mesh gaps, keep hand above table; retain yaw, right hand, gain and safety limits',
                   source_center_palm_m=h.c0.tolist(),retarget_center_palm_m=cnew.tolist(),closed_optimization=fit,open_optimization=ofit,
                   original_57_open=old_open,original_57_closed=old_closed,original_90_open=unadapt,retarget_90_open=op,retarget_90_closed=closed,
                   closure_min_cross_finger=cross,closure_samples=len(path),closure_table_min_mm=min(v['table_min_mm'] for v in path),
                   closure_proximal_gap_min_mm=min(v['phalanges_min_mm'] for v in path),
                   closure_dipped_mesh_min_mm=min(z['all_mesh_gap_mm'] for v in path for z in v['fingers'].values()))

def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    g,r=build()
    with (out/'grasp90.json').open('x') as f:f.write(asjson(g))
    r['grasp_sha256']=sha(out/'grasp90.json')
    with (out/'geometry_retarget.json').open('x') as f:f.write(asjson(r))
    return r

def arm_preflight(out):
    """Independent CPU IK/time parameterization at the two nominal yaw/reset poses."""
    from opposition_teacher import OppositionPlanner
    out=Path(out);g=out/'grasp90.json';h=json.loads((out/'geometry_retarget.json').read_text())
    if sha(g)!=h['grasp_sha256']:raise ValueError('Changed grasp')
    records=[]
    for seed in (21611,21612):
        p=OppositionPlanner(g);cfg=p.cfg;rng=np.random.default_rng(seed)
        pos=np.array(cfg['cube_xyz']);pos[:2]+=rng.uniform(-.015,.015,2);pos[2]=cfg['table_z']+EDGE/2
        yaw=float(rng.uniform(-.15,.15));ph=p.plan(pos.copy(),yaw)
        if [n for n,v in ph][:6]!=['ready','preshape','lower','approach','squeeze','primary_lift']:raise ValueError('Wrong pickup sequence')
        records.append(dict(seed=seed,nominal_center_world_m=pos.tolist(),yaw_rad=yaw,planned_waypoints={n:len(v) for n,v in ph}))
    receipt=dict(kind='offline_arm_ik_not_simulation',grasp_sha256=sha(g),planner_sha256=sha(ROOT/'opposition_teacher.py'),
                 generator_sha256=sha(__file__),nominal_upright_pose=True,records=records)
    with (out/'arm_ik_preflight.json').open('x') as f:f.write(asjson(receipt))
    return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--arm-preflight',action='store_true');a=ap.parse_args()
    if a.arm_preflight:
        print(asjson(arm_preflight(a.output_dir)));raise SystemExit(0)
    receipt=generate(a.output_dir)
    print(json.dumps({k:receipt[k] for k in ('grasp_sha256','closure_table_min_mm','retarget_90_open','retarget_90_closed')},indent=2))
