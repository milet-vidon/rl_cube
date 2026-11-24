"""Host-only parallel form of the final opposed path gate, without changing it.

Call validate_paths(request, phases, target_palm_matrix, workers=...) only with
the exact original request and the planner's `target` matrix. No IK or search is
performed here. The constructor must still describe the unchanged fixed/moving
collision geometry; callers that mutate scene geometry need a new contract.
"""
import hashlib,json,multiprocessing,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

CONTACT_PHASES=('right_close','opposition_hold','opposed_face_turn','opposed_face_hold')
TURN_PHASES=('opposed_face_turn','opposed_face_hold')
PHASES=('right_preshape','right_route','right_approach','right_close','opposition_hold','opposed_face_turn','opposed_face_hold')
_scene=None
_target=None
_names=None

def samples(initial,names,phases):
    """Byte-for-byte formula/order of the original final-loop interpolants."""
    if len(names)!=56 or len(set(names))!=56:raise ValueError('Expected unique 56-joint schema')
    if [p for p,_ in phases]!=list(PHASES):raise ValueError('Unexpected final phase order')
    previous=np.array([initial[n] for n in names]);jobs=[];turn_end=[]
    for phase,path in phases:
        path=np.asarray(path,dtype=float)
        if path.ndim!=2 or path.shape[1]!=len(names) or not len(path) or not np.isfinite(path).all():raise ValueError('Invalid path')
        for step,point in enumerate(path):
            for u in np.linspace(0,1,max(1,int(np.ceil(np.max(np.abs(point-previous))/.01)))+1)[1:]:
                state=(1-u)*previous+u*point
                jobs.append((len(jobs),phase,step,state))
            if phase=='opposed_face_turn':turn_end.append(len(jobs)-1)
            previous=point
    return jobs,turn_end

def _initialize(request,target):
    global _scene,_target,_names
    from opposed_grasp_plan import OppositionScene
    _scene=OppositionScene(request);_target=np.asarray(target);_names=request['joint_names']

def _chunk(jobs):
    result=[]
    for index,phase,step,state in jobs:
        _scene.q=dict(zip(_names,state));_scene.contact=phase in CONTACT_PHASES;_scene.theta=0.
        if phase in TURN_PHASES:
            p=_scene.k.fk_link(_scene.q,'right_palm_link');delta=_scene.C.T@p[:3,:3]@_target[:3,:3].T@_scene.C
            _scene.theta=float(-R.from_matrix(delta).as_rotvec()[2]*180/np.pi)
        valid=bool(_scene.valid(np.array([_scene.q[n] for n in _scene.names])))
        result.append({'index':index,'phase':phase,'step':step,'theta_deg':_scene.theta,'valid':valid,'failure_pair':list(_scene.last_pair) if not valid else None})
    return result

def assemble(rows,turn_end,elapsed,workers):
    rows=sorted(rows,key=lambda r:r['index'])
    if [r['index'] for r in rows]!=list(range(len(rows))):raise ValueError('Incomplete/duplicate predicate results')
    failed=next((r for r in rows if not r['valid']),None)
    # Match the original first failure and effective checks, even if workers
    # already checked later samples. Extra calls cannot promote a failed path.
    error=None if failed is None else f"{failed['phase']} step {failed['step']}: {tuple(failed['failure_pair'])}"
    return {'passed':failed is None,'checks':len(rows) if failed is None else failed['index']+1,'executed_checks':len(rows),'failure':error,'first_failure':failed,'turn_path_angles_deg':[rows[i]['theta_deg'] for i in turn_end] if failed is None else [],'predicate_results':rows,'elapsed_s':elapsed,'workers':workers,'physics_validated':False}

def validate_paths(request,phases,target_palm_matrix,workers=1):
    if not isinstance(workers,int) or not 1<=workers<=32:raise ValueError('Workers must be in [1,32]')
    target=np.asarray(target_palm_matrix,dtype=float)
    if target.shape!=(4,4) or not np.isfinite(target).all() or not np.allclose(target[3],[0,0,0,1]):raise ValueError('Exact planner palm target matrix required')
    started=time.monotonic();jobs,turn_end=samples(request['initial'],request['joint_names'],phases)
    if workers==1:
        _initialize(request,target);rows=_chunk(jobs)
    else:
        chunks=[jobs[i:i+max(1,len(jobs)//(workers*2))] for i in range(0,len(jobs),max(1,len(jobs)//(workers*2)))]
        with multiprocessing.get_context('spawn').Pool(workers,initializer=_initialize,initargs=(request,target)) as pool:
            rows=[row for chunk in pool.map(_chunk,chunks) for row in chunk]
    return assemble(rows,turn_end,time.monotonic()-started,workers)

def require_valid(report):
    if not report['passed']:raise ValueError(report['failure'])
    return report['checks'],report['turn_path_angles_deg']

def input_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def provenance(request,phases,target):
    """Identify exact gate inputs and the code/assets reconstructed by workers."""
    import xml.etree.ElementTree as ET
    root=Path(__file__).resolve().parent;assetroot=root.parent/'revotron_revo3/source/revotron_system';urdf=assetroot/'urdf/revotron_bimanual_revo3_56dof.urdf'
    assets={urdf}
    for mesh in ET.parse(urdf).getroot().findall('.//collision/geometry/mesh'):assets.add((assetroot/'urdf'/mesh.get('filename')).resolve())
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    return {'input_sha256':input_hash({'request':request,'phases':[(p,np.asarray(x).tolist()) for p,x in phases],'target':np.asarray(target).tolist()}),'source_sha256':{n:sha(root/n) for n in ('opposed_path_validation.py','opposed_grasp_plan.py','safe_collision_plan.py','kinematics.py','torso_collision.py','config.json')},'collision_asset_sha256':{str(p.relative_to(root.parent)):sha(p) for p in sorted(assets)},'contract':'unchanged final scene.valid, .01 rad interpolants and exact planner target matrix'}
