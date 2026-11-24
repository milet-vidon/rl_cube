"""Offline compare original final predicates to multiprocessing on saved plans."""
from pathlib import Path
import argparse,json,time,hashlib,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from opposed_grasp_plan import OppositionScene
from opposed_path_validation import validate_paths,input_hash

def original(request,phases,target):
    scene=OppositionScene(request);k=scene.k;names=request['joint_names'];arm=scene.names
    previous=np.array([request['initial'][n] for n in names]);rows=[];angles=[]
    started=time.monotonic()
    for phase,path in phases:
        for step,point in enumerate(np.asarray(path)):
            for u in np.linspace(0,1,max(1,int(np.ceil(np.max(np.abs(point-previous))/.01)))+1)[1:]:
                state=(1-u)*previous+u*point;scene.q=dict(zip(names,state));scene.contact=phase in ('right_close','opposition_hold','opposed_face_turn','opposed_face_hold');scene.theta=0.
                if phase in ('opposed_face_turn','opposed_face_hold'):
                    p=k.fk_link(scene.q,'right_palm_link');delta=scene.C.T@p[:3,:3]@target[:3,:3].T@scene.C;scene.theta=float(-R.from_matrix(delta).as_rotvec()[2]*180/np.pi)
                passed=bool(scene.valid(np.array([scene.q[n] for n in arm])))
                rows.append({'index':len(rows),'phase':phase,'step':step,'theta_deg':scene.theta,'valid':passed,'failure_pair':list(scene.last_pair) if not passed else None})
            if phase=='opposed_face_turn':angles.append(scene.theta)
            previous=point
    return rows,angles,time.monotonic()-started

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    results=[]
    root=Path(__file__).resolve().parent;assetroot=root.parent/'revotron_revo3/source/revotron_system';urdf=assetroot/'urdf/revotron_bimanual_revo3_56dof.urdf';assetpaths={urdf}
    for m in ET.parse(urdf).getroot().findall('.//collision/geometry/mesh'):assetpaths.add((assetroot/'urdf'/m.get('filename')).resolve())
    sources={n:sha(root/n) for n in ('opposed_path_validation.py','opposed_grasp_plan.py','safe_collision_plan.py','kinematics.py','torso_collision.py','benchmark_opposed_path_validation.py')}
    for run in a.run:
        manifest=json.loads((run/'artifact_manifest.json').read_text())
        for n in ('opposition_plan.json','opposition_request.json','experiment.json'):
            if sha(run/n)!=manifest['sha256'][n]:raise ValueError('Run artifact hash mismatch: '+n)
        experiment=json.loads((run/'experiment.json').read_text())
        args=experiment.get('arguments',experiment.get('args',experiment))
        # Saved `target` is absent from old plan JSON. For the audited joint
        # refiner ONLY, original code explicitly assigns target=fk(closed q).
        if args.get('opposition_planner')!='joint':raise ValueError('Cannot reconstruct exact target for non-joint legacy plan')
        request=json.loads((run/'opposition_request.json').read_text());plan=json.loads((run/'opposition_plan.json').read_text())
        if not plan['passed']:raise ValueError('Benchmark requires saved successful geometric plan')
        phases=plan['phases'];scene=OppositionScene(request);end=next(path[0] for phase,path in phases if phase=='opposition_hold');target=scene.k.fk_link(dict(zip(request['joint_names'],end)),'right_palm_link')
        print('BEGIN',run.name,flush=True);old,angles,serial_s=original(request,phases,target)
        if len(old)!=plan['checks'] or not all(r['valid'] for r in old):raise ValueError('Original predicates no longer reproduce saved plan')
        if not np.allclose(angles,plan['turn_path_angles_deg'],atol=1e-12,rtol=0):raise ValueError('Reconstructed target does not reproduce saved turn angles')
        print('ORIGINAL_DONE',run.name,len(old),serial_s,flush=True)
        serial=validate_paths(request,phases,target,1);parallel=validate_paths(request,phases,target,a.workers)
        if old!=serial['predicate_results'] or old!=parallel['predicate_results']:raise ValueError('Predicate-level mismatch')
        if angles!=serial['turn_path_angles_deg'] or angles!=parallel['turn_path_angles_deg']:raise ValueError('Turn angle mismatch')
        record={'run':run.name,'request_sha256':sha(run/'opposition_request.json'),'plan_sha256':sha(run/'opposition_plan.json'),'source_sha256':sources,'asset_sha256':{str(p):sha(p) for p in sorted(assetpaths)},'target_palm_matrix':target.tolist(),'target_derivation':'joint refiner assigns exact FK(closed opposition_hold target); validated against every saved turn angle','checks':len(old),'original_loop_s_excluding_scene_init':serial_s,'new_serial_s_including_scene_init':serial['elapsed_s'],'parallel_s_including_scene_init':parallel['elapsed_s'],'workers':a.workers,'speedup_vs_new_serial':serial['elapsed_s']/parallel['elapsed_s'],'all_predicates_exact_match':True,'turn_angles_exact_match':True,'predicate_results':old,'turn_path_angles_deg':angles,'input_fingerprint':input_hash({'request':request,'phases':phases,'target':target.tolist()})}
        (a.output/(run.name+'.json')).write_text(json.dumps(record,indent=2));results.append({k:v for k,v in record.items() if k not in ('predicate_results','asset_sha256','source_sha256','turn_path_angles_deg','target_palm_matrix')});print('BENCHMARK',json.dumps(results[-1]),flush=True)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
