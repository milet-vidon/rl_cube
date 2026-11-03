"""Reject any pregrasp/approach/close target with unsafe mesh clearance.

Coal runs in the host ABI. PhysX feedback remains mandatory; this check cannot
certify contact forces or a feedback correction applied after nominal planning.
"""
import json,sys
from pathlib import Path
import numpy as np
from safe_collision_plan import CollisionScene


def audit(request):
    scene=CollisionScene(request['initial'],{},request['cube_position'],request['cube_quaternion'],True)
    fixed=scene.fixed.copy();route_cube=next(b for b in fixed if b[0]=='cube_route_envelope')
    checks=0;names=request['joint_names'];previous=np.array([request['initial'][n]for n in names],dtype=float)
    for phase,path in request['phases']:
        if phase=='bilateral_hold':path=path[:1]
        scene.fixed=fixed if phase in ('right_preshape','right_route') else [b for b in fixed if b[0]!='cube_route_envelope']
        for step,target in enumerate(path):
            target=np.asarray(target);last=target if previous is None else previous
            subdivisions=max(1,int(np.ceil(np.max(np.abs(target-last))/.01)))
            for u in np.linspace(0,1,subdivisions+1)[1:]:
                values=(1-u)*last+u*target
                for n,v in zip(names,values):
                    if n.startswith('right_') and not(scene.k.limits[n][0]-1e-6<=v<=scene.k.limits[n][1]+1e-6):
                        return {'passed':False,'phase':phase,'step':step,'pair':[n,'joint_limits',float(v)],'checks':checks,'physics_validated':False}
                scene.q=request['initial']|{n:float(v) for n,v in zip(names,values) if n.endswith('_R') or n.startswith('right_')}
                scene.last_pair=None;checks+=1
                if not scene.valid(np.array([scene.q[n]for n in scene.names])):
                    return {'passed':False,'phase':phase,'step':step,'interpolation':float(u),'pair':scene.last_pair,'checks':checks,'minimum_clearance_m':.003,'physics_validated':False}
            previous=target
    return {'passed':True,'checks':checks,'minimum_clearance_m':.003,'all_nominal_targets_and_interpolation_checked':True,'physics_validated':False}


if __name__=='__main__':
    result=audit(json.loads(Path(sys.argv[1]).read_text()));Path(sys.argv[2]).write_text(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
