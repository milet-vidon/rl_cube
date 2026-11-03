"""Opt-in left index support candidate: bounded contact feedback and offline envelope.

This is an incremental offset from the applied target left by center feedback.
No cube actuation or acceptance thresholds are changed. Offline checks are not
physical validation. The caller must validate this envelope before using offsets.
"""
import math

JOINT = 'left_index_MPR_joint'
LOWER = -.003
UPPER = 0.
STEP = .00005
MIN_IMPULSE = .0005
MAX_IMPULSE = .002


def parameters():
    return dict(joint=JOINT, lower_offset_rad=LOWER, upper_offset_rad=UPPER,
                step_rad=STEP, minimum_impulse_Ns=MIN_IMPULSE,
                maximum_impulse_Ns=MAX_IMPULSE)


class IndexSupport:
    def __init__(self):
        self.offset = 0.

    def update(self, impulse):
        if not math.isfinite(impulse) or impulse < 0:
            raise ValueError('Invalid measured index impulse')
        if impulse < MIN_IMPULSE:
            self.offset = max(LOWER, self.offset - STEP)
        elif impulse > MAX_IMPULSE:
            self.offset = min(UPPER, self.offset + STEP)
        return self.offset


def envelope_values(initial_value, target_value):
    """Absolute positions covering measured q, applied target, and all negative offsets."""
    if not all(math.isfinite(x) for x in (initial_value, target_value)):
        raise ValueError('Nonfinite index baseline')
    low = min(initial_value, target_value + LOWER)
    high = max(initial_value, target_value)
    count = max(1, math.ceil((high-low)/.001))
    return sorted(set([low + (high-low)*i/count for i in range(1,count)] +
                      [low, high, initial_value, target_value, target_value + LOWER]))


def index_impulse(row):
    import numpy as np
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(np.asarray(row['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
    value = 0.
    for point in row['finger_contact_points']:
        if point['side'] != 'left' or point['finger'] != 'index':
            continue
        local = rotation.T @ (np.asarray(point['position_world_m']) - row['cube_position'])
        if local[0] >= .0215 and np.max(np.abs(local)) <= .036:
            value += point['impulse_Ns']
    return float(value)


def _check_shard(payload):
    import numpy as np
    import coal
    from scipy.spatial.transform import Rotation
    from opposed_grasp_plan import OppositionScene
    request, jobs, values, baseline = payload
    scene = OppositionScene(request)
    initial_frames = scene.k.fk(request['initial'])
    # Store link-local geometry. Updating q alone does not update fixed meshes.
    fixed_local = [(n,m,np.linalg.inv(initial_frames[n])@t,c,r) if n in initial_frames
                   else (n,m,None,c,r) for n,m,t,c,r in scene.fixed]
    original_fixed = list(scene.fixed)
    checks = 0
    support_cache = set()
    layer_cache = set()
    for phase, step, state, theta in jobs:
        for value in values:
            q = dict(state)
            # Other left joints retain measured q. The caller's runtime only
            # changes JOINT; its measured/applied discrepancy is fully sampled.
            q[JOINT] = value
            scene.q = q
            scene.theta = theta
            scene.contact = phase in ('right_close','opposition_hold','opposed_face_turn','opposed_face_hold')
            frames = scene.k.fk(q)
            scene.fixed = [(n,m,frames[n]@local,c,r) if local is not None else original
                           for (n,m,local,c,r), original in zip(fixed_local, original_fixed)]
            checks += 1
            if not scene.k.limits[JOINT][0] <= value <= scene.k.limits[JOINT][1]:
                return dict(passed=False, phase=phase, step=step, index_value_rad=value,
                            pair=[JOINT,'joint_limits',value], checks=checks)
            if not scene.valid(np.array([q[n] for n in scene.names])):
                return dict(passed=False, phase=phase, step=step, index_value_rad=value,
                            pair=scene.last_pair, checks=checks)
            rings = [a for a in scene.fixed if a[0].startswith('left_index_')]
            for a in ([] if value in support_cache else rings):
                n,m,t,c,r = a
                if (np.asarray(m.points())@t[:3,:3].T+t[:3,3])[:,2].min() < -.246:
                    return dict(passed=False,phase=phase,step=step,index_value_rad=value,
                                pair=[n,'table'],checks=checks)
                for b in scene.fixed:
                    name = b[0]
                    # Same finger and its adjacent palm/wrist links are excluded
                    # as in the existing center scene. All other fingers,
                    # non-adjacent arm links and exported torso remain checked.
                    other_finger = any(name.startswith('left_'+f+'_') for f in ('thumb','middle','ring','little'))
                    arm = name.startswith('Link') and name.endswith('_L') and int(name[4]) < 5
                    torso = name not in frames
                    if other_finger or arm or torso:
                        distance = scene.distance(a,b)
                        if distance < .003:
                            return dict(passed=False,phase=phase,step=step,index_value_rad=value,
                                        pair=[n,name,distance],checks=checks)
                t_cube=np.eye(4);t_cube[:3,:3]=scene.C;t_cube[:3,3]=scene.center+scene.C@[0,0,.0095]
                stationary=('stationary_layers',coal.Box(.057,.057,.038),t_cube,np.zeros(3),.05)
                distance=scene.distance(a,stationary)
                required=-.002 if n.startswith('left_index_DIP') else .002
                if distance < required:
                    return dict(passed=False,phase=phase,step=step,index_value_rad=value,
                                pair=[n,'stationary_layers',distance],checks=checks)
            support_cache.add(value)
            # Keep the moving layer clear of the left hand, including before turn.
            turn = scene.C @ Rotation.from_euler('z',-theta,degrees=True).as_matrix()
            t=np.eye(4);t[:3,:3]=turn;t[:3,3]=scene.center+scene.C@[0,0,-.019]
            layer=('target_nZ_layer',coal.Box(.057,.057,.019),t,np.zeros(3),.045)
            for a in ([] if (value,theta) in layer_cache else rings):
                distance=scene.distance(a,layer)
                if distance < .001:
                    return dict(passed=False,phase=phase,step=step,index_value_rad=value,
                                pair=[a[0],layer[0],distance],checks=checks)
            layer_cache.add((value,theta))
    return dict(passed=True,checks=checks)


def validate_plan(request, plan, baseline_target, workers=1):
    """Validate the whole original plan for the left-index candidate envelope.

    request.initial is measured q; baseline_target is the applied 56D target
    vector/dict captured at entry. Each original path segment is sampled at
    <=.01rad, each left-index envelope at <=.001rad. A fresh process pool may
    parallelize CPU geometry work; it starts no Isaac process.
    """
    import time
    import numpy as np
    from scipy.spatial.transform import Rotation
    from kinematics import RobotKinematics
    start=time.monotonic();names=request['joint_names'];initial=request['initial']
    baseline=dict(zip(names,baseline_target)) if not isinstance(baseline_target,dict) else baseline_target
    if not plan.get('passed'):
        raise ValueError('Cannot validate a rejected opposition plan')
    if JOINT not in baseline or JOINT not in initial:
        raise ValueError('Missing left index measured/applied baseline')
    values=envelope_values(initial[JOINT],baseline[JOINT])
    k=RobotKinematics();C=Rotation.from_quat(np.asarray(request['cube_quaternion'])[[1,2,3,0]]).as_matrix()
    close=next(path[-1] for phase,path in plan['phases'] if phase=='right_close')
    reference=k.fk_link(dict(zip(names,close)),'right_palm_link')[:3,:3]
    previous=np.array([initial[n] for n in names]);jobs=[]
    for phase,path in plan['phases']:
        for step,point in enumerate(path):
            point=np.asarray(point)
            if point.shape!=(len(names),) or not np.isfinite(point).all():
                raise ValueError('Invalid path target')
            for fraction in np.linspace(0,1,max(1,int(np.ceil(np.max(np.abs(point-previous))/.01)))+1)[1:]:
                state=dict(zip(names,(1-fraction)*previous+fraction*point));theta=0.
                if any(abs(state[n]-initial[n])>1e-10 for n in names if n.startswith('left_') or n.endswith('_L')):
                    raise ValueError('Support cache requires a fixed original left-hand path')
                if phase in ('opposed_face_turn','opposed_face_hold'):
                    p=k.fk_link(state,'right_palm_link')[:3,:3]
                    theta=float(-Rotation.from_matrix(C.T@p@reference.T@C).as_rotvec()[2]*180/np.pi)
                    if phase=='opposed_face_turn' and fraction==1.:
                        recorded=plan.get('turn_path_angles_deg',[])
                        if len(recorded)!=len(path) or abs(theta-recorded[step])>1e-6:
                            raise ValueError('Turn reference does not match original plan')
                jobs.append((phase,step,state,theta))
            previous=point
    workers=max(1,min(int(workers),12,len(jobs)))
    payloads=[(request,jobs[i::workers],values,baseline) for i in range(workers)]
    if workers==1:
        results=[_check_shard(payloads[0])]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(_check_shard,payloads))
    failed=next((r for r in results if not r['passed']),None)
    return dict(passed=failed is None,parameters=parameters(),absolute_index_samples_rad=values,
                initial_index_rad=initial[JOINT],baseline_index_rad=baseline[JOINT],
                checks=sum(r['checks'] for r in results),failure=failed,
                workers=workers,elapsed_s=time.monotonic()-start,physics_validated=False)


def validate_index_ring_pair(request, baseline_target):
    """Check only the coupled, static index/ring mesh pairs.

    This result alone is NOT a full combined envelope certificate. Combining
    separately successful full-path checks is valid only for the same request,
    plan, baseline, and fixed other-left joints: every right/index or right/ring
    distance depends on one offset; only index/ring distances depend on both.
    Contact dynamics still require a new physical rollout.
    """
    import hashlib
    import json
    import numpy as np
    from opposed_grasp_plan import OppositionScene
    from left_ring_support import JOINT as RING_JOINT, envelope_values as ring_values
    names=request['joint_names'];initial=request['initial']
    baseline=dict(zip(names,baseline_target)) if not isinstance(baseline_target,dict) else baseline_target
    iv=envelope_values(initial[JOINT],baseline[JOINT])
    rv=ring_values(initial[RING_JOINT],baseline[RING_JOINT])
    scene=OppositionScene(request);initial_frames=scene.k.fk(initial)
    local=[(n,m,np.linalg.inv(initial_frames[n])@t,c,r) for n,m,t,c,r in scene.fixed
           if n.startswith(('left_index_','left_ring_'))]
    checks=0;minimum=float('inf');pair=None
    for a in iv:
        for b in rv:
            q=initial|{JOINT:a,RING_JOINT:b};frames=scene.k.fk(q)
            if not all(scene.k.limits[n][0]<=q[n]<=scene.k.limits[n][1] for n in (JOINT,RING_JOINT)):
                return dict(static_pair_clearance_passed=False,full_combined_envelope_validated=False,
                            failure='joint_limits',checks=checks)
            index=[(n,m,frames[n]@t,c,r) for n,m,t,c,r in local if n.startswith('left_index_')]
            ring=[(n,m,frames[n]@t,c,r) for n,m,t,c,r in local if n.startswith('left_ring_')]
            for first in index:
                for second in ring:
                    distance=scene.distance(first,second);checks+=1
                    if distance<minimum:minimum=distance;pair=[first[0],second[0],a,b]
                    if distance<.003:
                        return dict(static_pair_clearance_passed=False,full_combined_envelope_validated=False,
                                    checks=checks,minimum_distance_m=minimum,pair=pair)
    return dict(static_pair_clearance_passed=True,full_combined_envelope_validated=False,
                checks=checks,minimum_distance_m=minimum,pair=pair,
                index_values_rad=iv,ring_values_rad=rv,physics_validated=False,
                request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                baseline_sha256=hashlib.sha256(json.dumps(baseline,sort_keys=True).encode()).hexdigest())
