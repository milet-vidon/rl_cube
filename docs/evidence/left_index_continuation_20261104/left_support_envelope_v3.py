"""Joint index/ring envelope using the original right-path collision checks.

The original left-ring checker remains unchanged. Both support fingers' full
sampled ranges are combined, including their pairwise collision checks; no
candidate is certified by checking each finger in isolation.
"""
from left_ring_support import envelope_values, parameters, JOINT
from left_index_continuation import index_envelope_values, SETTINGS, ContinuedIndexSupport
INDEX_JOINT=SETTINGS['joint']


def validate_handover_baseline(state, baseline):
    import math
    ContinuedIndexSupport(state)
    fields = ('reference_target_rad','last_applied_offset_rad','next_offset_rad',
              'last_nominal_target_rad','last_applied_target_rad')
    if not all(math.isfinite(float(state[name])) for name in fields):
        raise ValueError('Nonfinite index handover metadata')
    if float(baseline[INDEX_JOINT]) != float(state['last_applied_target_rad']):
        raise ValueError('Index handover belongs to a different applied baseline')
    if abs(state['reference_target_rad']+state['last_applied_offset_rad']-state['last_nominal_target_rad']) > 1e-12:
        raise ValueError('Index handover reference is inconsistent')
    if abs(state['last_nominal_target_rad']-state['last_applied_target_rad']) > 2e-7:
        raise ValueError('Index handover nominal command was not actually applied')
    if not SETTINGS['lower_offset_rad'] <= state['last_applied_offset_rad'] <= 0:
        raise ValueError('Previous index offset outside original bounds')
    if abs(state['next_offset_rad']-state['last_applied_offset_rad']) > SETTINGS['step_rad']+1e-12:
        raise ValueError('Discontinuous preflight handover')

def _check_shard(payload):
    import numpy as np
    import coal
    from scipy.spatial.transform import Rotation
    from opposed_grasp_plan import OppositionScene
    request, jobs, values, baseline = payload
    scene = OppositionScene(request)
    from opposed_distance_cache_v1 import ExactJobDistanceCache
    distance_cache = ExactJobDistanceCache(scene)
    scene.distance = distance_cache.distance
    initial_frames = scene.k.fk(request['initial'])
    # Store link-local geometry. Updating q alone does not update fixed meshes.
    fixed_local = [(n,m,np.linalg.inv(initial_frames[n])@t,c,r) if n in initial_frames
                   else (n,m,None,c,r) for n,m,t,c,r in scene.fixed]
    original_fixed = list(scene.fixed)
    checks = 0
    support_cache = set()
    layer_cache = set()
    for phase, step, state, theta in jobs:
        distance_cache.begin_job()
        for index_value, value in values:
            q = dict(state)
            # The moving left index and ring joints are checked jointly;
            # other left joints retain measured q as in the original preflight.
            q[JOINT] = value
            q[INDEX_JOINT] = index_value
            scene.q = q
            scene.theta = theta
            scene.contact = phase in ('right_close','opposition_hold','opposed_face_turn','opposed_face_hold')
            frames = scene.k.fk(q)
            scene.fixed = [(n,m,frames[n]@local,c,r) if local is not None else original
                           for (n,m,local,c,r), original in zip(fixed_local, original_fixed)]
            checks += 1
            if not (scene.k.limits[JOINT][0] <= value <= scene.k.limits[JOINT][1] and scene.k.limits[INDEX_JOINT][0] <= index_value <= scene.k.limits[INDEX_JOINT][1]):
                return dict(passed=False, phase=phase, step=step, ring_value_rad=value,index_value_rad=index_value,
                            pair=[JOINT if not scene.k.limits[JOINT][0] <= value <= scene.k.limits[JOINT][1] else INDEX_JOINT,'joint_limits',value if not scene.k.limits[JOINT][0] <= value <= scene.k.limits[JOINT][1] else index_value], checks=checks)
            if not scene.valid(np.array([q[n] for n in scene.names])):
                return dict(passed=False, phase=phase, step=step, ring_value_rad=value,index_value_rad=index_value,
                            pair=scene.last_pair, checks=checks)
            rings = [a for a in scene.fixed if a[0].startswith(('left_ring_', 'left_index_'))]
            for a in ([] if (index_value,value) in support_cache else rings):
                n,m,t,c,r = a
                if (np.asarray(m.points())@t[:3,:3].T+t[:3,3])[:,2].min() < -.246:
                    return dict(passed=False,phase=phase,step=step,ring_value_rad=value,index_value_rad=index_value,
                                pair=[n,'table'],checks=checks)
                for b in scene.fixed:
                    name = b[0]
                    # Same finger and its adjacent palm/wrist links are excluded
                    # as in the existing center scene. All other fingers,
                    # non-adjacent arm links and exported torso remain checked.
                    finger = 'index' if n.startswith('left_index_') else 'ring'
                    other_finger = any(name.startswith('left_'+f+'_') for f in ('thumb','index','middle','ring','little') if f != finger)
                    arm = name.startswith('Link') and name.endswith('_L') and int(name[4]) < 5
                    torso = name not in frames
                    if other_finger or arm or torso:
                        distance = scene.distance(a,b)
                        if distance < .003:
                            return dict(passed=False,phase=phase,step=step,ring_value_rad=value,index_value_rad=index_value,
                                        pair=[n,name,distance],checks=checks)
                t_cube=np.eye(4);t_cube[:3,:3]=scene.C;t_cube[:3,3]=scene.center+scene.C@[0,0,.0095]
                stationary=('stationary_layers',coal.Box(.057,.057,.038),t_cube,np.zeros(3),.05)
                distance=scene.distance(a,stationary)
                required=-.002 if n.startswith(('left_ring_DIP','left_index_DIP')) else .002
                if distance < required:
                    return dict(passed=False,phase=phase,step=step,ring_value_rad=value,index_value_rad=index_value,
                                pair=[n,'stationary_layers',distance],checks=checks)
            support_cache.add((index_value,value))
            # Keep the moving layer clear of the left hand, including before turn.
            turn = scene.C @ Rotation.from_euler('z',-theta,degrees=True).as_matrix()
            t=np.eye(4);t[:3,:3]=turn;t[:3,3]=scene.center+scene.C@[0,0,-.019]
            layer=('target_nZ_layer',coal.Box(.057,.057,.019),t,np.zeros(3),.045)
            for a in ([] if (index_value,value,theta) in layer_cache else rings):
                distance=scene.distance(a,layer)
                if distance < .001:
                    return dict(passed=False,phase=phase,step=step,ring_value_rad=value,index_value_rad=index_value,
                                pair=[a[0],layer[0],distance],checks=checks)
            layer_cache.add((index_value,value,theta))
    return dict(passed=True,checks=checks,distance_cache=distance_cache.report())


def validate_plan(request, plan, baseline_target, index_state, workers=1):
    """Validate the whole original plan for the joint index/ring candidate envelope.

    request.initial is measured q; baseline_target is the applied 56D target
    vector/dict captured at entry. Each original path segment is sampled at
    <=.01rad, each left-support envelope at <=.001rad. A fresh process pool may
    parallelize CPU geometry work; it starts no Isaac process.
    """
    import time
    import numpy as np
    from scipy.spatial.transform import Rotation
    from kinematics import RobotKinematics
    start=time.monotonic();names=request['joint_names'];initial=request['initial']
    baseline=dict(zip(names,baseline_target)) if not isinstance(baseline_target,dict) else baseline_target
    validate_handover_baseline(index_state,baseline)
    if not plan.get('passed'):
        raise ValueError('Cannot validate a rejected opposition plan')
    if JOINT not in baseline or JOINT not in initial:
        raise ValueError('Missing left ring measured/applied baseline')
    ring_values=envelope_values(initial[JOINT],baseline[JOINT])
    index_values=index_envelope_values(initial[INDEX_JOINT],index_state)
    values=[(i,r) for i in index_values for r in ring_values]
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
    return dict(passed=failed is None,parameters=parameters(),absolute_ring_samples_rad=ring_values,absolute_index_samples_rad=index_values,index_continuation=index_state,
                initial_ring_rad=initial[JOINT],baseline_ring_rad=baseline[JOINT],
                checks=sum(r['checks'] for r in results),failure=failed,
                workers=workers,elapsed_s=time.monotonic()-start,physics_validated=False,distance_cache_shards=[r.get('distance_cache') for r in results])
