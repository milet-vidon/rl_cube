"""Joint arm/finger refinement; all original final collision/posture gates remain."""
import numpy as np
from scipy.optimize import least_squares
FINGERS=('thumb','index','middle','ring','little')

def refine(q,scene,calibration,goals,names):
    inset=float(getattr(scene,'refinement_options',{}).get('ring_inset_m',0.))
    if not 0.<=inset<=.0015:raise ValueError('Ring inset outside 0..1.5 mm development envelope')
    goals['ring'][1]-=inset
    variables=[n for n in names if n.endswith('_R') or n.startswith('right_')]
    reference=np.array([q[n] for n in variables]);limits=np.array([scene.k.limits[n] for n in variables])
    lo=limits[:,0]+.02;hi=limits[:,1]-.02
    lo[variables.index('Joint2_R')]=max(lo[variables.index('Joint2_R')],np.radians(-100))
    left=scene.k.fk_link(q,'left_palm_link')
    def residual(v):
        state=q|dict(zip(variables,v));frames=scene.k.fk(state)
        shapes=[(n,m,frames[n]@t,c,r) for n,m,t,c,r in scene.moving]
        groups={f:[a for a in shapes if a[0].startswith('right_'+f+'_')] for f in FINGERS}
        errors=[]
        for finger in FINGERS:
            anchor=calibration[finger];t=frames[anchor['link']]
            pos=scene.C.T@(t[:3,3]+t[:3,:3]@anchor['point_local']-scene.center)
            goal=np.array(goals[finger])
            errors.extend([200*(pos[1]-goal[1]),200*max(pos[2]+.0175,0),
                           200*max(-.027-pos[2],0),200*max(abs(pos[0])-.0265,0),
                           5*(pos[0]-goal[0]),5*(pos[2]-goal[2])])
        palm=frames['right_palm_link'];delta=palm[:3,3]-left[:3,3];direction=delta/max(np.linalg.norm(delta),1e-9)
        errors.extend([20*max(.70+np.dot(palm[:3,0],direction),0),
                       20*max(.70-np.dot(left[:3,0],direction),0),
                       20*max(np.dot(left[:3,0],palm[:3,0])+.35,0),
                       200*max(palm[1,3]-scene.cube_y+.025,0),
                       200*max(palm[1,3]-left[1,3]+.055,0),
                       200*max(frames['Link4_R'][1,3]-frames['Link4_L'][1,3]+.045,0)])
        for i,f in enumerate(FINGERS):
            for g in FINGERS[i+1:]:
                distance=min(scene.distance(a,b) for a in groups[f] for b in groups[g])
                errors.append(600*max(.0038-distance,0))
        for a in shapes:
            if a[0]=='Link1_R':continue
            distance=min(scene.distance(a,b) for b in scene.fixed)
            errors.append(400*max(.0035-distance,0))
        errors.extend(.015*(v-reference))
        return np.asarray(errors)
    fit=least_squares(residual,np.clip(reference,lo+1e-6,hi-1e-6),bounds=(lo,hi),
                      max_nfev=220,diff_step=1e-4,ftol=1e-7)
    result=q|dict(zip(variables,fit.x))
    frames=scene.k.fk(result)
    for finger in FINGERS:
        anchor=calibration[finger];t=frames[anchor['link']]
        pos=scene.C.T@(t[:3,3]+t[:3,:3]@anchor['point_local']-scene.center)
        if abs(pos[1]-goals[finger][1])>.0015 or pos[2]>-.015 or pos[2]<-.029 or abs(pos[0])>.029:
            raise ValueError('Joint refinement finger goal rejected: '+finger+' '+str(pos))
    print('JOINT_REFINE',fit.nfev,'cost',fit.cost,'maximum_residual',max(abs(residual(fit.x))),flush=True)
    return result
