"""Bounded right-finger gain comparison; keep arm gains and effort limits."""
import torch

def configure_right_fingers(env,stiffness=None):
    names=env.cfg['joint_names']
    selected=[(i,n) for i,n in enumerate(names) if n.startswith('right_')]
    if len(selected)!=21:raise ValueError('Expected the original 21 right finger joints')
    if stiffness is not None and stiffness not in (10,20,30):
        raise ValueError('Right finger stiffness must be 10, 20 or 30')
    ids=[env.ids[i] for i,_ in selected]
    if stiffness is not None:
        env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(ids)),float(stiffness),dtype=torch.float32),joint_ids=ids)
    return {'right_finger_stiffness':float(stiffness if stiffness is not None else env.cfg.get('fingers_stiffness',30.)),
            'override_applied':stiffness is not None,'joint_names':[n for _,n in selected],
            'arm_gains_changed':False,'effort_limits_changed':False,'phase':'before_right_preshape'}
