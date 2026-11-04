"""Left-only residual BC, conditioned on a recorded phase schedule (not autonomous)."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn

PHASES=('ready','preshape','lower','approach','squeeze','primary_lift','elbow_outward','tilt_90','wrist_reconfigure','tilt_hold','left_center_transport','center_hold')
FINGERS=('thumb','index','middle','ring','little')
SCHEMA='pre_q56_qd56_target56_cube7_contact10_phase12_progress1_left_delta28_v1'
FEATURES=198

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def phase_progress(rows):
    result=[];start=0
    while start<len(rows):
        end=start+1
        while end<len(rows) and rows[end]['phase']==rows[start]['phase']:end+=1
        result.extend((i-start)/max(1,end-start-1) for i in range(start,end));start=end
    return result

def features(previous,phase,progress):
    if phase not in PHASES or not 0<=progress<=1:raise ValueError('Unknown phase or invalid schedule progress')
    contacts=[]
    for side in ('left','right'):
        for finger in FINGERS:
            value=sum(float(p['impulse_Ns']) for p in previous['finger_contact_points'] if p['side']==side and p['finger']==finger)
            if value<0:raise ValueError('Negative contact impulse')
            contacts.append(np.log1p(value/.001))
    values=[]
    for key,count in (('q',56),('joint_velocity',56),('applied_target',56),('cube_position',3),('cube_quaternion_wxyz',4)):
        part=list(previous[key])
        if len(part)!=count:raise ValueError('Bad observation shape: '+key)
        values.extend(part)
    x=np.asarray(values+contacts+[float(phase==p) for p in PHASES]+[progress],dtype=np.float32)
    if x.shape!=(FEATURES,) or not np.isfinite(x).all():raise ValueError('Invalid feature vector')
    return x

def pair(previous,current,indices,progress):
    # Logs are post-action: previous features predict CURRENT applied target increment.
    x=features(previous,current['phase'],progress)
    delta=(np.asarray(current['applied_target'])-np.asarray(previous['applied_target']))[indices]
    if not np.isfinite(delta).all() or np.abs(delta).max()>.04001:raise ValueError('Applied increment exceeds preserved step cap')
    return x,delta.astype(np.float32)

def accepted_prefix(run):
    """Accept only a safe left-center prefix, explicitly excluding any failed turn tail."""
    run=Path(run);manifest=json.loads((run/'artifact_manifest.json').read_text())
    for name in ('trajectory.json','result.json','config.json','asset_contract.json','center_plan.json','center_result.json'):
        if sha(run/name)!=manifest['sha256'][name]:raise ValueError('Evidence hash mismatch: '+name)
    result=json.loads((run/'result.json').read_text());center=json.loads((run/'center_result.json').read_text())
    if not (result['full_episode'] and result['dynamic_grasp_passed'] and result['center_transport_passed'] and center['passed'] and center['style']=='j56' and result['cube_pose_writes_during_control']==0):raise ValueError('Left prefix acceptance missing')
    rows=json.loads((run/'trajectory.json').read_text());end=next((i for i,r in enumerate(rows) if r['phase'] not in PHASES),len(rows));prefix=rows[:end]
    if not prefix or prefix[-1]['phase']!='center_hold':raise ValueError('Incomplete left center prefix')
    if any(r['phase'] in PHASES for r in rows[end:]):raise ValueError('Noncontiguous left prefix')
    names=json.loads((run/'config.json').read_text())['joint_names'];indices=[i for i,n in enumerate(names) if n.endswith('_L') or n.startswith('left_')]
    if len(names)!=56 or len(indices)!=28:raise ValueError('Joint schema mismatch')
    plan=json.loads((run/'center_plan.json').read_text())
    from centered_transport_audit import assess_center_row
    hold=longest=0;lifted=False
    for row in prefix:
        if row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['drop'] or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit']):raise ValueError('Unsafe prefix')
        lifted=lifted or row['height_m']>=.08
        if lifted and row['height_m']<.05:raise ValueError('Returned to table in prefix')
        if row['phase']=='center_hold':
            accepted=assess_center_row(row,names,plan['center_target_world_m'],plan['cube_target_quaternion_wxyz'])['qualified']
            hold=hold+1 if accepted else 0;longest=max(longest,hold)
    if longest<90:raise ValueError('Prefix does not reproduce 3-second center hold')
    meta={'run':run.name,'seed':result['seed'],'manifest_sha256':sha(run/'artifact_manifest.json'),'prefix_rows':end,'excluded_tail_rows':len(rows)-end,'full_episode_failure':result['failure'],'recomputed_center_hold_s':longest/30.,'source_scope':'left prefix only; no right-turn success label'}
    return prefix,names,indices,meta

class ResidualPolicy(nn.Module):
    def __init__(self,mean,std,lower,upper,width=512):
        super().__init__()
        for name,value in (('mean',mean),('std',std),('lower',lower),('upper',upper)):self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(FEATURES,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,28))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return torch.tanh(self.net(((x-self.mean)/self.std).clamp(-15,15)))*.04
    def targets(self,x,previous_left):return torch.clamp(previous_left+self(x),self.lower,self.upper)

def restored(saved):
    m=saved['metadata'];s=saved['model'];model=ResidualPolicy(s['mean'],s['std'],s['lower'],s['upper'],m['width']);model.load_state_dict(s);return model
