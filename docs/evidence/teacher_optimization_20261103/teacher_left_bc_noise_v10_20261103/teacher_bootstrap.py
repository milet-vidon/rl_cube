"""State BC bootstrap from an audited left grasp/center demonstration.

This does not certify closed-loop grasping or unlock face-turn/PPO gates.
Inputs are the PRE-ACTION q56, cube xyz/quaternion and elapsed control time.
Outputs are absolute targets for the 28 LEFT joints, in config order.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from torch import nn

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset(run,target_field='nominal_target'):
    manifest = json.loads((run/'artifact_manifest.json').read_text())
    for name in ('trajectory.json', 'result.json', 'config.json'):
        if sha(run/name) != manifest['sha256'][name]:
            raise ValueError('Evidence hash mismatch: '+name)
    result = json.loads((run/'result.json').read_text())
    if not (result['full_episode'] and result['dynamic_grasp_passed'] and
            result['center_transport_passed'] and not result['failure'] and
            result['cube_pose_writes_during_control'] == 0 and
            result['robot_table_impulse_sum_Ns'] <= 1e-6 and
            result['self_contact_impulse_sum_Ns'] <= 1e-6):
        raise ValueError('Source is not an accepted full left-center episode')
    names = json.loads((run/'config.json').read_text())['joint_names']
    indices = [i for i,n in enumerate(names) if n.endswith('_L') or n.startswith('left_')]
    assert len(indices) == 28
    rows = json.loads((run/'trajectory.json').read_text())
    x, y = [], []
    for previous, current in zip(rows, rows[1:]):
        # q in each logged row is POST-action: pair previous state with current target.
        x.append(previous['q'] + previous['cube_position'] + previous['cube_quaternion_wxyz'] + [current['step']/3000.])
        y.append(np.asarray(current[target_field])[indices])
    x,y=np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Non-finite demonstration')
    return x,y,names,indices,result

class BoxOutput(nn.Module):
    def __init__(self,bounds):
        super().__init__();self.register_buffer('lower',torch.tensor(bounds[0]));self.register_buffer('upper',torch.tensor(bounds[1]))
    def forward(self,x):return torch.maximum(self.lower,torch.minimum(self.upper,x))

def network(bounded=False,box_bounds=None):
    layers=[nn.Linear(64,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,28)]
    if bounded:layers.append(nn.Tanh())
    if box_bounds is not None:layers.append(BoxOutput(box_bounds))
    return nn.Sequential(*layers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=6000)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--state-noise',type=float,default=0.,help='q noise in rad; pose noise scales with it, clock stays exact')
    p.add_argument('--bounded',action='store_true')
    p.add_argument('--box-output',action='store_true')
    p.add_argument('--extra-source',type=Path,action='append',default=[])
    p.add_argument('--applied-targets',action='store_true')
    a=p.parse_args()
    target_field='applied_target' if a.applied_targets else 'nominal_target'
    x,y,names,indices,source_result=dataset(a.source,target_field)
    provenance=[];seen_sources={a.source.resolve()}
    for extra in a.extra_source:
        if extra.resolve() in seen_sources:raise ValueError('Duplicate demonstration source')
        seen_sources.add(extra.resolve())
        xx,yy,nnames,iindices,rr=dataset(extra,target_field)
        if nnames!=names or iindices!=indices:raise ValueError('Additional source joint schema mismatch')
        provenance.append({'source':str(extra.resolve()),'seed':rr['seed'],'manifest_sha256':sha(extra/'artifact_manifest.json'),'samples':len(xx)})
        x=np.concatenate([x,xx]);y=np.concatenate([y,yy])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'teacher_bootstrap.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    # Held-out contiguous 30-step blocks; same episode, explicitly NOT independent evaluation.
    val=(np.arange(len(x))//30)%5==4;train=~val
    mean=x[train].mean(0);std=np.maximum(x[train].std(0),.02)
    center=y[train].mean(0);scale=np.maximum(y[train].std(0),.05)
    box_bounds=None
    if a.bounded or a.box_output:
        from kinematics import RobotKinematics
        k=RobotKinematics();limits=np.array([k.limits[names[i]] for i in indices],dtype=np.float32)
        # Runtime arm bounds from the accepted demonstration take precedence.
        contract=json.loads((a.source/'asset_contract.json').read_text())
        for j,i in enumerate(indices):
            bound=contract.get('joint_limit_safety',{}).get('soft_limits_rad',{}).get(names[i])
            if bound is not None:limits[j]=bound
        limits[:,0]+=1e-4;limits[:,1]-=1e-4
        if a.bounded:center=limits.mean(1);scale=(limits[:,1]-limits[:,0])/2
        if a.box_output:box_bounds=[((limits[:,0]-center)/scale).tolist(),((limits[:,1]-center)/scale).tolist()]
    X=torch.from_numpy((x-mean)/std);Y=torch.from_numpy((y-center)/scale)
    loss_weights=torch.from_numpy(scale/np.maximum(y[train].std(0),.05)) if a.bounded else torch.ones(28)
    training=torch.from_numpy(np.where(train)[0]);validation=torch.from_numpy(np.where(val)[0])
    model=network(a.bounded,box_bounds);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    meta={'scope':'left_grasp_and_center_state_BC_bootstrap','right_turn_trained':False,
          'closed_loop_validated':False,'stage_gate_unlocked':False,'source':str(a.source.resolve()),
          'source_seed':source_result['seed'],'source_manifest_sha256':sha(a.source/'artifact_manifest.json'),
          'training_script_sha256':sha(__file__),'training_seed':a.seed,'train_samples':int(train.sum()),
          'validation_samples':int(val.sum()),'validation_kind':'temporal_blocks_same_episode_not_independent',
          'schema':'previous_q56_xyz3_wxyz4_time1_to_left_target28_v1',
          'joint_names':names,'output_indices':indices,'updates':a.updates,'state_noise_rad':a.state_noise,'bounded':a.bounded,'box_bounds':box_bounds,
          'additional_sources':provenance,'target_field':target_field}
    (a.output/'experiment.json').write_text(json.dumps(meta,indent=2))
    best=float('inf');started=time.monotonic()
    with (a.output/'metrics.jsonl').open('w') as log:
        for step in range(1,a.updates+1):
            ids=training[torch.randint(len(training),(256,))]
            inputs=X[ids]
            if a.state_noise:
                physical_noise=torch.tensor([a.state_noise]*56+[a.state_noise*.1]*3+[a.state_noise*.2]*4+[0.])
                inputs=inputs+torch.randn_like(inputs)*physical_noise/torch.from_numpy(std)
            loss=((model(inputs)-Y[ids])*loss_weights).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%100==0 or step==1:
                with torch.no_grad():
                    residual=(model(X[validation])-Y[validation])*torch.from_numpy(scale)
                    rmse=float(residual.square().mean().sqrt());maximum=float(residual.abs().max())
                metrics={'update':step,'loss':float(loss.detach()),'validation_target_rmse_rad':rmse,
                         'validation_max_target_error_rad':maximum,'elapsed_s':time.monotonic()-started}
                log.write(json.dumps(metrics)+'\n');log.flush();print(json.dumps(metrics),flush=True)
                if rmse<best:
                    best=rmse
                    torch.save({'state_dict':model.state_dict(),'mean':torch.from_numpy(mean),
                                'std':torch.from_numpy(std),'center':torch.from_numpy(center),
                                'scale':torch.from_numpy(scale),'metadata':meta,'metrics':metrics},a.output/'best.pt')
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=True)
    restored=network(a.bounded,box_bounds);restored.load_state_dict(saved['state_dict'])
    with torch.no_grad():
        predicted=restored(X[validation])*saved['scale']+saved['center']
        reloaded_rmse=float((predicted-torch.from_numpy(y[val])).square().mean().sqrt())
    if abs(reloaded_rmse-best)>1e-6:raise RuntimeError('Reload mismatch')
    report=meta|{'completed':True,'checkpoint':'best.pt','checkpoint_sha256':sha(a.output/'best.pt'),
                 'best_validation_target_rmse_rad':best,'reload_verified':True,'elapsed_s':time.monotonic()-started}
    (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
