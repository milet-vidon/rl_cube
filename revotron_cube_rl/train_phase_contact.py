"""GPU left-subskill BC with explicit provenance, resume, plateau stop and bounded output."""
from pathlib import Path
import argparse,json,time,shutil,os
import numpy as np
import torch
from phase_contact_policy import PHASES,SCHEMA,FEATURES,ResidualPolicy,accepted_prefix,phase_progress,pair,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=12000);p.add_argument('--batch',type=int,default=2048);p.add_argument('--width',type=int,default=512);p.add_argument('--seed',type=int,default=20261104);p.add_argument('--patience',type=int,default=20);p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_per_process_memory_fraction(.30)
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);device=torch.device(a.device)
    xs=[];ys=[];validation=[];phases=[];provenance=[];seen=set();names=indices=None
    for run in a.source:
        if run.resolve() in seen:raise ValueError('Duplicate source')
        seen.add(run.resolve());rows,nn,ii,meta=accepted_prefix(run)
        if names is not None and (nn!=names or ii!=indices):raise ValueError('Source joint schema mismatch')
        names,indices=nn,ii;provenance.append(meta);progress=phase_progress(rows)
        for i in range(1,len(rows)):
            x,y=pair(rows[i-1],rows[i],indices,progress[i]);xs.append(x);ys.append(y);phases.append(PHASES.index(rows[i]['phase']))
            # Hold out the same phase-progress ranges across correlated same-seed trajectories.
            validation.append(int(progress[i]*100)//10%5==4)
    x=np.asarray(xs);y=np.asarray(ys);val=np.asarray(validation);phase=np.asarray(phases)
    if not val.any() or val.all():raise ValueError('Empty train/validation split')
    from kinematics import RobotKinematics
    k=RobotKinematics();bounds=np.asarray([k.limits[names[i]] for i in indices],dtype=np.float32)
    for run in a.source:
        contract=json.loads((run/'asset_contract.json').read_text())['joint_limit_safety']['soft_limits_rad']
        for j,i in enumerate(indices):
            if names[i] in contract:bounds[j,0]=max(bounds[j,0],contract[names[i]][0]);bounds[j,1]=min(bounds[j,1],contract[names[i]][1])
    mean=x[~val].mean(0);std=np.maximum(x[~val].std(0),.02);model=ResidualPolicy(mean,std,bounds[:,0],bounds[:,1],a.width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    x=torch.from_numpy(x).to(device);y=torch.from_numpy(y).to(device);train_ids=torch.as_tensor(np.where(~val)[0],device=device);val_ids=torch.as_tensor(np.where(val)[0],device=device)
    counts=np.bincount(phase[~val],minlength=len(PHASES));weights=torch.as_tensor(1/np.maximum(counts[phase[~val]],1),dtype=torch.float32,device=device)
    # Reweight moving joints so large static sections cannot dominate selection.
    scale=y[train_ids].std(0).clamp_min(.001)
    source_hashes={n:sha(Path(__file__).parent/n) for n in ('phase_contact_policy.py','train_phase_contact.py','centered_transport_audit.py','tilt_audit.py','wrist_audit.py','elbow_audit.py','kinematics.py')}
    meta={'schema':SCHEMA,'scope':'left_phase_contact_residual_BC_with_recorded_phase_and_right_schedule','right_turn_trained':False,'closed_loop_validated':False,'stage_gate_unlocked':False,'sources':provenance,'unique_source_seeds':sorted({m['seed'] for m in provenance}),'joint_names':names,'output_indices':indices,'width':a.width,'device':a.device,'training_seed':a.seed,'batch':a.batch,'patience_checks':a.patience,'source_sha256':source_hashes,'validation_kind':'shared_phase_progress_blocks_of_seen_seeds_not_independent','training_samples':len(train_ids),'validation_samples':len(val_ids),'max_target_increment_rad':.04,'memory_fraction_cap':.30 if a.device=='cuda' else None}
    start=0;best=float('inf');stale=0;elapsed=0.
    if a.resume:
        saved=torch.load(a.output/'last.pt',map_location=device,weights_only=True)
        if saved['metadata']!=meta:raise ValueError('Resume provenance/config mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['step'];best=saved['best'];stale=saved['stale'];elapsed=saved['elapsed_s'];torch.set_rng_state(saved['cpu_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        snap=a.output/'source_snapshot';snap.mkdir()
        for n in source_hashes:shutil.copy2(Path(__file__).parent/n,snap/n)
        (a.output/'experiment.json').write_text(json.dumps(meta|{'requested_updates':a.updates,'batch':a.batch,'patience_checks':a.patience},indent=2))
    if start>=a.updates:raise ValueError('Resume updates must exceed the last completed update')
    started=time.monotonic();reason='update_budget'
    def save(name,step):
        checkpoint={'metadata':meta,'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'best':best,'stale':stale,'elapsed_s':elapsed+time.monotonic()-started,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else torch.tensor([],dtype=torch.uint8)}
        tmp=a.output/(name+'.tmp');torch.save(checkpoint,tmp);os.replace(tmp,a.output/name)
    with (a.output/'metrics.jsonl').open('a') as log:
        for step in range(start+1,a.updates+1):
            ids=train_ids[torch.multinomial(weights,a.batch,replacement=True)];inputs=x[ids].clone()
            noise=torch.zeros_like(inputs);noise[:,:56]=torch.randn_like(inputs[:,:56])*.0003;noise[:,56:112]=torch.randn_like(inputs[:,56:112])*.002;noise[:,168:171]=torch.randn_like(inputs[:,168:171])*.00005
            pred=model(inputs+noise);loss=((pred-y[ids])/scale).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%100==0 or step==a.updates:
                with torch.no_grad():
                    residual=model(x[val_ids])-y[val_ids];rmse=float(residual.square().mean().sqrt());score=float((residual/scale).square().mean());maximum=float(residual.abs().max())
                improved=score<best*(1-1e-4)
                if improved:best=score;stale=0;save('best.pt',step)
                else:stale+=1
                record={'update':step,'train_scaled_loss':float(loss.detach()),'validation_scaled_loss':score,'validation_delta_rmse_rad':rmse,'validation_max_delta_error_rad':maximum,'plateau_checks':stale,'elapsed_s':elapsed+time.monotonic()-started,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0}
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True);save('last.pt',step)
                (a.output/'progress.json').write_text(json.dumps(record|{'completed':False,'scope':meta['scope']},indent=2))
                if stale>=a.patience:reason='validation_plateau';break
    result=meta|{'completed':True,'updates_completed':step,'stop_reason':reason,'best_validation_scaled_loss':best,'best_checkpoint_sha256':sha(a.output/'best.pt'),'last_checkpoint_sha256':sha(a.output/'last.pt'),'elapsed_s':elapsed+time.monotonic()-started,'task_success_claim':False}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
