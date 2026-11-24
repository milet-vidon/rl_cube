"""Fork of the pinned 90 mm fit loop with outcome-weighted supervised training.
The unweighted seed-held-out metric, lineage verification and AdamW resume
semantics stay identical. The legacy fit source must remain byte-for-byte intact.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from contact_hold90_cuda_fit import ALGORITHM, RESERVED, ROUND15, _host, _json, _put, _sha, _state, load_batches, restored
from contact_hold90_controls import OBJECTIVE, validate_params
from contact_hold90_objective import Weighted, imitation_loss, weights_for_rows

def fit(parent,data,outdir,batch=32768,updates=5000,lr=5e-5,*,device='cuda:0',save_every=250,seed=90017,
        exclude_seeds=(),stop_after=None,controls=None):
    """Resume an interrupted block exactly; stop_after is only an artificial interruption for tests.

    Best is selected by mean per-phase held-out loss (baseline is eligible).
    Across rounds the selected actor AND its AdamW moments are inherited. The
    first 90 mm update block initializes from the public round-15 actor weights.
    """
    for n in (batch,updates,save_every):
        if type(n)!=int or n<=0:raise ValueError('Batch, updates and save interval must be positive integers')
    if not 0<lr<=1 or not math.isfinite(lr):raise ValueError('Invalid learning rate')
    if stop_after is not None and (type(stop_after)!=int or not 0<stop_after<=updates):raise ValueError('Invalid stop point')
    out=Path(outdir).resolve(); out.mkdir(exist_ok=True,parents=True)
    with (out/'fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _fit(Path(parent).resolve(),data,out,batch,updates,lr,device,save_every,seed,exclude_seeds,stop_after,validate_params(controls or {}))


def _fit(parent,data,out,batch,updates,lr,device,save_every,seed,exclude,stop_after,controls):
    if parent.is_dir():parent=parent/'best.pt'
    if parent.name!='best.pt' or parent.parent==out:raise ValueError('Expect distinct parent best.pt')
    device=torch.device(device)
    if device.type not in ('cuda','cpu'):raise ValueError('Use CUDA (or CPU for verification)')
    if device.type=='cuda':
        # PyTorch needs this before CUDA initialization for deterministic GEMMs.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'):raise ValueError('Invalid deterministic cuBLAS workspace')
        if not torch.cuda.is_available():raise RuntimeError('GPU requested but CUDA unavailable; no silent fallback')
        if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high' if device.type=='cuda' else 'highest')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    parent_sha=_sha(parent); ck=_state(parent); meta=ck['metadata']
    if parent_sha!=ROUND15 and (meta.get('algorithm')!=ALGORITHM or meta.get('warm_start_sha256')!=ROUND15
          or not _json(parent.parent/'result.json').get('completed')
          or _json(parent.parent/'result.json')['checkpoint_sha256']!=parent_sha):
        raise ValueError('Expected the frozen R15 actor or its completed 90 mm fit descendant')
    prior=list(meta.get('physical90_ancestor_sha256',[]))
    allowed=set(prior)|{parent_sha}
    arrays,details,train_seeds,val_seeds=load_batches(data,allowed,meta.get('training_seeds',[]),meta.get('loss_val_seeds_90',[]),exclude)
    if meta.get('algorithm')==ALGORITHM and (meta['asset_sha256'],meta['grasp_sha256'])!=(details[0]['asset_sha256'],details[0]['grasp_sha256']):
        raise ValueError('Parent used a different physical geometry or grasp')
    names=('contact_hold90_cuda_fit.py','contact_hold_policy.py','phase_contact_policy.py','contact_hold90_controls.py','contact_hold90_objective.py','contact_hold90_weighted_fit.py','contact_hold90_reward_sample.py')
    code={n:_sha(Path(__file__).with_name(n)) for n in names}
    objective={k:controls[k] for k in OBJECTIVE}
    spec=dict(objective=objective,parent_sha256=parent_sha,parent=str(parent),collections=details,batch=batch,updates=updates,lr=lr,
              save_every=save_every,seed=seed,exclude_seeds=sorted(set(exclude)|RESERVED),code=code,
              torch_version=str(torch.__version__),device=str(device),gpu=(torch.cuda.get_device_name(device) if device.type=='cuda' else None),
              cuda_version=torch.version.cuda,device_capability=(list(torch.cuda.get_device_capability(device)) if device.type=='cuda' else None))
    sig=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    plan=out/'fit_plan.json'; last=out/'last.pt'; best=out/'best.pt'
    if plan.exists():
        if _json(plan)!=spec:raise ValueError('Changed parent, data, software, device or settings on resume')
    else:
        if any(f.exists() for f in (last,best,out/'result.json')):raise ValueError('Unidentified fit outputs already exist')
        _put(plan,spec)
    if (out/'result.json').exists():
        done=_json(out/'result.json')
        if done['plan_sha256']!=sig or done['checkpoint_sha256']!=_sha(best) or done['last_sha256']!=_sha(last):
            raise ValueError('Completed checkpoint differs from receipt')
        return done
    model=restored(ck).to(device); model.train()
    x,y,mask,ph,v=arrays
    low=model.lower.detach().cpu().numpy(); high=model.upper.detach().cpu().numpy()
    if not (model.scale>0).all() or np.any(mask & ((y<low-1e-5)|(y>high+1e-5))):raise ValueError('Label outside measured joint bounds')
    weights,hits=weights_for_rows(details,controls)
    groups=Weighted(x,y,mask,ph,v,device,weights,controls); del arrays,x,y,mask,ph,v,weights
    if batch<len(groups.tables[0][1]):raise ValueError('Batch smaller than number of represented phases')
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5,fused=(device.type=='cuda'))
    opt_inherited='physical90_optimizer' in ck
    if opt_inherited:
        opt.load_state_dict(ck['physical90_optimizer'])
        for group in opt.param_groups:group['lr']=lr
    gen=torch.Generator(device=device).manual_seed(seed)
    if last.exists():
        block=_state(last)
        if block['plan_sha256']!=sig:raise ValueError('Last checkpoint belongs to another block')
        model.load_state_dict(block['model']);opt.load_state_dict(block['optimizer']);gen.set_state(block['rng'])
        start=block['step']; best_so_far=block['best']; latest=block['latest_score']
        if not 0<=start<=updates:raise ValueError('Invalid saved iteration')
    else:
        start=0;latest=groups.score(model)
        best_so_far=dict(step=0,score=latest,model=_host(model.state_dict()),optimizer=_host(opt.state_dict()))
    def persist(step):
        _put(last,dict(plan_sha256=sig,step=step,model=_host(model.state_dict()),optimizer=_host(opt.state_dict()),
                       rng=gen.get_state(),best=best_so_far,latest_score=latest),True)
        # last.pt is authoritative after any interruption between the two atomic replaces.
        best_meta=dict(meta,algorithm=ALGORITHM,warm_start_sha256=ROUND15,physical90_ancestor_sha256=prior+[parent_sha],
                 plan_sha=sig,parent_checkpoint_sha256=parent_sha,parent_run=str(parent.parent),
                 source_sha256=meta.get('source_sha256',{})|code,device=str(device),learning_rate=lr,
                 batch_size=batch,updates=step,selected_step=best_so_far['step'],optimizer_reset=not opt_inherited,
                 training_objective=objective,training_episode_outcome_counts=hits,seed=seed,training_samples=sum(groups.phase_rows['train'].values()),validation_samples=sum(groups.phase_rows['val'].values()),
                 closed_loop_success_demonstrated=False,training_seeds=sorted(train_seeds),loss_val_seeds_90=sorted(val_seeds),
                 validation_kind='collection_seed_holdout_loss_only',asset_sha256=details[0]['asset_sha256'],grasp_sha256=details[0]['grasp_sha256'],
                 long_gpu_updates=int(meta.get('long_gpu_updates',0))+best_so_far['step'],
                 physical90_scheduled_updates=int(meta.get('physical90_scheduled_updates',0))+step)
        _put(best,dict(model=best_so_far['model'],metadata=best_meta,metrics=best_so_far['score'],
                       physical90_optimizer=best_so_far['optimizer']),True)
    if not last.exists():persist(0)
    end=updates if stop_after is None else max(start,stop_after)
    for step in range(start+1,end+1):
        bx,by,bm,bonus=groups.draw(batch,step,gen)
        err=(model(bx)-by)/model.scale
        loss=imitation_loss(err,bm,bonus)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step%save_every==0 or step==updates:
            latest=groups.score(model)
            if latest['phase_balanced_normalized_mse']<best_so_far['score']['phase_balanced_normalized_mse']:
                best_so_far=dict(step=step,score=latest,model=_host(model.state_dict()),optimizer=_host(opt.state_dict()))
            print('FIT90',step,'/',updates,'heldout_mse',latest['phase_balanced_normalized_mse'],'best_step',best_so_far['step'],flush=True)
            persist(step)
        elif step==end:persist(step)
    if end<updates:return dict(completed=False,step=end,plan_sha256=sig)
    if start==updates:persist(updates)
    loaded=_state(best);reloaded=restored(loaded).to(device)
    probe=groups.tables[1][0][0][:64]
    with torch.inference_mode():
        model.load_state_dict(best_so_far['model'])
        if not torch.equal(model(probe),reloaded(probe)):raise ValueError('Exported actor reload mismatch')
    result=dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256=_sha(best),last_sha256=_sha(last),
                plan_sha256=sig,updates=updates,selected_step=best_so_far['step'],best_metrics=best_so_far['score'],
                training_objective=objective,training_episode_outcome_counts=hits,last_metrics=latest,phase_rows=groups.phase_rows,checkpoint_selection='lowest seed-held-out phase-balanced normalized MSE',
                metric_scope='offline teacher imitation on 90 mm; does not establish physical holding or rotation')
    _put(out/'experiment.json',loaded['metadata']);_put(out/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--batch',type=int,default=32768)
    p.add_argument('--updates',type=int,default=5000);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--save-every',type=int,default=250);p.add_argument('--seed',type=int,default=90017)
    p.add_argument('--device',default='cuda:0');p.add_argument('--exclude-seeds',nargs='*',type=int,default=[])
    p.add_argument("--control-receipt",type=Path,required=True)
    args=vars(p.parse_args());args["controls"]=_json(args.pop("control_receipt"))["params"];print(json.dumps(fit(**args),allow_nan=False,indent=2))
