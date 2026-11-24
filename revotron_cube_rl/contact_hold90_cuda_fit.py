"""Fit the existing 28-target actor to verified training splits of measured 90 mm teacher queries.

The held-out episodes in collection reports are used only for offline loss
selection; closed-loop performance is decided by a separate physical panel.
Example: python contact_hold90_cuda_fit.py --parent R15/best.pt --data wave/batch.npz --outdir round/fit
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

from contact_hold_grace_worker import CHECKPOINT as ROUND15
from contact_hold_policy import restored
from phase_contact_policy import FEATURES, PHASES

SCHEMA='contact_hold90_feedback_v1'
ALGORITHM='physical_90mm_phase_balanced_dagger_v1'
RESERVED={21611,21612,51611,51612,51613}


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _json(path):
    return json.loads(Path(path).read_bytes())


def _put(path,obj,tensor=False):
    temp=path.with_name('.'+path.name+'.tmp')
    with temp.open('wb') as f:
        if tensor:torch.save(obj,f)
        else:f.write((json.dumps(obj,indent=2,allow_nan=False)+'\n').encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def _state(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def _ints(values):
    if not isinstance(values,list) or any(type(v)!=int or v<0 for v in values) or len(values)!=len(set(values)):
        raise ValueError('Invalid seed list')
    return set(values)


def load_batches(paths, allowed_parents, previous_train, previous_val, exclude_seeds=()):
    """Validate the typed per-row provenance before taking any optimization samples."""
    train=set(previous_train); val=set(previous_val); reserved=RESERVED|set(exclude_seeds)
    batch_info=[]; batches=[]; unique_hashes=set(); physics=None; seen=set()
    for name in paths:
        path=Path(name).resolve(); path_report=path.with_name('report.json'); report=_json(path_report)
        digest=_sha(path)
        if (path.name!='batch.npz' or report.get('role')!='collection' or report.get('schema')!=SCHEMA
            or report.get('teacher_label_source')!='projected_90mm_feedback_teacher' or report.get('cube_edge_m')!=.09
            or digest!=report.get('data_sha256') or report.get('parent_sha') not in allowed_parents
            or report.get('physics_device')!='cuda:0' or report.get('control_hz')!=30 or report.get('physics_hz')!=240
            or report.get('action_dim')!=56 or report.get('eval_seeds')!=[] or report.get('cross_lane_contacts')!=0):
            raise ValueError(f'Ineligible physical collection or lineage: {path}')
        if digest in unique_hashes:raise ValueError('Duplicate collection')
        unique_hashes.add(digest)
        sig=tuple(report[k] for k in ('grasp_sha256','asset_sha256'))
        if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in sig):
            raise ValueError('Missing physical geometry hash')
        if physics is not None and sig!=physics:raise ValueError('Mixed 90 mm geometry/calibration')
        physics=sig
        ts,vs,all_=_ints(report['train_seeds']),_ints(report['loss_val_seeds']),_ints(report['seeds'])
        if (not ts or not vs or ts|vs!=all_ or ts&vs or all_&reserved or ts&val or vs&train
                or min(all_)<10_000_000 or max(all_)>=20_000_000):
            raise ValueError('Invalid seed split: held-out, previously trained, or plot-only data')
        if seen&all_:raise ValueError('Duplicate physical seeds across collection files')
        seen|=all_;train|=ts;val|=vs
        with np.load(path,allow_pickle=False) as npz:
            x,y,mask,phase,seeds,v=(npz[k] for k in ('x','y','mask','phase','seed','val'))
        n=len(x)
        if (n<2 or x.shape!=(n,FEATURES) or y.shape!=(n,28) or mask.shape!=y.shape or
            any(a.shape!=(n,) for a in (phase,seeds,v)) or any(a.dtype!=np.float32 for a in (x,y)) or
            mask.dtype!=bool or v.dtype!=bool or phase.dtype.kind not in 'iu' or seeds.dtype.kind not in 'iu'
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not mask.any(axis=1).all()
            or phase.min()<0 or phase.max()>=len(PHASES)):
            raise ValueError('Invalid collection arrays')
        if not np.array_equal(x[:,-len(PHASES)-1:-1],np.eye(len(PHASES),dtype=np.float32)[phase]):
            raise ValueError('Phase tags disagree with observed actor inputs')
        if not set(map(int,np.unique(seeds))).issubset(all_) or not np.array_equal(v,np.isin(seeds,list(vs))) or not v.any() or v.all():
            raise ValueError('Row seed/split disagreement')
        batches.append((x,y,mask,phase.astype(np.int64),v))
        batch_info.append(dict(path=str(path),data_sha256=digest,report_sha256=_sha(path_report),parent_sha=report['parent_sha'],
                    rows=n,train_rows=int((~v).sum()),val_rows=int(v.sum()),grasp_sha256=sig[0],asset_sha256=sig[1]))
    if not batches:raise ValueError('Empty fit corpus')
    return tuple(np.concatenate(parts) for parts in zip(*batches)),batch_info,train,val


class Balanced:
    """Uniform over distinct training phases; no row or seed from the validation split."""
    def __init__(self,x,y,mask,phase,v,device):
        self.phase_rows={};self.tables=[]
        for split in (False,True):
            keep=(v==split)
            indices=np.flatnonzero(keep)
            indices=indices[np.argsort(phase[indices],kind='stable')]
            labels,sizes=np.unique(phase[indices],return_counts=True)
            slices=np.cumsum(np.r_[0,sizes])
            self.phase_rows['val' if split else 'train']={PHASES[int(p)]:int(s) for p,s in zip(labels,sizes)}
            a=tuple(torch.as_tensor(arr[indices],device=device) for arr in (x,y,mask))
            self.tables.append((a,labels,sizes,slices))

    def sample(self,batch,step,gen):
        (x,y,m),labels,sizes,bounds=self.tables[0]
        if batch<len(labels):raise ValueError('Batch smaller than number of training phases')
        k=len(labels); base,extra=divmod(batch,k)
        # Rotate remainder so every phase gets exactly the same mass in k steps.
        idx=torch.cat([torch.randint(int(bounds[j]),int(bounds[j+1]),(base+int((j-step)%k<extra),),
                       generator=gen,device=x.device) for j in range(k)])
        return x[idx],y[idx],m[idx]

    @torch.inference_mode()
    def score(self,model):
        (x,y,m),labels,sizes,bounds=self.tables[1]
        phase={}
        for j,label in enumerate(labels):
            loss=torch.zeros((),device=x.device,dtype=torch.float64)
            mae=torch.zeros_like(loss)
            for a in range(int(bounds[j]),int(bounds[j+1]),4096):
                b=min(a+4096,int(bounds[j+1]));diff=model(x[a:b])-y[a:b];mask=m[a:b];den=mask.sum(-1)
                loss+=(((diff/model.scale).square()*mask).sum(-1)/den).sum(dtype=torch.float64)
                mae+=((diff.abs()*mask).sum(-1)/den).sum(dtype=torch.float64)
            phase[PHASES[int(label)]]=dict(normalized_mse=loss.item()/int(sizes[j]),mae_rad=mae.item()/int(sizes[j]),rows=int(sizes[j]))
        result=dict(phase_balanced_normalized_mse=float(np.mean([m['normalized_mse'] for m in phase.values()])),
                    phase_balanced_mae_rad=float(np.mean([m['mae_rad'] for m in phase.values()])),by_phase=phase)
        if not math.isfinite(result['phase_balanced_normalized_mse']):raise ValueError('Nonfinite validation loss')
        return result


def _host(obj):
    if torch.is_tensor(obj):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:_host(v) for k,v in obj.items()}
    if isinstance(obj,list):return [_host(v) for v in obj]
    if isinstance(obj,tuple):return tuple(_host(v) for v in obj)
    return obj


def fit(parent,data,outdir,batch=32768,updates=5000,lr=5e-5,*,device='cuda:0',save_every=250,seed=90017,
        exclude_seeds=(),stop_after=None):
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
        return _fit(Path(parent).resolve(),data,out,batch,updates,lr,device,save_every,seed,exclude_seeds,stop_after)


def _fit(parent,data,out,batch,updates,lr,device,save_every,seed,exclude,stop_after):
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
    names=('contact_hold90_cuda_fit.py','contact_hold_policy.py','phase_contact_policy.py')
    code={n:_sha(Path(__file__).with_name(n)) for n in names}
    spec=dict(parent_sha256=parent_sha,parent=str(parent),collections=details,batch=batch,updates=updates,lr=lr,
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
    groups=Balanced(x,y,mask,ph,v,device); del arrays,x,y,mask,ph,v
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
                 seed=seed,training_samples=sum(groups.phase_rows['train'].values()),validation_samples=sum(groups.phase_rows['val'].values()),
                 closed_loop_success_demonstrated=False,training_seeds=sorted(train_seeds),loss_val_seeds_90=sorted(val_seeds),
                 validation_kind='collection_seed_holdout_loss_only',asset_sha256=details[0]['asset_sha256'],grasp_sha256=details[0]['grasp_sha256'],
                 long_gpu_updates=int(meta.get('long_gpu_updates',0))+best_so_far['step'],
                 physical90_scheduled_updates=int(meta.get('physical90_scheduled_updates',0))+step)
        _put(best,dict(model=best_so_far['model'],metadata=best_meta,metrics=best_so_far['score'],
                       physical90_optimizer=best_so_far['optimizer']),True)
    if not last.exists():persist(0)
    end=updates if stop_after is None else max(start,stop_after)
    for step in range(start+1,end+1):
        bx,by,bm=groups.sample(batch,step,gen)
        err=(model(bx)-by)/model.scale
        loss=(err.square().mul(bm).sum(-1)/bm.sum(-1)).mean()
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
                last_metrics=latest,phase_rows=groups.phase_rows,checkpoint_selection='lowest seed-held-out phase-balanced normalized MSE',
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
    args=vars(p.parse_args());print(json.dumps(fit(**args),allow_nan=False,indent=2))
