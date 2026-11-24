"""One resumable BC update block of the online teacher-query loop (fresh process per fit)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from contact_hold_dataset import sha, verified_json
from contact_hold_loop_data import INITIAL, INITIAL_SHA, ROOT, checkpoint, read, write
from contact_hold_policy import restored
from train_contact_hold_bc import atomic_save
from train_contact_hold_round2 import sources

FILES = ('contact_hold_loop_data.py','contact_hold_loop_fit.py','contact_hold_continuous.py',
         'contact_hold_student_eval.py','single_face_experiment.py','run_contact_hold_evaluation.sh')


def verify_collection(c, allowed):
    s=c['source'];run=Path(s['run'])
    if (s['role']!='collection' or s['seed'] not in allowed or sha(Path(c['path']))!=c['sha256']
            or sha(run/'artifact_manifest.json')!=s['manifest']):
        raise ValueError('Changed or ineligible correction corpus')
    m=read(run/'artifact_manifest.json')
    proto=verified_json(run,'student_evaluation.json',m)
    if (m['split']!='development' or not m['full_episode'] or proto.get('evaluation_role')!='collection'
            or proto['seed']!=s['seed'] or proto['checkpoint_sha256']!=s['checkpoint']
            or proto['training_run']!=s['fit'] or m['sha256']['evaluated_best.pt']!=s['checkpoint']):
        raise ValueError('Correction did not originate in the declared collection rollout')


def build(plan, parent):
    if checkpoint(INITIAL)!=INITIAL_SHA or checkpoint(plan['parent'])!=plan['parent_sha']:
        raise ValueError('Changed parent')
    _, oldmeta, history, reviews = sources()
    if (oldmeta['sources']!=parent['sources'] or reviews!=parent['correction_reviews']
            or sha(ROOT/'train_contact_hold_round2.py')!=parent['source_sha256']['train_contact_hold_round2.py']):
        raise ValueError('Base training lineage mismatch')
    left = [parent['joint_names'][i] for i in parent['output_indices']]
    arm = [left.index(n) for n in reviews[0]['output_names']]
    groups = []
    for i,(x,y,val) in enumerate(history):
        yy,mask = np.zeros((len(x),28),np.float32),np.zeros((len(x),28),bool)
        cols = arm if i else list(range(28))
        yy[:,cols] = y; mask[:,cols] = True
        groups.append((x,yy,mask,val))
    recent = []
    for c in plan['corpus']:
        verify_collection(c,plan['collection_seeds'])
        with np.load(c['path'],allow_pickle=False) as a:
            recent.append(tuple(a[k] for k in ('x','y','mask','val')))
    if not recent:
        raise ValueError('No eligible online teacher queries')
    merge = lambda values: tuple(np.concatenate(parts) for parts in zip(*values))
    return [groups[0],merge(groups[1:]),merge(recent)]


def run(planpath, device='cuda:0', stop_after=None):
    planpath=Path(planpath)
    plan=read(planpath); root=planpath.parent; sig=sha(planpath)
    dev=torch.device(device)
    torch.set_num_threads(4)
    parent=Path(plan['parent'])
    if checkpoint(parent)!=plan['parent_sha']:
        raise ValueError('Starting policy changed')
    c=torch.load(parent/'best.pt',map_location='cpu',weights_only=True)
    meta=c['metadata']
    for n,h in meta['source_sha256'].items():
        if sha(ROOT/n)!=h: raise ValueError('Changed model dependency: '+n)
    model=restored(c).to(dev)
    groups=build(plan,meta)
    def split(val):
        return [tuple(torch.from_numpy(a[v if val else ~v]).to(dev) for a in (x,y,m)) for x,y,m,v in groups]
    train,valid=split(False),split(True)
    sizes=(plan['batch_size']//2,plan['batch_size']//4,plan['batch_size']//4)
    opt=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=1e-5)
    gen=torch.Generator(device=dev).manual_seed(20261109+plan['round'])
    start=0
    inherited=int(meta.get('global_update',0))
    def load_state(path):
        s=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        gen.set_state(s['sampler_state'])
        return s
    last=root/'last.pt'
    if last.exists():
        s=load_state(last)
        if s['plan_sha']!=sig or s['device']!=device or s['global_update']!=inherited+s['update']:
            raise ValueError('Incompatible training resume')
        start=s['update']
    elif 'continuous_plan_sha256' in meta:
        if sha(parent/'last.pt')!=read(parent/'result.json')['last_sha256']:
            raise ValueError('Parent optimizer checkpoint changed')
        s=load_state(parent/'last.pt')
        if s['global_update']!=inherited or s['plan_sha']!=meta['continuous_plan_sha256'] or s['device']!=device:
            raise ValueError('Optimizer lineage mismatch')
    def loss(pred,y,mask):
        return ((((pred-y)/model.scale).square()*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()
    @torch.inference_mode()
    def metrics():
        vals=[float(loss(model(x),y,m)) if len(x) else None for x,y,m in valid]
        if any(a is not None and not math.isfinite(a) for a in vals):
            raise ValueError('Nonfinite validation')
        return vals
    for k in range(start+1,plan['updates']+1):
        ids=[torch.randint(len(x),(n,),device=dev,generator=gen) for (x,_,_),n in zip(train,sizes)]
        preds=model(torch.cat([x[t] for (x,_,_),t in zip(train,ids)])).split(sizes)
        losses=[loss(p,y[t],m[t]) for p,(_,y,m),t in zip(preds,train,ids)]
        total=losses[0]+.5*losses[1]+losses[2]
        opt.zero_grad(set_to_none=True);total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if k==1 or k%50==0 or k==plan['updates'] or k==stop_after:
            vals=metrics()
            rec=dict(update=k,global_update=inherited+k,validation_scaled_loss=vals)
            atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler_state=gen.get_state(),
                             device=device,plan_sha=sig,update=k,global_update=inherited+k),last)
            with (root/'metrics.jsonl').open('a') as log: print(json.dumps(rec),file=log)
            print('FIT_PROGRESS',json.dumps(rec),flush=True)
            if k==stop_after and k!=plan['updates']: return
    # The exact end-of-round policy is evaluated in physics. Champion selection happens there.
    hashes=dict(meta['source_sha256']);hashes.update({n:sha(ROOT/n) for n in FILES})
    meta=dict(meta,algorithm='continuous_DAgger_supervised_BC',device=device,updates=plan['updates'],
              batch_size=plan['batch_size'],learning_rate=plan['lr'],source_sha256=hashes,
              parent_run=str(parent),parent_checkpoint_sha256=plan['parent_sha'],continuous_plan_sha256=sig,
              training_seeds=sorted({s['seed'] for s in meta['sources']}|{c['source']['seed'] for c in plan['corpus']}),
              global_update=inherited+plan['updates'],optimizer_reset='continuous_plan_sha256' not in c['metadata'],
              training_samples=sum(len(a[0]) for a in train),validation_samples=sum(len(a[0]) for a in valid),
              group_loss_weights=[1.,.5,1.],checkpoint_selection='end_of_block_then_physical_validation')
    rec=dict(update=plan['updates'],global_update=meta['global_update'],validation_scaled_loss=metrics())
    write(root/'experiment.json',meta)
    atomic_save(dict(metadata=meta,model={n:t.detach().cpu() for n,t in model.state_dict().items()},metrics=rec),root/'best.pt')
    saved=restored(torch.load(root/'best.pt',map_location='cpu',weights_only=True))
    with torch.inference_mode():
        for x,_,_ in valid:
            if len(x):
                torch.testing.assert_close(saved(x.cpu()),model(x).cpu(),atol=1e-5,rtol=1e-5)
    write(root/'result.json',dict(completed=True,checkpoint_reload_verified=True,updates=plan['updates'],
          global_update=meta['global_update'],best_update=plan['updates'],checkpoint_sha256=sha(root/'best.pt'),
          last_sha256=sha(last),closed_loop_success_demonstrated=False,device=device))
    print('FIT_COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    run(p.parse_args().plan)
