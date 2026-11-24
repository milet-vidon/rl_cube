"""Second 11601 development BC correction, retaining both student distributions."""
import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from contact_hold_dataset import sha
from contact_hold_policy import restored
from phase_contact_policy import features
from prepare_contact_hold_arm_corrections import prepare
from train_contact_hold_bc import atomic_save
from train_contact_hold_corrected import (ROOT, CANDIDATES, read, write,
                                         reviewed_data, parent_data, residuals)

PARENT2 = ROOT/'runs/contact_hold_corrected_20261107_152443_486760012'
NEW = CANDIDATES.with_name('contact_hold_preshape_candidates_11601_20261107_153509')
NEW_SHA = '0ca39c235a04717fc8209bad8971bd544ba82832642b25a74b386add680a73b6'
GROUPS = ('demo', 'correction1', 'correction2')
WEIGHTS = (1., .5, .5)


def new_data():
    if sha(NEW/'summary.json') != NEW_SHA:
        raise ValueError('Unreviewed round-two candidate summary')
    s = read(NEW/'summary.json')
    rollout = Path(s['source'])
    data = NEW/'preshape_left_arm.npz'
    if sha(rollout/'artifact_manifest.json') != s['manifest_sha256'] or sha(data) != s['data_sha256']:
        raise ValueError('Changed round-two rollout or targets')
    with tempfile.TemporaryDirectory() as tmp:
        replay = Path(tmp)/'replay'
        prepare(rollout, replay)
        if read(replay/'summary.json') != s or sha(replay/data.name) != s['data_sha256']:
            raise ValueError('Safety replay does not reproduce second dataset')
    with np.load(data, allow_pickle=False) as pack:
        x,y,steps,joints = [pack[k] for k in ('features','projected_teacher_arm_rad','steps','output_names')]
    if (not np.array_equal(steps,np.arange(247,439)) or y.shape != (192,7)
            or joints.tolist() != [f'Joint{i}_L' for i in range(1,8)] or not np.isfinite(y).all()):
        raise ValueError('Unexpected second correction schema')
    rows = read(rollout/'trajectory.json')
    if ([r['phase'] for r in rows] != ['ready']*247+['preshape']*193 or
            not np.array_equal(x, np.stack([features(rows[i-1],'preshape',(i-247)/300.) for i in steps]))):
        raise ValueError('Correction input differs from causal phase clock')
    val = ((steps-247)//30)%5 == 4
    review = dict(candidate_path=str(NEW), candidate_summary_sha256=NEW_SHA, rollout_source=str(rollout),
                  rollout_manifest_sha256=s['manifest_sha256'], data_sha256=s['data_sha256'],
                  behavior_checkpoint_sha256=s['checkpoint_sha256'], output_names=joints.tolist(),
                  scope='development_only_projected_preshape_arm_targets', seed=s['seed'],
                  source_rollout_eligible_as_demonstration=False, teacher_correction_executed=False,
                  first_step=247, last_step=438, terminal_step_excluded=439, original_failure=s['original_failure'],
                  training_samples=int((~val).sum()), validation_samples=int(val.sum()))
    return (x,y,val), review


def sources():
    c1, r1, generation0 = reviewed_data()
    _, base_meta, demo = parent_data(generation0)
    c2, r2 = new_data()
    path = PARENT2/'best.pt'
    summary = read(PARENT2/'result.json')
    if not (summary['completed'] and summary['checkpoint_reload_verified'] and
            sha(path) == summary['checkpoint_sha256'] == r2['behavior_checkpoint_sha256']):
        raise ValueError('The reviewed second rollout did not come from this checkpoint')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    meta = ck['metadata']
    if (meta != read(PARENT2/'experiment.json') or ck['metrics']['update'] != summary['best_update']
            or meta['correction_review'] != r1 or meta['parent_checkpoint_sha256'] != generation0
            or any(meta[k] != base_meta[k] for k in ('sources','joint_names','output_indices','width'))
            or r2['output_names'] != r1['output_names']):
        raise ValueError('Parent lineage or output mapping does not match the review')
    for name,h in meta['source_sha256'].items():
        if sha(ROOT/name) != h:
            raise ValueError('Parent source changed: '+name)
    return restored(ck), meta, (demo,c1,c2), [dict(r1,behavior_checkpoint_sha256=generation0),r2]


def partition(groups, heldout, device):
    return [tuple(torch.from_numpy(a[val if heldout else ~val]).to(device) for a in (x,y))
            for x,y,val in groups]


@torch.no_grad()
def assess(model, dev, cols):
    score = 0.
    report = {}
    for i,(x,y) in enumerate(dev):
        e = residuals(model(x), y, cols if i else None)
        score += WEIGHTS[i]*float((e/(model.scale[cols] if i else model.scale)).square().mean())
        report[GROUPS[i]+'_rmse_rad'] = float(e.square().mean().sqrt())
    report['validation_scaled_loss'] = score
    if not all(map(math.isfinite,report.values())):
        raise ValueError('Nonfinite validation')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=65536)
    p.add_argument('--learning-rate',type=float,default=5e-5)
    p.add_argument('--device',default='cuda:0')
    args = p.parse_args()
    if (args.output.exists() or args.updates < 1 or args.batch_size < 4 or args.batch_size%4
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        p.error('Need fresh output directory, batch multiple of four, and positive updates and rate')
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    device = torch.device(args.device)
    model, parent, groups, reviews = sources()
    left = [parent['joint_names'][i] for i in parent['output_indices']]
    colids = [left.index(n) for n in reviews[0]['output_names']]
    for x,y,val in groups[1:]:
        if np.any(y<model.lower.numpy()[colids]-1e-6) or np.any(y>model.upper.numpy()[colids]+1e-6):
            raise ValueError('Correction outside saved bounds')
    hashes = dict(parent['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    out = args.output
    meta = {k:v for k,v in parent.items() if k not in ('correction_review','correction_weight')}
    meta.update(algorithm='round2_BC_demo28_two_correction_distributions_arm7', parent_run=str(PARENT2),
                parent_checkpoint_sha256=reviews[1]['behavior_checkpoint_sha256'], correction_reviews=reviews,
                seed=20261107, device=str(device), learning_rate=args.learning_rate, updates=args.updates,
                batch_size=args.batch_size, source_sha256=hashes, group_loss_weights=list(WEIGHTS),
                training_samples=sum(int((~v).sum()) for _,_,v in groups),
                validation_samples=sum(int(v.sum()) for _,_,v in groups),
                checkpoint_selection='lower_weighted_validation_and_better_round2_rmse_than_parent')
    train,dev = partition(groups,False,device),partition(groups,True,device)
    cols = torch.tensor(colids,device=device)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    out.mkdir(parents=True,exist_ok=False)
    snap = out/'source_snapshot'
    snap.mkdir()
    for name in hashes:
        (snap/name).write_bytes((ROOT/name).read_bytes())
    write(out/'experiment.json',meta)
    write(out/'correction_reviews.json',reviews)
    sizes = (args.batch_size//2,args.batch_size//4,args.batch_size//4)
    best = float('inf')
    with (out/'metrics.jsonl').open('x') as log:
        for step in range(args.updates+1):
            if step:
                ids = [torch.randint(len(x),(n,),device=device) for (x,_),n in zip(train,sizes)]
                preds = model(torch.cat([x[idx] for (x,_),idx in zip(train,ids)])).split(sizes)
                err = [residuals(yhat, y[idx], cols if i else None)/(model.scale[cols] if i else model.scale)
                       for i,(yhat,(_,y),idx) in enumerate(zip(preds,train,ids))]
                loss = sum(w*e.square().mean() for w,e in zip(WEIGHTS,err))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            if step not in (0,1,args.updates) and step%50:
                continue
            rec = dict(update=step,**assess(model,dev,cols))
            print('BC_PROGRESS',json.dumps(rec),flush=True)
            log.write(json.dumps(rec)+'\n');log.flush()
            if step == 0:
                baseline = rec
            if rec['validation_scaled_loss'] < best and (step == 0 or rec['correction2_rmse_rad'] < baseline['correction2_rmse_rad']):
                best = rec['validation_scaled_loss']
                atomic_save(dict(metadata=meta, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                 metrics=rec),out/'best.pt')
    # Recheck both actual rollouts/labels, audited teacher evidence and parent checkpoint at close.
    _,end_meta,_,end_reviews = sources()
    if end_meta != parent or end_reviews != reviews or any(sha(ROOT/n) != h for n,h in hashes.items()):
        raise ValueError('Training evidence changed during fit')
    saved = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    checked = assess(restored(saved),partition(groups,True,'cpu'),torch.tensor(colids))
    if any(abs(checked[n+'_rmse_rad']-saved['metrics'][n+'_rmse_rad'])>1e-5 for n in GROUPS):
        raise ValueError('Saved weights do not reproduce validation on all sources')
    result = dict(completed=True,updates=args.updates,device=str(device),best_update=saved['metrics']['update'],
                  checkpoint_reload_verified=True,checkpoint_sha256=sha(out/'best.pt'),
                  reload_rmse_rad=checked['demo_rmse_rad'],validation=checked,
                  closed_loop_success_demonstrated=False,resumes_v11=False)
    write(out/'result.json',result)
    print('TRAIN_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
