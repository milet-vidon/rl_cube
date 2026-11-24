"""Warm-start contact-hold BC with reviewed, counterfactual preshape arm labels."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from contact_hold_dataset import audit_demo, sha, training_pairs
from contact_hold_policy import restored
from phase_contact_policy import features
from prepare_contact_hold_arm_corrections import prepare
from train_contact_hold_bc import atomic_save

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/'runs/contact_hold_bc_20261107_143712_491446697'
CANDIDATES = ROOT.parent/'work/teacher_optimization_20261103/contact_hold_preshape_candidates_11601_20261107_150520'
# This review covers this particular development rollout, not arbitrary failed evaluations.
SUMMARY_SHA = '0e367dfab1a691f732b05494ac4bb5c30643815132248d26d9a31956dd115a77'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reviewed_data():
    summary_path = CANDIDATES/'summary.json'
    if sha(summary_path) != SUMMARY_SHA:
        raise ValueError('Unreviewed correction summary')
    summary = read(summary_path)
    rollout = Path(summary['source'])
    packed = CANDIDATES/'preshape_left_arm.npz'
    if sha(rollout/'artifact_manifest.json') != summary['manifest_sha256'] or sha(packed) != summary['data_sha256']:
        raise ValueError('Correction rollout or NPZ differs from reviewed evidence')
    # Repeat safety, nominal teacher and actual student projection checks from the rollout.
    with TemporaryDirectory() as tmp:
        regenerated = Path(tmp)/'replayed'
        prepare(rollout, regenerated)
        replay = read(regenerated/'summary.json')
        if replay != summary:
            raise ValueError('Reviewed evidence does not reproduce')
        with np.load(packed, allow_pickle=False) as orig, np.load(regenerated/packed.name, allow_pickle=False) as new:
            if set(orig.files) != set(new.files) or any(not np.array_equal(orig[n], new[n]) for n in orig.files):
                raise ValueError('Replayed inputs or projected arm labels differ')
            x, y, steps, joints = (orig[n] for n in ('features','projected_teacher_arm_rad','steps','output_names'))
    if (not np.array_equal(steps, np.arange(305,497)) or x.shape != (192,198) or y.shape != (192,7)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Unexpected preshape data')
    rows = read(rollout/'trajectory.json')
    phase_start = 0
    for i, row in enumerate(rows):
        if i == 0 or row['phase'] != rows[i-1]['phase']:
            phase_start = i
        if steps[0] <= i <= steps[-1]:
            expected = features(rows[i-1], row['phase'], min((i-phase_start)/300., 1.))
            if not np.array_equal(x[i-steps[0]], expected):
                raise ValueError('Correction differs from independently timed pre-action features')
    val = ((steps-steps[0])//30) % 5 == 4
    review = dict(candidate_path=str(CANDIDATES), candidate_summary_sha256=SUMMARY_SHA,
                  rollout_manifest_sha256=summary['manifest_sha256'], data_sha256=summary['data_sha256'],
                  rollout_source=str(rollout), original_failure=summary['original_failure'],
                  source_rollout_eligible_as_demonstration=False, teacher_correction_executed=False,
                  scope='development_only_projected_preshape_arm_targets', seed=summary['seed'],
                  output_names=joints.tolist(), first_step=int(steps[0]), last_step=int(steps[-1]),
                  terminal_step_excluded=summary['terminal_step_excluded'],
                  training_samples=int((~val).sum()), validation_samples=int(val.sum()))
    return (x, y, val), review, summary['checkpoint_sha256']


def parent_data(expected_digest):
    result = read(PARENT/'result.json')
    if not (result['completed'] and result['checkpoint_reload_verified']
            and sha(PARENT/'best.pt') == result['checkpoint_sha256'] == expected_digest):
        raise ValueError('Wrong starting checkpoint')
    saved = torch.load(PARENT/'best.pt', map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if meta != read(PARENT/'experiment.json') or saved['metrics']['update'] != result['best_update']:
        raise ValueError('Parent metadata mismatch')
    for name, digest in meta['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed parent training dependency: '+name)
    arrays = []
    for source in meta['sources']:
        rows, names, ids, _, audit = audit_demo(Path(source['path']))
        if audit != source or names != meta['joint_names'] or ids != meta['output_indices']:
            raise ValueError('Parent demonstration audit did not reproduce')
        arrays.append(training_pairs(rows, ids))
    return restored(saved), meta, tuple(np.concatenate(items) for items in zip(*arrays))


def residuals(pred, labels, columns=None):
    return (pred if columns is None else pred.index_select(1, columns)) - labels


@torch.no_grad()
def metrics(model, dev, cols, weight):
    (x, y), (cx, cy) = dev
    d = residuals(model(x), y)
    c = residuals(model(cx), cy, cols)
    scaled = (d/model.scale).square().mean() + weight*(c/model.scale[cols]).square().mean()
    values = dict(validation_rmse_rad=float(d.square().mean().sqrt()),
                  validation_arm_correction_rmse_rad=float(c.square().mean().sqrt()),
                  validation_scaled_loss=float(scaled))
    if not all(map(math.isfinite, values.values())):
        raise ValueError('Nonfinite validation')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=65536, help='total, split evenly across the two losses')
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--correction-weight', type=float, default=1.)
    args = parser.parse_args()
    if (args.output.exists() or args.batch_size < 2 or args.batch_size % 2 or args.updates < 1
            or not all(math.isfinite(v) and v > 0 for v in (args.learning_rate,args.correction_weight))):
        parser.error('Need a fresh output directory, positive updates/rates and a positive even batch size')
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    corr, review, parent_sha = reviewed_data()
    model, parent_meta, demo = parent_data(parent_sha)
    left_names = [parent_meta['joint_names'][i] for i in parent_meta['output_indices']]
    colids = [left_names.index(n) for n in review['output_names']]
    if np.any(corr[1] < model.lower.numpy()[colids]-1e-6) or np.any(corr[1] > model.upper.numpy()[colids]+1e-6):
        raise ValueError('Corrective supervision outside policy bounds')
    hashes = dict(parent_meta['source_sha256'])
    for n in ('prepare_contact_hold_arm_corrections.py', 'train_contact_hold_corrected.py'):
        hashes[n] = sha(ROOT/n)
    meta = dict(parent_meta, source_sha256=hashes, seed=20261107, device=str(device), updates=args.updates,
                batch_size=args.batch_size, learning_rate=args.learning_rate, algorithm='warm_started_BC_with_projected_arm_supervision',
                training_samples=int((~demo[2]).sum()+(~corr[2]).sum()), validation_samples=int(demo[2].sum()+corr[2].sum()),
                validation_kind='within_source_temporal_blocks_development_only',
                parent_run=str(PARENT), parent_checkpoint_sha256=parent_sha, optimizer_reset=True,
                correction_review=review, correction_weight=args.correction_weight)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for name in hashes:
        (snapshot/name).write_bytes((ROOT/name).read_bytes())
    write(args.output/'correction_review.json', review)
    write(args.output/'experiment.json', meta)
    def partition(heldout, target):
        return [tuple(torch.from_numpy(a[val if heldout else ~val]).to(target) for a in (x,y))
                for x, y, val in (demo,corr)]
    train, dev = partition(False,device), partition(True,device)
    model.to(device).train()
    cols = torch.tensor(colids, device=device)
    half = args.batch_size//2
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = float('inf')
    with (args.output/'metrics.jsonl').open('x') as log:
        for step in range(args.updates+1):
            if step:
                (dx,dy),(cx,cy) = train
                di = torch.randint(len(dx), (half,), device=device)
                ci = torch.randint(len(cx), (half,), device=device)
                dp,cp = model(torch.cat([dx[di],cx[ci]])).split(half)
                d = residuals(dp,dy[di])/model.scale
                c = residuals(cp,cy[ci],cols)/model.scale[cols]
                loss = d.square().mean() + args.correction_weight*c.square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
            if step not in (0,1,args.updates) and step % 200:
                continue
            rec = dict(update=step, **metrics(model, dev, cols, args.correction_weight))
            print('BC_PROGRESS', json.dumps(rec), flush=True)
            log.write(json.dumps(rec)+'\n')
            log.flush()
            if rec['validation_scaled_loss'] < best:
                best = rec['validation_scaled_loss']
                atomic_save(dict(metadata=meta, model={n:t.detach().cpu() for n,t in model.state_dict().items()},
                                 metrics=rec), args.output/'best.pt')
    for name, digest in hashes.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Training dependency changed: '+name)
    for source in meta['sources']:
        path = Path(source['path'])
        for name, digest in dict(source['evidence_sha256'], **{'artifact_manifest.json': source['manifest_sha256']}).items():
            if sha(path/name) != digest:
                raise ValueError('Teacher evidence changed: '+name)
    if (sha(PARENT/'best.pt') != parent_sha or sha(CANDIDATES/'summary.json') != SUMMARY_SHA
            or sha(CANDIDATES/'preshape_left_arm.npz') != review['data_sha256']
            or sha(Path(review['rollout_source'])/'artifact_manifest.json') != review['rollout_manifest_sha256']):
        raise ValueError('Correction sources changed during training')
    saved = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    checked = metrics(restored(saved), partition(True,'cpu'), torch.tensor(colids), args.correction_weight)
    for key in ('validation_rmse_rad','validation_arm_correction_rmse_rad'):
        if abs(checked[key]-saved['metrics'][key]) > 1e-5:
            raise ValueError('CPU reload does not reproduce '+key)
    result = dict(completed=True, updates=args.updates, device=str(device), best_update=saved['metrics']['update'],
                  **checked, reload_rmse_rad=checked['validation_rmse_rad'],
                  checkpoint_sha256=sha(args.output/'best.pt'), checkpoint_reload_verified=True,
                  closed_loop_success_demonstrated=False, resumes_v11=False)
    write(args.output/'result.json', result)
    print('TRAIN_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
