"""CUDA BC on newly audited successful teacher episodes; not V11 resume/PPO.

The policy consumes measured state plus the public controller phase/elapsed
time. All batches stay on the selected device. A temporal validation split is
development-only; a completed fit is not a closed-loop success certificate.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from contact_hold_dataset import SCHEMA, audit_demo, training_pairs, sha
from contact_hold_policy import ContactHoldPolicy


def atomic_save(data, path):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; use a new run name')
    if min(args.updates, args.batch_size, args.hidden_size) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('Training parameters must be positive and finite')
    if len({r.resolve() for r in args.source}) != len(args.source):
        p.error('Duplicate source')
    arrays, provenance = [], []
    names = indices = bounds = candidate = physical_contract = None
    from kinematics import RobotKinematics
    k = RobotKinematics()
    for run in args.source:
        rows, nn, ii, contract, meta = audit_demo(run)
        if names is not None and (names != nn or indices != ii or candidate != meta['candidate']):
            raise ValueError('Different joint layout or candidate control settings across sources')
        if physical_contract is not None and physical_contract != contract:
            raise ValueError('Different physical asset contracts across sources')
        physical_contract = contract
        names, indices, candidate = nn, ii, meta['candidate']
        limits = np.asarray([k.limits[names[i]] for i in indices], dtype=np.float32)
        for j, i in enumerate(indices):
            soft = contract['joint_limit_safety']['soft_limits_rad'].get(names[i])
            if soft is not None:
                limits[j] = [max(limits[j, 0], soft[0]), min(limits[j, 1], soft[1])]
        bounds = limits if bounds is None else np.stack((np.maximum(bounds[:, 0], limits[:, 0]), np.minimum(bounds[:, 1], limits[:, 1])), axis=1)
        arrays.append(training_pairs(rows, indices))
        provenance.append(meta)
    x, y, val = [np.concatenate([a[i] for a in arrays]) for i in range(3)]
    if not val.any() or val.all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Invalid validation split or limit intersection')
    if np.any(y < bounds[:, 0]-1e-6) or np.any(y > bounds[:, 1]+1e-6):
        raise ValueError('Demonstration target outside safety intersection')
    sources = ('contact_hold_dataset.py', 'train_contact_hold_bc.py', 'contact_hold_policy.py',
               'phase_contact_policy.py', 'wrist_audit.py', 'elbow_gate.py', 'elbow_audit.py',
               'palm_handover_audit.py', 'tilt_audit.py', 'kinematics.py', 'joint_safety.py')
    root = Path(__file__).resolve().parent
    hashes = {name: sha(root/name) for name in sources}
    metadata = dict(schema=SCHEMA, sources=provenance, joint_names=names, output_indices=indices,
                    width=args.hidden_size, device=args.device, seed=args.seed,
                    updates=args.updates, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    training_samples=int((~val).sum()), validation_samples=int(val.sum()),
                    source_sha256=hashes, phase_clock='min(elapsed_control_steps_in_current_public_phase/300,1)',
                    validation_kind='temporal_blocks_of_seen_teacher_episodes_not_independent',
                    initial_reset_action_supervised=False, resumes_v11=False, algorithm='supervised_state_BC',
                    original_physical_gates_retained=True, closed_loop_success_demonstrated=False)
    # Fail before creating an optimizer/output if the requested GPU is unavailable.
    device = torch.device(args.device)
    if not args.audit_only and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; run from the GPU host terminal')
    args.output.mkdir(parents=True)
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((root/name).read_bytes())
    (args.output/'experiment.json').write_text(json.dumps(metadata, indent=2))
    (args.output/'data_audit.json').write_text(json.dumps(dict(passed=True, sources=provenance), indent=2))
    if args.audit_only:
        print('DATA_AUDIT_PASSED', str(args.output), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ContactHoldPolicy(x[~val].mean(0), np.maximum(x[~val].std(0), .02), bounds[:, 0], bounds[:, 1],
                              y[~val].mean(0), np.maximum(y[~val].std(0), .05), args.hidden_size).to(device)
    X, Y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    train_ids, val_ids = [torch.as_tensor(np.flatnonzero(mask), device=device) for mask in (~val, val)]
    scale = model.scale.detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = float('inf')
    started = time.monotonic()
    print('TRAIN_START', json.dumps(dict(device=str(device), samples=len(x), updates=args.updates)), flush=True)
    with (args.output/'metrics.jsonl').open('w') as log:
        for step in range(1, args.updates+1):
            ids = train_ids[torch.randint(len(train_ids), (args.batch_size,), device=device)]
            loss = ((model(X[ids])-Y[ids])/scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 200 == 0 or step == args.updates:
                with torch.no_grad():
                    residual = model(X[val_ids])-Y[val_ids]
                    score = float((residual/scale).square().mean())
                    rmse = float(residual.square().mean().sqrt())
                record = dict(update=step, loss=float(loss.detach()), validation_rmse_rad=rmse,
                              validation_scaled_loss=score, elapsed_s=time.monotonic()-started,
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
                log.write(json.dumps(record)+'\n'); log.flush()
                print('BC_PROGRESS', json.dumps(record), flush=True)
                if score < best:
                    best = score
                    atomic_save(dict(metadata=metadata, model={n:v.detach().cpu() for n,v in model.state_dict().items()}, metrics=record), args.output/'best.pt')
    if hashes != {name: sha(root/name) for name in sources}:
        raise RuntimeError('Training sources changed during optimization')
    for source in provenance:
        if sha(Path(source['path'])/'artifact_manifest.json') != source['manifest_sha256']:
            raise RuntimeError('Demonstration manifest changed during optimization')
        for name, digest in source['evidence_sha256'].items():
            if sha(Path(source['path'])/name) != digest:
                raise RuntimeError('Demonstration bytes changed during optimization: '+name)
    saved = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
    model.cpu().load_state_dict(saved['model'])
    with torch.no_grad():
        error = model(torch.from_numpy(x[val])) - torch.from_numpy(y[val])
        reload_rmse = float(error.square().mean().sqrt())
    if abs(reload_rmse-saved['metrics']['validation_rmse_rad']) > 1e-5:
        raise RuntimeError('CPU checkpoint reload does not match validation')
    result = dict(completed=True, updates=args.updates, device=str(device),
                  best_update=saved['metrics']['update'], reload_rmse_rad=reload_rmse,
                  checkpoint_sha256=sha(args.output/'best.pt'), checkpoint_reload_verified=True,
                  closed_loop_success_demonstrated=False, resumes_v11=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print('TRAIN_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
