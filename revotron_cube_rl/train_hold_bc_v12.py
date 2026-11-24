"""GPU minibatch BC on accepted new teacher demonstrations; no PPO/DAgger claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from hold_bc_v12 import HoldBC, SCHEMA, load_demonstration, load_model, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--hidden-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if min(a.updates, a.batch_size, a.hidden_size) < 1 or not 0 < a.learning_rate < 1:
        p.error('Invalid training hyperparameters')
    if len({r.resolve() for r in a.run}) != len(a.run):
        p.error('Duplicate demonstration')
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    datasets = [load_demonstration(r) for r in a.run]
    names, left, limits = datasets[0][2:5]
    settings = datasets[0][5]['candidate']['settings']
    for d in datasets:
        if d[2] != names or d[3] != left or not np.array_equal(d[4], limits) or d[5]['candidate']['settings'] != settings:
            raise ValueError('Mixed action/limit/controller domains are not supported')
    provenance = [d[5] for d in datasets]
    if len({d['seed'] for d in provenance}) < 2:
        raise ValueError('Use at least two accepted reset seeds for this multiseed candidate')
    report = dict(schema=SCHEMA, sources=provenance, samples=sum(len(d[0]) for d in datasets),
                  device=a.device, original_physical_gates_recomputed=True,
                  validation_scope='temporal blocks within development demonstrations; not independent evaluation')
    if a.preflight_only:
        print(json.dumps(report, indent=2)); return
    if not a.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; this command requires host GPU access')
    device = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    x = torch.tensor(np.concatenate([d[0] for d in datasets]), device=device)
    y = torch.tensor(np.concatenate([d[1] for d in datasets]), device=device)
    val = torch.tensor(np.concatenate([(np.arange(len(d[0])) // 30) % 5 == 4 for d in datasets]), device=device)
    train_ids = torch.where(~val)[0]
    model = HoldBC(a.hidden_size).to(device)
    with torch.no_grad():
        model.mean.copy_(x[~val].mean(0)); model.std.copy_(x[~val].std(0).clamp_min(.02))
        model.center.copy_(y[~val].mean(0)); model.scale.copy_(y[~val].std(0).clamp_min(.05))
        model.limits.copy_(torch.tensor(limits, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-5)
    a.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name('hold_bc_v12.py')]
    source_sha = {s.name: sha(s) for s in sources}
    for src in sources:
        (a.output / src.name).write_bytes(src.read_bytes())
    meta = report | dict(hidden_size=a.hidden_size, training_seed=a.seed, updates=a.updates,
                         batch_size=a.batch_size, learning_rate=a.learning_rate,
                         joint_names=names, left_indices=left, controller_settings=settings,
                         training_sources_sha256=source_sha, closed_loop_validated=False,
                         training_method='new absolute-target state BC; not V11 resume or PPO')
    (a.output / 'experiment.json').write_text(json.dumps(meta, indent=2))
    print('V12_GPU_TRAIN', torch.cuda.get_device_name(device), 'samples', len(x), flush=True)
    best = float('inf'); selected = None; started = time.monotonic()
    with (a.output / 'metrics.jsonl').open('w') as log:
        for update in range(1, a.updates + 1):
            ids = train_ids[torch.randint(len(train_ids), (a.batch_size,), device=device)]
            loss = ((model(x[ids]) - y[ids]) / model.scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if update == 1 or update % 100 == 0 or update == a.updates:
                with torch.no_grad():
                    error = model(x[val]) - y[val]
                    metrics = dict(update=update, loss=float(loss), validation_rmse_rad=float(error.square().mean().sqrt()),
                                   validation_max_abs_rad=float(error.abs().max()), elapsed_s=time.monotonic()-started)
                log.write(json.dumps(metrics) + '\n'); log.flush()
                print('V12_FIT', json.dumps(metrics), flush=True)
                if metrics['validation_rmse_rad'] < best:
                    best = metrics['validation_rmse_rad']; selected = metrics
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                    metadata=meta, selection=metrics), a.output/'model.pt')
    if source_sha != {s.name: sha(s) for s in sources}:
        raise RuntimeError('Training source changed during fitting')
    for src in provenance:
        if sha(Path(src['run'])/'artifact_manifest.json') != src['manifest_sha256']:
            raise RuntimeError('Demonstration manifest changed during fitting')
    digest = sha(a.output/'model.pt')
    restored, _ = load_model(a.output/'model.pt', digest)
    with torch.no_grad():
        cpu_error = restored(x[val].cpu()) - y[val].cpu()
        rmse = float(cpu_error.square().mean().sqrt())
    if abs(rmse-best) > 1e-6:
        raise RuntimeError('CPU checkpoint reload did not reproduce selected validation error')
    result = dict(completed=True, updates=a.updates, selected_update=selected['update'],
                  checkpoint_sha256=digest, cpu_reload_verified=True, cpu_validation_rmse_rad=rmse,
                  closed_loop_validated=False, independent_success_claim=False,
                  elapsed_s=time.monotonic()-started)
    (a.output/'result.json').write_text(json.dumps(result, indent=2))
    print('V12_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
