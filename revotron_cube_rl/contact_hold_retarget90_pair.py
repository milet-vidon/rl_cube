"""Run one 90 mm calibrated grasp, unchanged across both frozen round-15 seed trials."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from contact_hold_grace_job import ROOT, RUNS, SEEDS, available, state, read, store, digest
from contact_hold_grace_worker import CHECKPOINT, FIT
from contact_hold_size_geometry import SOURCE, prepare_asset, verify_asset


def run(job, canonical=True):
    if job.parent != RUNS or not re.fullmatch(r'contact_hold_size_r15_[A-Za-z0-9_]*retarget[A-Za-z0-9_]*approach[A-Za-z0-9_]*',job.name):
        raise ValueError('Choose a new runs/contact_hold_size_r15_90mm_retarget_approach_<id>')
    job.mkdir(exist_ok=True)
    if (job/'state.json').exists(): raise ValueError('Refusing to overwrite a prior attempt')
    assets=job/'asset90'; usd=assets/'cube_model.usda'; g=assets/'geometry.json'
    if not g.exists(): store(g,prepare_asset(SOURCE,usd,.090))
    if verify_asset(usd,read(g))['edge_m']!=.090: raise ValueError('Not a 90 mm cube')
    ep=job/'from_reset'; ep.mkdir(exist_ok=True)
    if any(ep.iterdir()): raise ValueError('Episode directories must be fresh')
    calibration=job/'calibration'
    plan=calibration/'grasp90.json'
    geometry=calibration/'geometry_retarget.json'
    if not plan.is_file() or not geometry.is_file():
        env=os.environ.copy()
        for key in ('PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME'):env.pop(key,None)
        env.update(PYTHONPATH='/opt/ros/jazzy/lib/python3.12/site-packages',LD_LIBRARY_PATH='/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu',OPENBLAS_NUM_THREADS='1')
        subprocess.run(['/usr/bin/python3','-B',str(ROOT/'contact_hold_size_retarget_geometry.py'),'--output-dir',str(calibration)],env=env,cwd=ROOT,check=True)
    pins={str(p):digest(p) for p in (plan,geometry,usd,FIT/'best.pt')}
    if pins[str(FIT/'best.pt')]!=CHECKPOINT: raise ValueError('Wrong frozen model')
    store(job/'pair_plan.json',dict(seeds=list(SEEDS),kind='90mm_retarget_same_calibration',sha256=pins))
    with available() as fds:
        if fds is None: raise RuntimeError('Existing Isaac or training job has device; no rollout started')
        if shutil.disk_usage(job).free<2*1024**3:raise RuntimeError('Need 2 GiB free')
        try:
            for seed in SEEDS:
                if any(digest(Path(p))!=s for p,s in pins.items()):raise ValueError('A pinned input changed')
                out=ep/f'seed_{seed}';log=out.with_suffix('.log')
                state(job,'simulating',seed=seed)
                worker='contact_hold_size_canonical_worker.py' if canonical else 'contact_hold_size_retarget_worker.py'
                cmd=[sys.executable,'-B','-u',str(ROOT/worker),'--seed',str(seed),'--output',str(out),'--asset',str(usd)]
                with log.open('x') as f:subprocess.run(cmd,check=True,pass_fds=fds,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                for line in log.read_text(errors='replace').splitlines():
                    if '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower()):raise ValueError(line[:500])
                print('SIMULATED',seed,flush=True)
            state(job,'scoring')
            from contact_hold_retarget90_report import write_report
            dst=write_report(job)
            state(job,'completed',report_sha256=digest(dst));print(dst,flush=True)
        except BaseException as exc:state(job,'failed',reason=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path,nargs='?');p.add_argument('--output-only',action='store_true',help='Earlier diagnostic: apply only the output shift');args=p.parse_args()
    job=args.job or RUNS/('contact_hold_size_r15_90mm_retarget_approach_'+datetime.now().strftime('%Y%m%d_%H%M%S')+('' if args.output_only else '_canonical'))
    run(job.resolve(),not args.output_only)
