"""Run the round-15 physical grasp diagnostics after the existing training and paired control."""
import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from contact_hold_grace_job import (ROOT,RUNS,CAMPAIGN,TERMINAL,SEEDS,FIT,PANEL,available,
                                    read,store,state,digest)
OLD = RUNS/'contact_hold_grace_r15_lift20_20261111_1932'
MODES=('matched_lift','from_reset')
FILES=[ROOT/name for name in (
    'contact_hold_functional_job.py','contact_hold_functional_report.py','contact_hold_functional_worker.py',
    'contact_hold_physical_metrics.py','contact_hold_grace_worker.py','contact_hold_grace_compare.py',
    'contact_hold_grace_job.py','run_contact_hold_functional_job.sh')]+[FIT/'best.pt',PANEL,OLD/'plan.json']

def command(script,*args):
    return [sys.executable,'-B','-u',str(ROOT/script),*map(str,args)]

def report(job,baseline=False):
    out=job/('baseline.json' if baseline else 'report.json')
    return command('contact_hold_functional_report.py','--job',job,'--output',out,*(['--baseline-only'] if baseline else []))

def worker(job,mode,seed):
    return command('contact_hold_functional_worker.py','--mode',mode,'--seed',seed,'--output',job/mode/f'seed_{seed}')

def init(job):
    job.mkdir(exist_ok=False)
    for mode in MODES:
        (job/mode).mkdir()
        for seed in SEEDS:
            with (job/mode/f'preflight_{seed}.json').open('x') as f:
                subprocess.run(worker(job,mode,seed)+['--preflight'],check=True,stdout=f)
    subprocess.run(report(job,True),check=True)
    store(job/'plan.json',dict(round=15,model_sha256=digest(FIT/'best.pt'),seeds=SEEDS,modes=MODES,
        prior_control=str(OLD),source_sha256={str(f):digest(f) for f in FILES},baseline_sha256=digest(job/'baseline.json'),
        endpoint='airborne, physically supported by the left hand, stable and safe for the original pickup duration; no finger count',
        does_not_count_as_official_wrist_validation=True))
    state(job,'queued_after_campaign',training_stage=read(CAMPAIGN/'state.json')['stage'])

def verify(job):
    p=read(job/'plan.json')
    if p['source_sha256']!={str(f):digest(f) for f in FILES} or p['baseline_sha256']!=digest(job/'baseline.json'):
        raise ValueError('The frozen diagnostic sources or baseline changed.')

def receipt(dest,log):
    return {name:digest(p) for name,p in [('manifest_sha256',dest/'artifact_manifest.json'),
        ('physical_hold_sha256',dest/'physical_hold.json'),('log_sha256',log)]}

def ready():
    # Do not reserve slots at the boundaries between training stages.
    return (read(CAMPAIGN/'state.json')['status'] in TERMINAL and
            read(OLD/'state.json')['status'] in {'completed','failed'})

def drive(job):
    with (job/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(job/'state.json')['status']=='completed': return
        try:
            verify(job)
            for mode in MODES:
                for seed in SEEDS:
                    dest=job/mode/f'seed_{seed}';log=dest.with_suffix('.log');ok=dest.with_suffix('.receipt.json')
                    if ok.exists():
                        if read(ok)!=receipt(dest,log):raise ValueError(f'Changed diagnostic artifact: {dest}')
                        continue
                    if dest.exists() or log.exists():raise ValueError(f'Will not overwrite incomplete attempt: {dest}')
                    state(job,'queued_after_campaign',next_mode=mode,next_seed=seed,training_stage=read(CAMPAIGN/'state.json')['stage'])
                    while True:
                        if ready():
                            with available() as fds:
                                if fds is not None:
                                    verify(job)
                                    if shutil.disk_usage(job).free<2*1024**3:raise RuntimeError('Less than 2 GiB free')
                                    state(job,'evaluating',mode=mode,seed=seed)
                                    with log.open('x') as f:
                                        subprocess.run(worker(job,mode,seed),stdout=f,stderr=subprocess.STDOUT,pass_fds=fds,check=True,cwd=ROOT)
                                    with log.open(errors='replace') as f:
                                        for line in f:
                                            if '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower()):
                                                raise RuntimeError('Native physics error: '+line[:450])
                                    store(ok,receipt(dest,log));break
                        time.sleep(45)
            verify(job)
            subprocess.run(report(job),check=True)
            state(job,'completed',report_sha256=digest(job/'report.json'))
        except BaseException as exc:
            state(job,'failed',reason=str(exc));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','status'))
    ap.add_argument('job_dir',type=Path);args=ap.parse_args();job=args.job_dir.resolve()
    if job.parent!=RUNS or not re.fullmatch(r'contact_hold_functional_[A-Za-z0-9_]+',job.name):
        ap.error('Select a new, separate functional diagnostic directory in runs')
    if args.action=='init':init(job)
    elif args.action=='run':drive(job)
    else:print((job/'state.json').read_text())
