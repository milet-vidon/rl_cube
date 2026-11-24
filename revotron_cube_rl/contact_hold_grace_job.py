"""One diagnostic pair; never reserve a GPU or an Isaac slot during active long training."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs'
CAMPAIGN = RUNS/'contact_hold_cuda_long_v1'
FIT = CAMPAIGN/'round_015/fit'
PANEL = FIT.parent/'validation/panel.json'
LOCKS = ROOT.parent/'work/teacher_optimization_20261103'
SEEDS = (21611, 21612)
TERMINAL = {'plateau','max_rounds','confirmed_on_test','heldout_target_not_met'}
FILES = [ROOT/f for f in ('contact_hold_grace_job.py','contact_hold_grace_worker.py','contact_hold_grace_compare.py','run_contact_hold_grace_job.sh')] + [FIT/'best.pt',PANEL]
def read(p):
    return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def store(p,j):
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(p)
def state(job,status,**fields):
    store(job/'state.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**fields))
def verify_plan(job):
    p=read(job/'plan.json')
    if p['source_sha256']!={str(f):digest(f) for f in FILES} or p['baseline_sha256']!=digest(job/'baseline.json'):
        raise RuntimeError('Pinned source, checkpoint, panel, or baseline analysis changed; no run admitted.')
def init(job):
    job.mkdir(exist_ok=False)
    for seed in SEEDS:
        with (job/f'preflight_{seed}.json').open('x') as f:
            subprocess.run([sys.executable,'-B',str(ROOT/'contact_hold_grace_worker.py'),'--seed',str(seed),'--output',str(job/f'seed_{seed}'),'--preflight'],check=True,stdout=f)
    subprocess.run([sys.executable,'-B',str(ROOT/'contact_hold_grace_compare.py'),'--output',str(job/'baseline.json')],check=True)
    store(job/'plan.json',dict(campaign=str(CAMPAIGN),fit=str(FIT),seeds=SEEDS,baseline_frames=10,treatment_frames=20,
          launch_only_after_terminal=True,baseline_sha256=digest(job/'baseline.json'),source_sha256={str(f):digest(f) for f in FILES}))
    state(job,'queued_after_campaign',campaign_stage=read(CAMPAIGN/'state.json')['stage'])

@contextmanager
def available():
    # In particular DO NOT acquire (even temporarily) either global lock while the current
    # campaign is running. GPU inactivity, fit/validation boundaries or an interrupted
    # training session are not a green light.
    if read(CAMPAIGN/'state.json')['status'] not in TERMINAL:
        yield None;return
    with ExitStack() as s:
        fds=[]
        try:
            for path,mode in [(CAMPAIGN/'campaign.lock',fcntl.LOCK_SH),
                              (LOCKS/'contact_hold_gpu_training.lock',fcntl.LOCK_EX),(LOCKS/'isaac_gui_evaluation.lock',fcntl.LOCK_EX)]:
                f=s.enter_context(path.open('rb'))
                fcntl.flock(f,mode|fcntl.LOCK_NB);fds.append(f.fileno())
        except BlockingIOError:
            s.close();yield None;return
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if jobs or read(CAMPAIGN/'state.json')['status'] not in TERMINAL:
            s.close();yield None;return
        yield tuple(fds)

def drive(job):
    with (job/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(job/'state.json')['status']=='completed':return
        try:
            verify_plan(job)
            for seed in SEEDS:
                dest=job/f'seed_{seed}';log=job/f'seed_{seed}.log';receipt=job/f'passed_worker_{seed}.json'
                if receipt.exists():
                    last=read(receipt)
                    if last!={'manifest_sha256':digest(dest/'artifact_manifest.json'),'log_sha256':digest(log)}:
                        raise RuntimeError(f'Changed completed diagnostic for {seed}')
                    continue
                if log.exists() or dest.exists():raise RuntimeError(f'Incomplete prior diagnostic for {seed}; refusing to overwrite evidence')
                state(job,'queued_after_campaign',next_seed=seed,campaign_stage=read(CAMPAIGN/'state.json')['stage'])
                while True:
                    with available() as fds:
                        if fds is not None:
                            verify_plan(job)
                            if shutil.disk_usage(job).free<2*1024**3:raise RuntimeError('Need 2 GiB of free disk space to evaluate')
                            state(job,'evaluating',seed=seed,campaign_status=read(CAMPAIGN/'state.json')['status'])
                            with log.open('x') as out:
                                subprocess.run([sys.executable,'-B','-u',str(ROOT/'contact_hold_grace_worker.py'),'--seed',str(seed),'--output',str(dest)],
                                               cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,pass_fds=fds)
                            with log.open(errors='replace') as f:
                                for line in f:
                                    if '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower()):
                                        raise RuntimeError('Native physics error: '+line.strip()[:400])
                            store(receipt,{'manifest_sha256':digest(dest/'artifact_manifest.json'),'log_sha256':digest(log)})
                            break
                    time.sleep(45)
            verify_plan(job)
            subprocess.run([sys.executable,'-B',str(ROOT/'contact_hold_grace_compare.py'),'--treatment',
                           *(str(job/f'seed_{n}') for n in SEEDS),'--output',str(job/'report.json')],check=True)
            state(job,'completed',report_sha256=digest(job/'report.json'))
        except BaseException as e:
            state(job,'failed',reason=str(e),traceback=traceback.format_exc()[-4000:]);raise
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['init','run','status'])
    ap.add_argument('job_dir',type=Path);a=ap.parse_args();job=a.job_dir.resolve()
    if job.parent!=RUNS or not re.fullmatch(r'contact_hold_grace_[A-Za-z0-9_]+',job.name):ap.error('Expected an isolated new diagnostic namespace in runs')
    if a.action=='init':init(job)
    elif a.action=='run':drive(job)
    else:
        print((job/'state.json').read_text());c=read(CAMPAIGN/'state.json');print('long training:',c['status'],'/',c['stage'])
