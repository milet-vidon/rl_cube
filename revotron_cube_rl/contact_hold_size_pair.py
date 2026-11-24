"""Run two fresh 90 mm trials with the existing device leases, then compare them to 57 mm."""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from contact_hold_grace_job import ROOT, RUNS, SEEDS, available, state, read, store, digest
from contact_hold_size_geometry import SOURCE, prepare_asset, verify_asset

def run(job, relax_approach=False):
    if job.parent!=RUNS or not re.fullmatch(r'contact_hold_size_r15_[0-9A-Za-z_]+',job.name):
        raise ValueError('Expected a new runs/contact_hold_size_r15_<id>')
    job.mkdir(exist_ok=True)
    a=job/'asset90/cube_model.usda'; g=a.with_name('geometry.json')
    if not g.exists():store(g,prepare_asset(SOURCE,a,.090))
    geom=verify_asset(a,read(g))
    if geom['edge_m']!=.090:raise ValueError('Wrong edge')
    (job/'from_reset').mkdir(exist_ok=True)
    if (job/'state.json').exists() or any((job/'from_reset').iterdir()):raise ValueError('Refusing to overwrite an attempt')
    runpy=[sys.executable,'-B','-u']
    with available() as fds:
        if fds is None:raise RuntimeError('Training or graphics device busy; did not start')
        if shutil.disk_usage(job).free<2*1024**3:raise RuntimeError('Need 2 GiB free')
        try:
            for seed in SEEDS:
                out=job/'from_reset'/f'seed_{seed}'
                worker='contact_hold_size_approach_worker.py' if relax_approach else 'contact_hold_size_worker.py'
                cmd=[*runpy,str(ROOT/worker),'--seed',str(seed),'--output',str(out),'--asset',str(a)]
                state(job,'simulating',seed=seed)
                log=out.with_suffix('.log')
                with log.open('x') as f:subprocess.run(cmd,check=True,pass_fds=fds,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                for line in log.read_text(errors='replace').splitlines():
                    if '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower()):
                        raise ValueError('Physics runtime error: '+line[:500])
                print('SIMULATED',seed,flush=True)
            state(job,'scoring')
            subprocess.run([*runpy,str(ROOT/'contact_hold_size_report.py'),str(job)],check=True)
            state(job,'completed',report_sha256=digest(job/'size_comparison.json'))
        except BaseException as exc:
            state(job,'failed',reason=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path,nargs='?');p.add_argument('--relax-approach',action='store_true',help='Separate 90mm probe with the looser pre-contact alignment gate');args=p.parse_args()
    tag='approach_' if args.relax_approach else ''
    out=args.job or RUNS/('contact_hold_size_r15_90mm_'+tag+datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(out.resolve(),flush=True);run(out.resolve(),args.relax_approach)
