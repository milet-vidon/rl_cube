"""Run a measured real sampler probe using the current parent, seed range and code hashes."""
import argparse
import sys
from pathlib import Path
from contact_hold_cuda_campaign import ROOT,checkpoint,read,write,sampler_code,show_sampling_progress
from contact_hold_cuda_monitor import monitor
from contact_hold_vram_probe import slot

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--envs',type=int,default=1536)
    ap.add_argument('--first-seed',type=int,default=7_000_000);ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--source',type=Path,default=ROOT/'runs/contact_hold_wide_v1')
    a=ap.parse_args();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False);dest=d/'samples'
    parent=Path(read(a.source/'state.json')['current']);cp=d/'collect.json';log=dest.with_suffix('.log')
    with slot() as fds:
        write(cp,dict(parent=str(parent),parent_sha=checkpoint(parent),code=sampler_code(),lane_offset=0,seeds=list(range(a.first_seed,a.first_seed+a.envs))))
        with show_sampling_progress(log):
            t=monitor([sys.executable,'-u',str(ROOT/'contact_hold_cuda_sample.py'),'--plan',str(cp),'--output',str(dest),'--steps',str(a.steps)],log,fds)
        write(dest.with_suffix('.resources.json'),t)
    print(t,flush=True)
    if not t['ok']:raise SystemExit(2)
if __name__=='__main__':main()
