"""Two real updates on a copy of a prepared plan; never advance the campaign."""
import argparse
import sys
from pathlib import Path
from contact_hold_cuda_train import ROOT,checkpoint,read,write,identity
from contact_hold_cuda_monitor import monitor
from contact_hold_vram_probe import slot

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); d=a.output.resolve();d.mkdir(parents=True,exist_ok=False)
    fp=read(a.plan);parent=Path(fp['parent'])
    if fp['code']!=identity() or checkpoint(parent)!=fp['parent_sha']:raise ValueError('Prepared plan changed')
    write(d/'plan.json',dict(fp,updates=2))
    with slot() as fds:
        t=monitor([sys.executable,'-u',str(ROOT/'contact_hold_cuda_train.py'),'--fit',str(d/'plan.json')],d/'fit.log',fds)
        write(d/'fit.resources.json',t)
    if not t['ok']:raise RuntimeError(t['reason'])
    checkpoint(d);r=read(d/'result.json')
    if r['global_update']!=read(parent/'result.json')['global_update']+2:raise ValueError('Fit did not continue the parent')
    print('ISOLATED_FIT_CHECK',r['global_update'],'batch',fp['batch'],'peak device MiB',t['peak_device_mib'],flush=True)
if __name__=='__main__':main()
