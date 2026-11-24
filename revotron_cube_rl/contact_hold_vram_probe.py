"""Size a real contact-hold backward pass on an idle CUDA device; never changes its parent run."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager

from contact_hold_gpu_campaign import fit, identity
from contact_hold_continuous import LOCKS
from contact_hold_loop_data import checkpoint, read, write
from contact_hold_dataset import sha


def gpu():
    fields='name,memory.total,memory.used,utilization.gpu'
    line=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    name,total,used,busy=map(str.strip,line.split(','))
    return name,int(total),int(used),int(busy)


@contextmanager
def slot():
    # Block at the next stage boundary. The existing launcher uses these locks in the same order.
    print('Waiting for an exclusive GPU slot; the current training stage can finish.',flush=True)
    while True:
        with ExitStack() as stack:
            fs=[stack.enter_context((LOCKS/k).open('a')) for k in ('contact_hold_gpu_training.lock','isaac_gui_evaluation.lock')]
            for f in fs:fcntl.flock(f,fcntl.LOCK_EX)
            jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not jobs:
                yield tuple(f.fileno() for f in fs)
                return
        time.sleep(2)


def worker(plan,out,headroom):
    import torch
    dev=torch.device('cuda:0')
    free,total=torch.cuda.mem_get_info(dev)
    # Limit the allocator even in deliberately oversize trials. The parent discards failed trials.
    torch.cuda.set_per_process_memory_fraction(max(0.01,(free-(headroom//2)*2**20)/total),dev)
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        fit(plan)
        if not read(plan.parent/'result.json')['checkpoint_reload_verified']:raise RuntimeError('reload failed')
    except torch.OutOfMemoryError:
        write(out,dict(ok=False,reason='cuda_oom'));return
    torch.cuda.synchronize()
    write(out,dict(ok=True,peak_reserved_mib=math.ceil(torch.cuda.max_memory_reserved(dev)/2**20),
                   peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated(dev)/2**20),
                   reserved_end_mib=math.ceil(torch.cuda.memory_reserved(dev)/2**20),
                   driver_end_mib=gpu()[2]))


def probe(source,output,headroom):
    source=source.resolve();output=output.resolve()
    template=read(source)
    if template['code']!=identity() or checkpoint(Path(template['parent']))!=template['parent_sha']:
        raise ValueError('Changed source fit plan or parent')
    if not 512<=headroom<=2048:raise ValueError('headroom must be 512–2048 MiB')
    output.mkdir(parents=True,exist_ok=False)
    with slot() as fds:
        name,total,idle,_=gpu();target=total-headroom;unit=8192
        if target<=idle+512:raise ValueError('Insufficient free VRAM for a useful trial')
        lo=0;hi=None;b=262144;records=[]
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            for i in range(8):
                work=Path(tmp)/str(i);work.mkdir()
                plan=work/'plan.json';result=work/'memory.json';log=output/f'trial_{b}.log'
                write(plan,dict(template,batch=b,updates=3))
                print('VRAM_PROBE batch',b,'target device MiB',target,flush=True)
                peak=idle
                with log.open('x') as f, subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(plan),str(result),
                        '--headroom-mib',str(headroom)],stdout=f,stderr=subprocess.STDOUT,pass_fds=fds) as proc:
                    while proc.poll() is None:
                        peak=max(peak,gpu()[2]);time.sleep(.25)
                    if proc.returncode:raise RuntimeError('Fit probe failed: '+str(log))
                r=read(result)
                if r['ok']:
                    # Allocator peak plus non-allocator driver usage at the end also covers short unsampled peaks.
                    peak=max(peak,r['driver_end_mib'],r['peak_reserved_mib']+r['driver_end_mib']-r['reserved_end_mib'])
                good=r['ok'] and peak<=target
                records.append(dict(batch=b,peak_device_mib=peak,within_budget=good,**r))
                print('MEASURED',records[-1],flush=True)
                if good:
                    lo=max(lo,b)
                    if target-peak<=max(160,int(target*.025)):break
                else:hi=b if hi is None else min(hi,b)
                if hi is not None:
                    nxt=((lo+hi)//(2*unit))*unit
                    if nxt<=lo or nxt>=hi:break
                else:
                    estimate=b*(target-idle)/max(1,peak-idle)
                    nxt=max(b+unit,min(b*3,int(estimate*.99/unit)*unit))
                b=nxt
        if not lo:raise RuntimeError('No measured batch with the requested free VRAM')
        chosen=next(r for r in records if r['batch']==lo)
        report=dict(kind='contact_hold_real_fit_memory_calibration',gpu=name,total_mib=total,idle_mib=idle,
                    headroom_mib=headroom,target_device_mib=target,source_plan=str(source),source_plan_sha256=sha(source),
                    parent_sha256=template['parent_sha'],code=identity(),probe_sha256=sha(Path(__file__)),
                    steps_per_trial=3,selected=chosen,trials=records)
        write(output/'calibration.json',report);print('CHOSEN',chosen['batch'],'peak MiB',chosen['peak_device_mib'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--headroom-mib',type=int,default=768);p.add_argument('--worker',nargs=2,type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:worker(*a.worker,a.headroom_mib)
    elif a.plan and a.output:probe(a.plan,a.output,a.headroom_mib)
    else:p.error('--plan and --output required')
