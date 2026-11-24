"""Measure live physics VRAM, host RAM and throughput in isolated ascending trials."""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
from contact_hold_vram_probe import slot,gpu
from contact_hold_loop_data import ROOT,read,write
from contact_hold_dataset import sha

def available_mib():
    line=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    return int(line.split()[1])//1024

def monitored(cmd,log,fds,*,gpu_floor=512,ram_floor=4096,deadline_s=None):
    peak=0;rss=0;low=available_mib();reason=None;start=time.monotonic();seen=False
    with log.open('x') as f, subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,pass_fds=fds,start_new_session=True) as proc:
        try:
            while proc.poll() is None:
                name,total,used,util=gpu();avail=available_mib();peak=max(used,peak);low=min(low,avail);seen=True
                try:
                    r=int(next(l.split()[1] for l in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))//1024
                    rss=max(rss,r)
                except (FileNotFoundError,StopIteration):pass
                if total-used<gpu_floor or avail<ram_floor:reason='device_or_host_memory_headroom'
                if deadline_s and time.monotonic()-start>deadline_s:reason='trial_time_budget'
                if reason:break
                time.sleep(.4)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=4)
                except subprocess.TimeoutExpired:pass
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    if not seen:raise RuntimeError('No GPU telemetry was recorded')
    if not reason and proc.returncode:reason=f'worker_exit_{proc.returncode}'
    if not reason:
        for line in log.read_text(errors='replace').splitlines():
            if '[Error]' in line and ('physx' in line.lower() or 'physics.tensors' in line.lower()):
                reason='PhysX error: '+line[:240];break
    return dict(ok=reason is None,reason=reason,peak_device_mib=peak,min_host_available_mib=low,peak_worker_rss_mib=rss,
                gpu=name,total_mib=total,elapsed_s=time.monotonic()-start)

def measure(out,ns,steps,hold_for_fix=False):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    with slot() as fds:
        for n in ns:
            # During commissioning, a rejected backend can be fixed in a bounded GPU test window.
            for attempt in range(3 if hold_for_fix else 1):
                code={k:sha(ROOT/k) for k in ('contact_hold_wide_scene.py','contact_hold_wide_scene_probe.py')}
                stem=f'n{n}_{attempt}';report=out/(stem+'.json');log=out/(stem+'.log')
                print('TRIAL',n,attempt,flush=True)
                r=monitored([sys.executable,'-u',str(ROOT/'contact_hold_wide_scene_probe.py'),'--num-envs',str(n),'--steps',str(steps),'--output',str(report)],log,fds,deadline_s=180)
                r.update(n=n,log=str(log),code=code)
                if r['ok'] and report.is_file():r.update(physics=read(report),physics_sha=sha(report))
                else:r['ok']=False
                print('RESULT',n,r['ok'],r['reason'],r['peak_device_mib'],r['min_host_available_mib'],flush=True)
                rows.append(r);write(out/'trials.json',rows)
                if r['ok'] or not hold_for_fix or attempt==2 or r['reason'] in ('device_or_host_memory_headroom','trial_time_budget'):break
                until=time.monotonic()+90
                while time.monotonic()<until and all(sha(ROOT/k)==h for k,h in code.items()):time.sleep(1)
                if all(sha(ROOT/k)==h for k,h in code.items()):break
            if not r['ok'] or r['peak_device_mib']>=r['total_mib']-900:break
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--envs',type=int,nargs='+',default=[32,128,256,512,768,1024,1280,1536]);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--commission',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.envs!=sorted(set(a.envs)) or min(a.envs)<2 or max(a.envs)>2048 or a.steps<4:p.error('invalid trials')
    rr=measure(a.output,a.envs,a.steps,a.commission)
    if not any(r['ok'] for r in rr):sys.exit(2)
