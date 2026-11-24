"""Time only the 1,536-lane finger feedback kernel on measured squeeze/lift states.

Run CPU first, then GPU with the same directory. Excludes scene construction,
simulation, reset IK, staging and serialization. The legacy online loop dispatched
FingerArrival serially; its CPU baseline has one physical execution thread.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
# Bound host math to one thread before loading BLAS.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
from kinematics import RobotKinematics
from opposition_feedback import FingerArrival
from contact_hold_cuda_teacher_math import FingerFeedback
from phase_contact_policy import FINGERS

ROOT=Path(__file__).resolve().parent
START=(508,515,522,526,531,535,537,542,549,563,605,642)
NUM=1536
STEPS=3

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def setup(run):
    with (run/'trajectory.jsonl').open() as f:rows={r['step']:r for r in map(json.loads,f) if r['step'] in {s+d for s in START for d in range(STEPS+1)}}
    cfg=json.loads((run/'config.json').read_text());grasp=json.loads((ROOT/'opposition_grasp_v20.json').read_text())
    frames=[]
    for i in range(STEPS):
        seen=[rows[s+i] for s in START]
        next_=[rows[s+i+1] for s in START]
        assert all(r['phase'] in ('squeeze','primary_lift') for r in seen+next_)
        def pack(v):return np.tile(np.asarray(v,dtype=np.float32),(NUM//len(START),1))
        q=pack([r['q'] for r in seen]);goal=pack([r['nominal_target'] for r in next_])
        xyz=pack([r['cube_position'] for r in seen]);quat=pack([r['cube_quaternion_wxyz'] for r in seen])
        contact=pack([[r['finger_contact_impulses']['left'].get(k,0.) for k in FINGERS] for r in seen])
        frames.append((q,goal,xyz,quat,contact))
    return cfg,grasp,frames

def cpu(cfg,grasp,frames):
    k=RobotKinematics()
    make=lambda:FingerArrival(k,grasp,cfg['joint_names'],cfg['robot_z'])
    cached=[(q,goal,xyz,quat,[dict(zip(FINGERS,row)) for row in cont]) for q,goal,xyz,quat,cont in frames]
    q,g,p,r,co=cached[0]
    make().update(q[0],g[0],co[0],p[0],r[0])
    workers=[make() for _ in range(NUM)];output=np.empty((STEPS,NUM,56),dtype=np.float32)
    start=time.monotonic()
    for step,(q,goal,xyz,quat,cont) in enumerate(cached):
        for j,worker in enumerate(workers):output[step,j]=worker.update(q[j],goal[j],cont[j],xyz[j],quat[j])
    return time.monotonic()-start,output

def gpu(cfg,grasp,frames):
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    k=RobotKinematics();mask=torch.ones(NUM,device=dev,dtype=torch.bool)
    cache=[]
    for q,goal,xyz,quat,ct in frames:
        core=np.concatenate([xyz,quat[:,[1,2,3,0]]],axis=1)
        contacts=np.pad(ct,((0,0),(0,8)))
        cache.append(tuple(torch.from_numpy(a).to(dev) for a in (q,goal,contacts,core)))
    times=[];res=None
    for i in range(6):
        teacher=FingerFeedback(k,grasp,cfg['joint_names'],cfg['robot_z'],NUM,dev)
        torch.cuda.synchronize();t=time.monotonic()
        with torch.no_grad():out=[teacher.update(mask,*args) for args in cache]
        torch.cuda.synchronize();times.append(time.monotonic()-t)
        res=torch.stack(out).cpu().numpy()
    return times,res,torch.cuda.get_device_name(dev)

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('backend',choices=('cpu','gpu'));p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,g,fr=setup(a.run);dst=a.output;source_hashes={p:sha(ROOT/p) for p in
        ('contact_hold_cuda_teacher_benchmark.py','contact_hold_cuda_teacher_math.py','opposition_feedback.py','kinematics.py','opposition_grasp_v20.json')}
    meta=dict(kind='fixed_trace_finger_arrival_microbenchmark_v1',num_envs=NUM,steps=STEPS,unique_measured_sequences=len(START),
              trace=str((a.run/'trajectory.jsonl').resolve()),trajectory_sha256=sha(a.run/'trajectory.jsonl'),sequence_starts=START,
              missing_finger_fits_on_first_step=int((fr[0][-1]<=1e-5).sum()),lanes_with_missing_fingers_on_first_step=int((fr[0][-1]<=1e-5).any(1).sum()),
              initial_feedback='fresh per-lane state, tick 0; solve then two slew/hold updates on next measured frames',scope='finger feedback only, states already resident; no PhysX or reset IK timing',
              source_sha256=source_hashes)
    if a.backend=='cpu':
        dst.mkdir(exist_ok=False,parents=True)
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1)
        elapsed,ref=cpu(cfg,g,fr)
        np.save(dst/'reference.npy',ref)
        save(dst/'cpu.json',dict(**meta,scalar_cpu_elapsed_s=elapsed,reference_sha256=sha(dst/'reference.npy')))
        print('CPU_FEEDBACK',elapsed,flush=True)
    else:
        data=json.loads((dst/'cpu.json').read_text())
        if (data['source_sha256']!=source_hashes or data['trajectory_sha256']!=meta['trajectory_sha256']
                or data['reference_sha256']!=sha(dst/'reference.npy')):raise ValueError('Input or implementation changed between timing passes')
        ref=np.load(dst/'reference.npy')
        from contact_hold_vram_probe import slot
        with slot():times,test,card=gpu(cfg,g,fr)
        error=np.abs(ref-test)
        if not np.isfinite(error).all() or error.max()>.0005:raise ValueError(f'Target mismatch: {error.max()} rad')
        elapsed=float(np.median(times[1:]))
        result=dict(**data,gpu=card,cold_cuda_elapsed_s=times[0],cuda_repeats_s=times[1:],cuda_median_elapsed_s=elapsed,
                    scalar_over_batched_speedup=data['scalar_cpu_elapsed_s']/elapsed,max_target_error_rad=float(error.max()),max_error_by_step=error.max(axis=(1,2)).tolist())
        save(dst/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
