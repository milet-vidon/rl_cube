"""Compare CUDA reset trajectories with the unmodified SciPy teacher, plus a timed wide batch."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from contact_hold_cuda_ik import plan_reset
from contact_hold_vram_probe import slot
from opposition_teacher import OppositionPlanner

def check(n,seed,compare):
    ref=OppositionPlanner(Path(__file__).with_name('opposition_grasp_v20.json'))
    xyz=[];yaws=[]
    for k in range(seed,seed+n):
        rng=np.random.default_rng(k);pos=np.array(ref.cfg['cube_xyz'])
        pos[:2]+=rng.uniform(-.015,.015,2);pos[2]=ref.cfg['table_z']+.0285
        xyz.append(pos);yaws.append(rng.uniform(-.15,.15))
    pose=np.c_[xyz,np.zeros((n,2)),np.sin(np.array(yaws)/2),np.cos(np.array(yaws)/2)].astype(np.float32)
    core=torch.tensor(pose,device='cuda')
    torch.cuda.synchronize();start=time.perf_counter()
    plan=plan_reset(core);torch.cuda.synchronize();cold=time.perf_counter()-start
    del plan
    start=time.perf_counter();plan=plan_reset(core);torch.cuda.synchronize()
    warm=time.perf_counter()-start
    result=dict(lanes=n,seed=seed,cold_s=cold,warm_s=warm,accepted=int(plan.valid.sum()),
                peak_torch_mib=round(torch.cuda.max_memory_allocated()/2**20,2),
                device=torch.cuda.get_device_name(),inputs='nominal seated reset poses, before physics',cpu_comparisons=[])
    print('BATCH',json.dumps(result),flush=True)
    # Compare actual GPU lookup (including timing and interpolation), not just IK knots.
    for lane in np.linspace(0,n-1,min(compare,n)).round().astype(int):
        yaw=2*np.arctan2(float(pose[lane,5]),float(pose[lane,6]));t=time.perf_counter()
        try:gold=ref.plan(pose[lane,:3].astype(float),yaw)
        except ValueError:
            assert not plan.valid[lane].item();continue
        assert plan.valid[lane].item(),(seed+lane,plan.rejection_code[lane])
        err=0.
        for phase,(name,arr) in enumerate(gold):
            assert len(arr)==plan.lengths[lane,phase].item(),(seed+lane,name,len(arr),plan.lengths[lane,phase])
            for ix in (0,len(arr)//3,len(arr)//2,len(arr)-1):
                ph=torch.full_like(plan.rows,phase);ticks=torch.full_like(ph,ix)
                value=plan.at(ph,ticks)[lane].cpu().numpy()
                err=max(err,float(np.max(np.abs(value-arr[ix]))))
        if err>2.e-4:raise AssertionError((lane,err))
        result['cpu_comparisons'].append(dict(seed=int(seed+lane),max_abs_rad=err,elapsed_s=time.perf_counter()-t))
    # Invalid pose cannot be accepted or accidentally reused.
    invalid=core[:1].clone();invalid[:,0]=5.
    assert not plan_reset(invalid).valid.item()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--envs',type=int,default=1536);p.add_argument('--seed',type=int,default=1004096)
    p.add_argument('--compare',type=int,default=5);p.add_argument('--output',type=Path)
    a=p.parse_args()
    with slot():result=check(a.envs,a.seed,a.compare)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
