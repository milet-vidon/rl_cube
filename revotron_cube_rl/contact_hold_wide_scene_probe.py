"""Capacity trial: real isolated robot/cube physics and batched frozen student inference, not training."""
import argparse,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16)
if __name__=='__main__':
 a=p.parse_args()
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True,device='cuda:0').app
 from isaac_compat import close_application
 try:
    import torch, numpy as np
    from contact_hold_wide_scene import Scene
    from contact_hold_policy import restored
    from contact_hold_gpu_campaign import BASE,DIGEST
    from contact_hold_loop_data import checkpoint,read,write
    if checkpoint(BASE)!=DIGEST:raise ValueError('Reference policy changed')
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=True);m=restored(ck).cuda()
    e=Scene(Path(ck['metadata']['sources'][0]['path']),a.num_envs)
    e.reset(list(range(8_000_000,8_000_000+a.num_envs)))
    if (e.contacts[:,12]<=0).any():raise ValueError('Missing table contact')
    target=e.home.clone();ids=ck['metadata']['output_indices'];cap=torch.minimum(e.step,e.vel*(e.dt*e.dec))
    start=time.monotonic()
    for i in range(a.steps):
        q=e.robot.get_dof_positions()[:,e.ids];v=e.robot.get_dof_velocities()[:,e.ids];c=e.core()
        phase=torch.zeros((e.n,13),device='cuda');phase[:,0]=1.;phase[:,-1]=i/300.
        x=torch.cat([q,v,target,c[:,:3],c[:,[6,3,4,5]],torch.log1p(torch.as_tensor(e.feature_contacts,dtype=q.dtype,device='cuda')/.001),phase],1)
        with torch.inference_mode():target[:,ids]+=(m(x)-target[:,ids]).clamp(-cap[:,ids],cap[:,ids])
        target.clamp_(e.limits[...,0],e.limits[...,1]);e.step_physics(target)
    torch.cuda.synchronize()
    q=e.robot.get_dof_positions()[:,e.ids]
    if not q.isfinite().all() or ((q<e.hard[...,0]-1e-4)|(q>e.hard[...,1]+1e-4)).any() or (e.contacts[:,12]<=0).any():raise ValueError('Invalid post-probe state')
    write(a.output,dict(num_envs=e.n,steps=a.steps,env_steps_per_s=e.n*a.steps/(time.monotonic()-start),cross=e.cross,solver=e.readback,
        table_impulse_range=[float(e.contacts[:,12].min()),float(e.contacts[:,12].max())],kind='short_physics_capacity_not_task_evaluation'))
 except BaseException:traceback.print_exc();raise
 finally:close_application(app)
