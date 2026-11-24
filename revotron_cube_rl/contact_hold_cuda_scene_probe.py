"""Capacity trial: real isolated robot/cube physics and batched frozen student inference, not training."""
import argparse,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16)
p.add_argument('--compare-contacts',action='store_true')
p.add_argument('--dense-fixture',type=Path,help='Restore an observed squeeze state into every lane to stress contacts; no training samples.')
p.add_argument('--fixture-step',type=int,default=502)
if __name__=='__main__':
 a=p.parse_args()
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True,device='cuda:0').app
 from isaac_compat import close_application
 try:
    import torch, numpy as np
    from contact_hold_cuda_scene import Scene
    from contact_hold_policy import restored
    from contact_hold_gpu_campaign import BASE,DIGEST
    from contact_hold_loop_data import checkpoint,read,write
    if checkpoint(BASE)!=DIGEST:raise ValueError('Reference policy changed')
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=True);m=restored(ck).cuda()
    e=Scene(Path(ck['metadata']['sources'][0]['path']),a.num_envs)
    e.reset(list(range(8_000_000,8_000_000+a.num_envs)))
    target=e.home.clone();comparison={}
    if a.dense_fixture:
        import json
        from contact_hold_dataset import sha
        with a.dense_fixture.open() as f:
            row=next(r for r in map(json.loads,f) if r['step']==a.fixture_step)
        if row['phase']!='squeeze' or set(row['cubelets'])!=set(e.views):raise ValueError('Invalid squeeze fixture')
        def repeated(x):return torch.tensor(x,device='cuda',dtype=torch.float32).expand(e.n,-1).contiguous()
        target=repeated(row['applied_target'])
        e.robot.set_dof_positions(repeated(row['q'])[:,e.unorder].contiguous(),e.indices)
        e.robot.set_dof_velocities(repeated(row['joint_velocity'])[:,e.unorder].contiguous(),e.indices)
        e.robot.set_dof_position_targets(target[:,e.unorder].contiguous(),e.indices)
        for k,obj in e.views.items():
            body=row['cubelets'][k];quat=body['quaternion_wxyz'];v=repeated(body['position_m']+quat[1:]+quat[:1]);v[:,:3]+=e.offset
            obj.set_transforms(v,e.indices);obj.set_velocities(repeated(body['linear_velocity_m_s']+body['angular_velocity_rad_s']),e.indices)
        e.step_physics(target)
        touched=(e.hand_impulse>1.e-7).any(1).sum().item()
        if touched!=e.n:raise ValueError(f'Only {touched}/{e.n} lanes made hand contacts in squeeze fixture')
        comparison=dict(fixture=str(a.dense_fixture),fixture_sha256=sha(a.dense_fixture),fixture_step=a.fixture_step,dense_hand_contact_lanes=touched)
    if a.compare_contacts:
        from types import SimpleNamespace
        from contact_hold_wide_scene import Scene as Reference
        ref=SimpleNamespace(n=e.n,contact_views=e.contact_views,sv_warp_contact=e.sv_warp_contact,
                            lookup={},_lookup_ids=np.empty(0,np.uint64),cross=0,
                            contacts=np.zeros((e.n,13)),feature_contacts=np.zeros((e.n,10)),hand_impulse=np.zeros((e.n,2)))
        # Prime both lookup caches; time resident actor IDs on precisely the same PhysX frame.
        def measure(func,clear):
            clear();func()
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(12):clear();func()
            torch.cuda.synchronize();return (time.monotonic()-start)/12
        cpu_s=measure(lambda:Reference.read_contacts(ref),lambda:[v.fill(0) for v in (ref.contacts,ref.feature_contacts,ref.hand_impulse)])
        gpu_s=measure(e.read_contacts,lambda:e.reducer.clear(e))
        for k in ('contacts','feature_contacts','hand_impulse'):
            np.testing.assert_allclose(getattr(e,k).cpu().numpy(),getattr(ref,k),atol=1e-12,rtol=1e-12)
        comparison.update(legacy_contact_s=cpu_s,gpu_contact_s=gpu_s,contact_speedup=cpu_s/gpu_s)
        print('CONTACT_PARITY',comparison,flush=True)
    if not a.dense_fixture and (e.contacts[:,12]<=0).any():raise ValueError('Missing table contact')
    ids=ck['metadata']['output_indices'];cap=torch.minimum(e.step,e.vel*(e.dt*e.dec))
    start=time.monotonic()
    for i in range(a.steps):
        if a.dense_fixture:
            e.step_physics(target);continue
        q=e.robot.get_dof_positions()[:,e.ids];v=e.robot.get_dof_velocities()[:,e.ids];c=e.core()
        phase=torch.zeros((e.n,13),device='cuda');phase[:,0]=1.;phase[:,-1]=i/300.
        x=torch.cat([q,v,target,c[:,:3],c[:,[6,3,4,5]],torch.log1p(torch.as_tensor(e.feature_contacts,dtype=q.dtype,device='cuda')/.001),phase],1)
        with torch.inference_mode():target[:,ids]+=(m(x)-target[:,ids]).clamp(-cap[:,ids],cap[:,ids])
        target.clamp_(e.limits[...,0],e.limits[...,1]);e.step_physics(target)
    torch.cuda.synchronize()
    q=e.robot.get_dof_positions()[:,e.ids]
    if not q.isfinite().all() or ((q<e.hard[...,0]-1e-4)|(q>e.hard[...,1]+1e-4)).any() or (not a.dense_fixture and (e.contacts[:,12]<=0).any()):raise ValueError('Invalid post-probe state')
    write(a.output,dict(num_envs=e.n,steps=a.steps,env_steps_per_s=e.n*a.steps/(time.monotonic()-start),cross=e.cross,solver=e.readback,**comparison,
        table_impulse_range=[float(e.contacts[:,12].min()),float(e.contacts[:,12].max())],kind='short_physics_capacity_not_task_evaluation'))
 except BaseException:traceback.print_exc();raise
 finally:close_application(app)
