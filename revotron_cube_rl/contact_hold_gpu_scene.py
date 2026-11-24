"""Isolated reference clones of the audited dynamic-cube scene, one PhysX CUDA scene."""
from pathlib import Path
import re
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
from isaac_compat import physx_config, simulation_config, prepare_single_world
import isaaclab.sim as sim_utils
import omni.physx
import omni.physics.tensors as tensors
from contact_hold_dataset import sha
from contact_hold_loop_data import read
from joint_safety import soft_limits_from_hard,max_target_delta

FINGERS=('thumb','index','middle','ring','little')
LANE=re.compile(r'^/World/envs/env_(\d+)(/.*)$')


class Scene:
    def __init__(self,source,n):
        self.n=n;self.cfg=c=read(source/'config.json'); self.dt=1/c['physics_hz'];self.dec=c['physics_hz']//c['control_hz']
        path=source/'dynamic_training_scene.usda';manifest=read(source/'artifact_manifest.json')['sha256']
        for name in ('dynamic_training_scene.usda','runtime_robot_asset.json','cube_asset.json','asset_contract.json','config.json'):
            if sha(source/name)!=manifest[name]:raise ValueError('Changed source: '+name)
        for name,h in read(source/'runtime_robot_asset.json')['usd_layers_sha256'].items():
            if sha(Path(name))!=h:raise ValueError('Robot asset changed')
        src_stage=Usd.Stage.Open(str(path.resolve()))
        cube_source=src_stage.GetRootLayer().subLayerPaths[0]
        if sha(Path(cube_source))!=read(source/'cube_asset.json')['source_sha256']:raise ValueError('Cube source changed')
        phys=physx_config(solver_type=1,min_position_iteration_count=64,max_position_iteration_count=64,min_velocity_iteration_count=4,
            gpu_found_lost_aggregate_pairs_capacity=2**17,gpu_max_rigid_contact_count=2**19,gpu_max_rigid_patch_count=2**16)
        self.sim=sim_utils.SimulationContext(simulation_config(dt=self.dt,device='cuda:0',use_fabric=False,render_interval=self.dec,physx=phys))
        stage=self.sim.stage;self.stage=stage
        self.sim.set_setting('/physics/disableContactProcessing',False)
        self.offset=torch.tensor([[6.*(i%4),6.*(i//4),0.] for i in range(n)],device='cuda')
        UsdGeom.Xform.Define(stage,'/World/envs')
        for k,pos in enumerate(self.offset.tolist()):
            root=f'/World/envs/env_{k:03d}'; x=UsdGeom.Xform.Define(stage,root)
            x.GetPrim().GetReferences().AddReference(str(path.resolve()),'/World')
            x.AddTranslateOp().Set(Gf.Vec3d(*pos))
            # Each collision group contains its own complete world.
            group=UsdPhysics.CollisionGroup.Define(stage,f'/World/filters/lane_{k:03d}')
            group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(root)
            group.CreateFilteredGroupsRel().SetTargets([f'/World/filters/lane_{j:03d}' for j in range(n) if j!=k])
            bodies=[p for p in Usd.PrimRange(x.GetPrim()) if p.HasAPI(UsdPhysics.RigidBodyAPI) and ('/Cube/' in str(p.GetPath()) or p.GetName()=='Core')]
            if len(bodies)!=27 or any(UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() for p in bodies):
                raise RuntimeError('Wrong free cube')
            for obj in bodies:
                PhysxSchema.PhysxContactReportAPI.Apply(obj).CreateThresholdAttr(0.)
            for p in Usd.PrimRange(x.GetPrim()):
                if p.IsA(UsdPhysics.Joint):
                    for rel in (UsdPhysics.Joint(p).GetBody0Rel(),UsdPhysics.Joint(p).GetBody1Rel()):
                        if any(not str(t).startswith(root+'/') for t in rel.GetTargets()):
                            raise ValueError('Joint links another lane')
        self.paths={};self.contacts=np.zeros((n,13),np.float64);self.cross=0;self.feature_contacts=np.zeros((n,10))

        prepare_single_world(self.sim);self.sim.reset()
        self.sv=tensors.create_simulation_view('torch')
        self.sv.set_subspace_roots('/')
        self.robot=self.sv.create_articulation_view('/World/envs/env_*/Robot/root_joint')
        if self.robot.count!=n:raise RuntimeError(f'Expected {n} robot articulations, got {self.robot.count}')
        names=self.robot.shared_metatype.dof_names
        self.names=c['joint_names'];self.ids=[names.index(name) for name in self.names]
        self.unorder=np.argsort(self.ids).tolist()
        self.indices=torch.arange(n,device='cuda',dtype=torch.int32)
        self.views={p.GetName():self.sv.create_rigid_body_view(str(p.GetPath()).replace(f'env_{n-1:03d}','env_*')) for p in bodies}
        for view in (self.robot,*self.views.values()):
            if [int(LANE.match(p).group(1)) for p in view.prim_paths]!=list(range(n)): raise ValueError('Wrong lane mapping')
        self.sv_warp=tensors.create_simulation_view('warp')
        self.sv_warp_contact=self.sv_warp.create_rigid_contact_view('/World/envs/env_000/Core',max_contact_data_count=64)
        self.contact_views=[]
        for j in range(n):
            root=f'/World/envs/env_{j:03d}'
            sensors=[str(v.GetPath()) for v in Usd.PrimRange(stage.GetPrimAtPath(root+'/Robot')) if v.HasAPI(UsdPhysics.RigidBodyAPI)]
            for kind,paths in [('robot',sensors),('cube',[v.prim_paths[j] for k,v in self.views.items() if k!='Core'])]:
                view=self.sv.create_rigid_contact_view(paths,[[] for _ in paths],max_contact_data_count=2048)
                if view.sensor_count!=len(paths):raise ValueError(f'Unexpected {kind} contact sensor coverage: {view.sensor_count}/{len(paths)}')
                self.contact_views.append((j,kind,view))
        self.hard=self.robot.get_dof_limits()[:,self.ids].cuda()
        torch.testing.assert_close(self.hard,self.hard[:1].expand_as(self.hard))
        self.hand_impulse=np.zeros((n,2))
        self.limits=soft_limits_from_hard(self.hard[:1],self.names,c)
        self.scales=torch.tensor([[.025 if k.startswith('Joint') else .06 for k in self.names]],device='cuda')
        self.step=max_target_delta(self.names,self.scales,c)
        self.vel=torch.tensor([[3.1416*c['arm_velocity_limit_scale'] if k.startswith('Joint') else 10. for k in self.names]],device='cuda')
        self.home=torch.tensor([c['home'][name] for name in self.names],device='cuda').expand(n,-1).contiguous()
        for prop,values in [('stiffnesses',[c['arm_stiffness'].get(k,30.) for k in self.names]),
                            ('dampings',[8. if k.startswith('Joint') else 1. for k in self.names]),
                            ('max_forces',[c['arm_effort'].get(k,5.) for k in self.names]),
                            ('max_velocities',self.vel[0].tolist())]:
            # PhysX exposes articulation properties on CPU, even during CUDA simulation.
            old=getattr(self.robot,'get_dof_'+prop)()
            value=torch.tensor(values,device=old.device)[self.unorder].expand(n,-1).contiguous()
            getattr(self.robot,'set_dof_'+prop)(value,self.indices.to(old.device))
            torch.testing.assert_close(getattr(self.robot,'get_dof_'+prop)(),value)
        attr=stage.GetPrimAtPath('/physicsScene')
        self.readback={k:attr.GetAttribute('physxScene:'+k).Get() for k in ('enableGPUDynamics','broadphaseType','solverType','minPositionIterationCount','minVelocityIterationCount')}
        if not self.readback['enableGPUDynamics'] or self.readback['broadphaseType']!='GPU' or self.readback['solverType']!='TGS' or self.readback['minPositionIterationCount']!=64 or self.readback['minVelocityIterationCount']!=4 or not self.robot.get_dof_positions().is_cuda:
            raise RuntimeError('CUDA physics/solver verification failed: '+str(self.readback))

    def read_contacts(self):
        for lane,kind,view in self.contact_views:
            force,_,_,_,count,start,other=view.get_raw_contact_data(1.)
            count=count.cpu().numpy().astype(int);start=start.cpu().numpy().astype(int)
            if count.sum()==0: continue
            if count.sum()>=view.max_contact_data_count or (start[count>0]+count[count>0]).max()>view.max_contact_data_count:
                raise RuntimeError('Contact buffer exhausted')
            ix=np.concatenate([np.arange(i,i+n) for i,n in zip(start,count) if n])
            ids=other.cpu().numpy()[ix];force=np.abs(force.cpu().numpy().reshape(-1)[ix])
            unique=np.unique(ids);new=[int(i) for i in unique if int(i) not in self.paths]
            if new:
                # get_other_actor_paths_from_ids expects a host tensor with the same integer representation.
                import warp as wp
                keys=wp.array(np.array(new).astype(np.uint64),dtype=wp.uint64,device='cpu')
                paths=self.sv_warp_contact.get_other_actor_paths_from_ids(keys)
                for i,path in zip(new,paths):
                    mat=LANE.match(path)
                    if not mat:raise RuntimeError(f'Unmapped contact actor: {path}')
                    self.paths[i]=(int(mat.group(1)),mat.group(2))
            for i in unique:
                env,p=self.paths[int(i)]
                if lane!=env:self.cross+=1;raise ValueError('Cross-lane contact')
                vals=force[ids==i]
                if not np.isfinite(vals).all():raise ValueError('Nonfinite contact impulse')
                if kind=='robot':
                    # Each robot-robot contact is reported from both rigid body sensors.
                    if p.startswith('/Robot/'): self.contacts[lane,10]+=vals.sum()/2
                    if p.startswith('/Table'): self.contacts[lane,11]+=vals.sum()
                else:
                    if p.startswith('/Table'):self.contacts[lane,12]+=vals.sum()
                    if p.startswith('/Robot/'):
                        for side,(name,suffix) in enumerate((('left','L'),('right','R'))):
                            if '/'+name+'_' in p or p.endswith('/Link7_'+suffix):self.hand_impulse[lane,side]+=vals.sum()
                        col=next((j for j,(s,n) in enumerate((s,n) for s in ('left','right') for n in FINGERS) if '/'+s+'_'+n+'_' in p),None)
                        if col is not None:
                            self.contacts[lane,col]+=vals.sum();self.feature_contacts[lane,col]+=vals[vals>1e-7].sum()

    def reset(self,seeds):
        self.reset_xyz=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); pos=np.asarray(self.cfg['cube_xyz']).copy()
            pos[:2]+=rng.uniform(-.015,.015,2);pos[2]=self.cfg['table_z']+.0305
            yaw=rng.uniform(-.15,.15)
            self.reset_xyz.append([*pos,0,0,np.sin(yaw/2),np.cos(yaw/2)])
        v=torch.tensor(self.reset_xyz,dtype=torch.float32,device='cuda');v[:,:3]+=self.offset
        q=self.home[:,self.unorder].contiguous()
        self.robot.set_dof_positions(q,self.indices);self.robot.set_dof_velocities(torch.zeros_like(q),self.indices)
        self.robot.set_dof_position_targets(q,self.indices)
        for obj in self.views.values():
            obj.set_transforms(v,self.indices);obj.set_velocities(torch.zeros(self.n,6,device='cuda'),self.indices)
        self.contacts.fill(0);self.feature_contacts.fill(0);self.hand_impulse.fill(0)
        for _ in range(8):self.sim.step(render=False);self.read_contacts()
        xyz=self.core()[:,:3]
        if (xyz-v[:,:3]+self.offset).abs().max()>.015:raise ValueError('Reset poses do not match lane offsets')
        if self.cross:raise ValueError('Cross-lane contact or unmapped actor')
        torch.testing.assert_close(self.robot.get_root_transforms()[:,:3],self.offset+torch.tensor([0.,0.,self.cfg['robot_z']],device='cuda'),atol=1e-4,rtol=0)
        print('GPU_SCENE',self.readback,'table impulses',self.contacts[:,12].tolist(),'robot roots',self.robot.get_root_transforms()[:,:3].tolist(),flush=True)

    def core(self):
        v=self.views['Core'].get_transforms().clone();v[:,:3]-=self.offset;return v

    def lane_gain(self,lane,property_name,canonical_ids,values):
        # Tensor API accepts a full property buffer and writes only the indexed lane.
        view=self.robot
        buf=getattr(view,'get_dof_'+property_name)().clone()
        idx=[self.ids[i] for i in canonical_ids]
        buf[lane,idx]=values.reshape(-1).to(buf.device)
        getattr(view,'set_dof_'+property_name)(buf,self.indices[lane:lane+1].to(buf.device))
        torch.testing.assert_close(getattr(view,'get_dof_'+property_name)(),buf)

    def step_physics(self,cmd):
        self.robot.set_dof_position_targets(cmd[:,self.unorder].contiguous(),self.indices)
        self.contacts.fill(0);self.feature_contacts.fill(0);self.hand_impulse.fill(0)
        for _ in range(self.dec):self.sim.step(render=False);self.read_contacts()
        if self.cross:raise ValueError('Cross-lane contact')
