"""Single native PhysX prototype with batched contact sensors across isolated GPU environments."""
from pathlib import Path
import re
from contact_hold_cuda_scene_reduce import GpuContacts
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
            # The native attach transient needed 397874 aggregate pairs for 128 worlds.
            # Keep room for attach plus real contacts; PhysX otherwise silently discards interactions.
            gpu_found_lost_aggregate_pairs_capacity=max(2**17,64*n*min(n,64)),
            gpu_max_rigid_contact_count=max(2**19,n*1024),gpu_max_rigid_patch_count=max(2**16,n*128))
        self.sim=sim_utils.SimulationContext(simulation_config(dt=self.dt,device='cuda:0',use_fabric=False,render_interval=self.dec,physx=phys))
        stage=self.sim.stage;self.stage=stage
        self.sim.set_setting('/physics/disableContactProcessing',False)
        # Clone the rigid bodies, articulations and joints in PhysX. Only the prototype is parsed.
        import math
        from pxr import Sdf, UsdUtils
        side=math.ceil(math.sqrt(n))
        positions=np.array([[(k%side-(side-1)/2)*4.,(k//side-(side-1)/2)*4.,0.] for k in range(n)],np.float32)
        self.offset=torch.as_tensor(positions,device='cuda')
        roots=[f'/World/envs/env_{k:04d}' for k in range(n)];root=roots[0]
        UsdGeom.Xform.Define(stage,'/World/envs');proto=UsdGeom.Xform.Define(stage,root)
        proto.GetPrim().GetReferences().AddReference(str(path.resolve()),'/World')
        bodies=[v for v in Usd.PrimRange(proto.GetPrim()) if v.HasAPI(UsdPhysics.RigidBodyAPI) and
                ('/Cube/' in str(v.GetPath()) or v.GetName()=='Core')]
        if len(bodies)!=27 or any(UsdPhysics.RigidBodyAPI(v).GetKinematicEnabledAttr().Get() for v in bodies):
            raise ValueError('Expected the original free 27-body cube')
        for v in bodies:PhysxSchema.PhysxContactReportAPI.Apply(v).CreateThresholdAttr(0.)
        for v in Usd.PrimRange(proto.GetPrim()):
            if v.IsA(UsdPhysics.Joint):
                for rel in (UsdPhysics.Joint(v).GetBody0Rel(),UsdPhysics.Joint(v).GetBody1Rel()):
                    if any(not str(t).startswith(root+'/') for t in rel.GetTargets()):raise ValueError('External joint target')
        cube_paths={v.GetName():str(v.GetPath()).replace(root,'/World/envs/env_*') for v in bodies}
        robot_patterns=[str(v.GetPath()).replace(root,'/World/envs/env_*') for v in Usd.PrimRange(stage.GetPrimAtPath(root+'/Robot')) if v.HasAPI(UsdPhysics.RigidBodyAPI)]
        proto.AddTranslateOp().Set(Gf.Vec3d(*map(float,positions[0])))
        layer=stage.GetRootLayer()
        with Sdf.ChangeBlock():
            for j in range(1,n):
                spec=Sdf.CreatePrimInLayer(layer,roots[j]);spec.specifier=Sdf.SpecifierDef;spec.typeName='Xform'
                spec.inheritPathList.Prepend(Sdf.Path(root))
                Sdf.AttributeSpec(spec,'xformOp:translate',Sdf.ValueTypeNames.Double3).default=Gf.Vec3d(*map(float,positions[j]))
        prepare_single_world(self.sim)
        # Same native env-ID replication API as Isaac's Cloner. Preserve our padded paths for tensor lane order.
        stage.GetPrimAtPath('/physicsScene').CreateAttribute('physxScene:envIdInBoundsBitCount',Sdf.ValueTypeNames.Int).Set(4)
        native=omni.physx.get_physx_replicator_interface()
        sid=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        # Separate source poses reduce transiently overlapping aggregates at attach.
        group_size=64
        base=[0]
        def attach_end(stage_id):
            for lo in range(0,n,group_size):
                base[0]=lo
                native.replicate(stage_id,roots[lo],min(group_size,n-lo)-1,True,False)
        self.replication_callbacks=(lambda _:['/World/envs'],attach_end,lambda _p,i:roots[base[0]+i+1])
        native.register_replicator(sid,*self.replication_callbacks)
        print('NATIVE_CLONES',n,flush=True)
        self.paths={};self.contacts=np.zeros((n,13),np.float64);self.cross=0;self.feature_contacts=np.zeros((n,10))

        self.sim.reset()
        self.sv=tensors.create_simulation_view('torch')
        self.sv.set_subspace_roots('/')
        self.robot=self.sv.create_articulation_view('/World/envs/env_*/Robot/root_joint')
        if self.robot.count!=n:raise RuntimeError(f'Expected {n} robot articulations, got {self.robot.count}')
        names=self.robot.shared_metatype.dof_names
        self.names=c['joint_names'];self.ids=[names.index(name) for name in self.names]
        self.unorder=np.argsort(self.ids).tolist()
        self.indices=torch.arange(n,device='cuda',dtype=torch.int32)
        self.views={k:self.sv.create_rigid_body_view(v) for k,v in cube_paths.items()}
        for view in (self.robot,*self.views.values()):
            if [int(LANE.match(p).group(1)) for p in view.prim_paths]!=list(range(n)): raise ValueError('Wrong lane mapping')
        self.sv_warp=tensors.create_simulation_view('warp')
        self.sv_warp_contact=self.sv_warp.create_rigid_contact_view(root+'/Core',max_contact_data_count=16)
        self.contact_views=[];self.lookup={}
        for kind,patterns in [('robot',robot_patterns),('cube',[v for k,v in cube_paths.items() if k!='Core'])]:
            view=self.sv.create_rigid_contact_view(patterns,[[] for _ in patterns],max_contact_data_count=n*512)
            paths=list(view.sensor_paths)
            if len(paths)!=len(patterns)*n or len(set(paths))!=len(paths) or len(paths)!=view.sensor_count:
                raise ValueError(f'Wrong {kind} contact sensor coverage: {len(paths)}, {view.sensor_count} vs {len(patterns)*n}')
            lanes=np.array([int(LANE.match(v).group(1)) for v in paths])
            if not np.all(np.bincount(lanes,minlength=n)==len(patterns)):raise ValueError('Contact lane mapping mismatch')
            self.contact_views.append((kind,view,lanes))
        self.reducer=GpuContacts(self)
        self.hard=self.robot.get_dof_limits()[:,self.ids].cuda()
        torch.testing.assert_close(self.hard,self.hard[:1].expand_as(self.hard))

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
        self.reducer.update(self)

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
        self.reducer.clear(self)
        for _ in range(8):self.sim.step(render=False);self.read_contacts()
        self.reducer.validate()
        xyz=self.core()[:,:3]
        if (xyz-v[:,:3]+self.offset).abs().max()>.015:raise ValueError('Reset poses do not match lane offsets')
        if self.cross:raise ValueError('Cross-lane contact or unmapped actor')
        torch.testing.assert_close(self.robot.get_root_transforms()[:,:3],self.offset+torch.tensor([0.,0.,self.cfg['robot_z']],device='cuda'),atol=1e-4,rtol=0)
        print('GPU_SCENE',self.readback,'lanes',self.n,'table impulse range',[self.contacts[:,12].min(),self.contacts[:,12].max()],flush=True)

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

    def batch_gain(self,lanes,property_name,canonical_ids,values):
        """One tensor API call for all changed lanes. Isaac gain property buffers live on CPU."""
        buf=getattr(self.robot,'get_dof_'+property_name)().clone()
        lane_ids=torch.as_tensor(lanes,device=buf.device,dtype=torch.long).reshape(-1)
        if lane_ids.numel()==0:return
        idx=torch.tensor([self.ids[i] for i in canonical_ids],device=buf.device)
        buf[lane_ids[:,None],idx]=torch.as_tensor(values,device=buf.device,dtype=buf.dtype)
        getattr(self.robot,'set_dof_'+property_name)(buf,lane_ids.int())
        torch.testing.assert_close(getattr(self.robot,'get_dof_'+property_name)()[lane_ids[:,None],idx],buf[lane_ids[:,None],idx])

    def step_physics(self,cmd):
        self.robot.set_dof_position_targets(cmd[:,self.unorder].contiguous(),self.indices)
        self.reducer.clear(self)
        for _ in range(self.dec):self.sim.step(render=False);self.read_contacts()
        self.reducer.validate()
        if self.cross:raise ValueError('Cross-lane contact')
