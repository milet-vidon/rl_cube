from isaac_compat import prepare_single_world, normalize_windows_asset_paths
"""Single-environment CPU PhysX + real chest RTX RGB baseline for RSL-RL.
An asymmetric critic sees simulation state; the actor receives RGB and proprioception only.
"""
from isaac_compat import tensor, quaternion_to_native, quaternion_from_native, physx_config, simulation_config, runtime_versions, LAB3
from pathlib import Path
import os
import json, math
import numpy as np
import torch
from tensordict import TensorDict
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
import omni.physx
from assets import build_locked_cube, fixed_frame
from rewards import pickup_reward
from joint_safety import soft_limits_from_hard, max_target_delta, limit_metadata, project_target
ROOT=Path(__file__).resolve().parent

class CubePickupEnv:
    def __init__(self,seed=7,vision=True,preview=False,auto_reset=True,arm_damping=None):
        self.cfg=json.loads((ROOT/'config.json').read_text());c=self.cfg
        self.arm_damping=c.get('arm_damping',1.) if arm_damping is None else arm_damping
        self.num_envs=1;self.num_actions=56;self.device='cpu';self.vision=vision;self.auto_reset=auto_reset
        self.proprio_dim=224
        self.dt=1/c['physics_hz'];self.decimation=c['physics_hz']//c['control_hz'];self.control_dt=self.dt*self.decimation
        self.max_episode_length=c.get('episode_steps',600);self.episode_length_buf=torch.zeros(1,dtype=torch.long)
        self.rng=np.random.default_rng(seed);torch.manual_seed(seed)
        self.sim=sim_utils.SimulationContext(simulation_config(dt=self.dt,device='cpu',use_fabric=False,render_interval=self.decimation,physx=physx_config(solver_type=1,min_position_iteration_count=32,max_position_iteration_count=64,min_velocity_iteration_count=4)))
        self.stage=self.sim.stage
        self.sim.set_setting('/physics/disableContactProcessing',False)
        self.sim.set_setting('/physics/updateToUsd',True)
        self.sim.set_setting('/physics/updateVelocitiesToUsd',True)
        robot_path=Path(os.environ.get('REVOTRON_USD_PATH', str(ROOT.parent/'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd'))).expanduser().resolve()
        if not robot_path.is_file():
            raise FileNotFoundError(f'Revo3 USD not found: {robot_path}. Set REVOTRON_USD_PATH for this host.')
        arms=[n for n in c['joint_names'] if n.startswith('Joint')];fingers=[n for n in c['joint_names'] if not n.startswith('Joint')]
        cfg=ArticulationCfg(prim_path='/World/Robot',spawn=sim_utils.UsdFileCfg(usd_path=str(robot_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=c['gravity_compensation_robot'],max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=32,solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,c['robot_z']),joint_pos=c['home']),
            actuators={'arms':ImplicitActuatorCfg(joint_names_expr=arms,stiffness=c['arm_stiffness'],damping=self.arm_damping,effort_limit_sim=c['arm_effort'],velocity_limit_sim=3.1416*c.get('arm_velocity_limit_scale',1.0)),
                       'fingers':ImplicitActuatorCfg(joint_names_expr=fingers,stiffness=30.,damping=1.,effort_limit_sim=5.,velocity_limit_sim=10.)},soft_joint_pos_limit_factor=1.)
        self.robot=Articulation(cfg)
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Robot/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        # Official single-arm lift exclusion for co-located wrist housings, applied to each arm.
        for side in ('L','R'):
            p=self.stage.GetPrimAtPath('/World/Robot/Link5_'+side)
            UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().AddTarget('/World/Robot/Link7_'+side)
        table=sim_utils.CuboidCfg(size=(.8,.9,.04),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.32,.35,.4)),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7))
        table.func('/World/Table',table,translation=(.60,0.,c['table_z']-.02))
        ground=sim_utils.CuboidCfg(size=(5.,5.,.02),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.17,.19,.22)))
        ground.func('/World/Ground',ground,translation=(0.,0.,-.011))
        cube_path=build_locked_cube()
        self.cube=RigidObject(RigidObjectCfg(prim_path='/World/Cube',spawn=sim_utils.UsdFileCfg(usd_path=str(cube_path),activate_contact_sensors=True,rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,max_depenetration_velocity=1.,solver_position_iteration_count=32,solver_velocity_iteration_count=4)),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(c['cube_xyz']))))
        self.frames={}
        for side in ('left','right'):
            self.frames[side]=fixed_frame(self.stage,'/World/Robot',side+'_palm_link')
        light=sim_utils.DomeLightCfg(intensity=1800.);light.func('/World/Light',light)
        self.camera=None;self.overview=None
        if vision:
            eye=Gf.Vec3d(*c['camera_eye']);target=Gf.Vec3d(*c['camera_target'])
            m=Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse();q=m.ExtractRotationQuat()
            cp='/World/Robot/world/ChestCamera'
            self.camera=Camera(CameraCfg(prim_path=cp,height=c['camera_resolution'],width=c['camera_resolution'],update_period=0.,update_latest_camera_pose=True,data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=16.,horizontal_aperture=24.,clipping_range=(.02,5.)),
                offset=CameraCfg.OffsetCfg(pos=(eye[0],eye[1],eye[2]-c['robot_z']),rot=quaternion_to_native((q.GetReal(),*q.GetImaginary())),convention='opengl')))
            mount=sim_utils.CuboidCfg(size=(.025,.06,.025),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.02,.025,.03)))
            mount.func('/World/Robot/world/ChestMount',mount,translation=(eye[0]-.02,eye[1],eye[2]-c['robot_z']))
            if preview:
                m=Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.1,-1.9,1.9),Gf.Vec3d(.20,0,1.05),Gf.Vec3d(0,0,1)).GetInverse();q=m.ExtractRotationQuat()
                self.overview=Camera(CameraCfg(prim_path='/World/Overview',height=512,width=640,update_latest_camera_pose=True,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=30.,horizontal_aperture=24.),offset=CameraCfg.OffsetCfg(pos=(2.1,-1.9,1.9),rot=quaternion_to_native((q.GetReal(),*q.GetImaginary())),convention='opengl')))
        self.contact_impulse=np.zeros(2);self.table_impulse=0.;self.robot_table_impulse=0.;self.contact_events=0;self.pair_impulses={};self._contact_paths={}
        self.sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.asset_path_overrides = normalize_windows_asset_paths(self.stage)
        prepare_single_world(self.sim)
        self.sim.reset()
        self.ids,_=self.robot.find_joints(c['joint_names'],preserve_order=True)
        assert len(self.ids)==56 and len(set(self.ids))==56
        self.home=torch.tensor([[c['home'][n] for n in c['joint_names']]],dtype=torch.float32)
        self.hard_limits=tensor(self.robot.data.joint_pos_limits)[:,self.ids].clone()
        self.limits=soft_limits_from_hard(self.hard_limits,c['joint_names'],c)
        assert ((self.home>=self.hard_limits[...,0]-1e-4)&(self.home<=self.hard_limits[...,1]+1e-4)).all()
        assert ((self.home>=self.limits[...,0]-1e-4)&(self.home<=self.limits[...,1]+1e-4)).all()
        self.scales=torch.tensor([[.025 if n.startswith('Joint') else .06 for n in c['joint_names']]])
        self.max_target_delta=max_target_delta(c['joint_names'],self.scales,c)
        self.command_velocity_limits=torch.tensor([[3.1416*c.get('arm_velocity_limit_scale',1.) if n.startswith('Joint') else 10. for n in c['joint_names']]])
        self.last_actions=torch.zeros(1,56);self.previous_actions=self.last_actions.clone();self.targets=self.home.clone()
        self.hold=0;self.successes=0;self.episodes=0;self.previous_potential=0.;self.reset()
        for _ in range(4):self._physics()
        self.export_scene()
        self.contract={'action_names':c['joint_names'],'sim_joint_indices':self.ids,'observation_policy':['chest_rgb','joint_position','joint_velocity','previous_action','joint_target_error'],
            'critic_only':['cube_pose','cube_velocity','palm_to_cube','contact_impulses','hold_progress','remaining_episode_fraction'],
            'cube_mode':c['cube_mode'],'cube_gravity':True,'robot_gravity_compensation':True,'self_collision':True,
            'arm_damping':self.arm_damping,
            'revolute_dofs':self.robot.num_joints,'camera_parent':'/World/Robot/world','camera_resolution':c['camera_resolution'],'upstream_commit':c['upstream_commit'],'asset_override_release':c['local_asset_release']}
        self.contract['runtime_versions']=runtime_versions()
        self.contract['asset_path_overrides']=self.asset_path_overrides
        self.contract['project_quaternion_convention']='wxyz'
        self.contract['native_quaternion_convention']='xyzw' if LAB3 else 'wxyz'
        self.contract['joint_limit_safety']=limit_metadata(self.hard_limits,self.limits,c['joint_names'],c)
        (ROOT/'asset_contract.json').write_text(json.dumps(self.contract,indent=2))

    def export_scene(self):
        path=ROOT/'training_scene.usda';self.stage.GetRootLayer().Export(str(path))
        snapshot=Usd.Stage.Open(str(path))
        for prim_name,asset in [('Robot','../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd'),('Cube','locked_cube.usda')]:
            refs=snapshot.GetPrimAtPath('/World/'+prim_name).GetReferences();refs.ClearReferences();refs.AddReference(asset)
        # Tensor actuator commands are not automatically authored to USD. Preserve
        # the pregrasp PD targets in the inspectable snapshot only.
        for prim in snapshot.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in self.cfg['home']:
                n=prim.GetName();drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
                drive.CreateTargetPositionAttr(math.degrees(self.cfg['home'][n]))
                drive.CreateStiffnessAttr(self.cfg['arm_stiffness'][n] if n.startswith('Joint') else 30.)
                drive.CreateDampingAttr(self.arm_damping if n.startswith('Joint') else 1.)
                drive.CreateMaxForceAttr(self.cfg['arm_effort'][n] if n.startswith('Joint') else 5.)
        snapshot.GetRootLayer().Save()

    def _on_contact(self,headers,data):
        for h in headers:
            paths=[]
            for x in (h.actor0,h.actor1):
                if x not in self._contact_paths:self._contact_paths[x]=str(PhysicsSchemaTools.intToSdfPath(x))
                paths.append(self._contact_paths[x])
            cube_contact=any(p.startswith('/World/Cube') for p in paths)
            robot_table=any(p.startswith('/World/Robot') for p in paths) and any(p.startswith('/World/Table') for p in paths)
            if not (cube_contact or robot_table):continue
            impulse=0.
            for item in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                impulse+=float(np.linalg.norm(np.array(item.impulse)))
            self.contact_events+=1
            key=' | '.join(paths);self.pair_impulses[key]=self.pair_impulses.get(key,0.)+impulse
            if robot_table:self.robot_table_impulse+=impulse
            if not cube_contact:continue
            for i,(side,suffix) in enumerate([('left','L'),('right','R')]):
                if any('/'+side+'_' in p or p.endswith('/Link7_'+suffix) for p in paths):self.contact_impulse[i]+=impulse
            if any('/Table' in p for p in paths):self.table_impulse+=impulse

    def _physics(self):
        self.robot.write_data_to_sim();self.cube.write_data_to_sim()
        self.sim.step(render=False);self.robot.update(self.dt);self.cube.update(self.dt)

    def _render(self):
        if self.camera:
            omni.physx.get_physx_interface().update_transformations(False,True,True,False)
            self.sim.render();self.camera.update(self.control_dt,force_recompute=True)
            if self.overview:self.overview.update(self.control_dt,force_recompute=True)

    def reset(self):
        self.robot.reset();self.cube.reset()
        q=tensor(self.robot.data.default_joint_pos).clone();q[:,self.ids]=self.home
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q));self.robot.set_joint_position_target(q)
        state=tensor(self.cube.data.default_root_state).clone()
        state[0,:2]+=torch.tensor(self.rng.uniform(-.015,.015,2),dtype=torch.float32)
        state[0,2]=self.cfg['table_z']+.0305
        angle=float(self.rng.uniform(-.15,.15));state[0,3:7]=quaternion_to_native(torch.tensor([math.cos(angle/2),0,0,math.sin(angle/2)]))
        state[:,7:]=0.;self.cube.write_root_state_to_sim(state)
        self.targets=self.home.clone();self.last_actions.zero_();self.previous_actions.zero_();self.episode_length_buf.zero_();self.hold=0;self.previous_potential=0.
        self.contact_impulse[:]=0.;self.table_impulse=0.;self.robot_table_impulse=0.
        for _ in range(8):self._physics()
        initial_distances=[torch.linalg.norm(self.palm_pos(s)[0]-tensor(self.cube.data.root_pos_w)[0]).item() for s in ('left','right')]
        self.previous_potential=.2*sum(math.exp(-d/.15) for d in initial_distances)/2
        self._render()

    def palm_pos(self,side):
        name,pos,quat=self.frames[side];ids,_=self.robot.find_bodies(name)
        center=torch.tensor([pos],dtype=torch.float32)+quat_apply(quaternion_to_native(torch.tensor([quat],dtype=torch.float32)),torch.tensor([[0.,0.,.09]]))
        return tensor(self.robot.data.body_link_pos_w)[:,ids[0]]+quat_apply(tensor(self.robot.data.body_link_quat_w)[:,ids[0]],center)

    def get_observations(self):
        q=tensor(self.robot.data.joint_pos)[:,self.ids];v=tensor(self.robot.data.joint_vel)[:,self.ids]
        proprio=torch.cat(((q-self.home),v*.1,self.last_actions,self.targets-q),dim=-1)
        rgb=tensor(self.camera.data.output['rgb'])[...,:3].float().reshape(1,-1)/255. if self.camera else torch.zeros(1,self.cfg['camera_resolution']**2*3)
        cube=tensor(self.cube.data.root_state_w).clone()
        cube[:,3:7]=quaternion_from_native(cube[:,3:7])
        delta=torch.cat([self.palm_pos(s)-cube[:,:3] for s in ('left','right')],dim=-1)
        contacts=torch.tensor(self.contact_impulse,dtype=torch.float32).reshape(1,2).clamp(max=.5)
        progress=torch.tensor([[self.hold/(self.cfg['success']['hold_seconds']*self.cfg['control_hz']),1.-float(self.episode_length_buf[0])/self.max_episode_length]])
        obs=TensorDict({'policy':torch.cat((proprio,rgb),-1),'critic':torch.cat((proprio,cube,delta,contacts,progress),-1)},batch_size=[1])
        assert all(torch.isfinite(x).all() for x in obs.values()),'Non-finite observation'
        return obs

    def step(self,actions):
        assert actions.shape==(1,56) and torch.isfinite(actions).all()
        self.previous_actions=self.last_actions.clone();self.last_actions=actions.clamp(-1,1).cpu().clone()
        # Incremental joint-position commands -> official implicit PD actuators.
        requested=self.targets+self.scales*actions.cpu()
        self.targets,soft_hit=project_target(self.targets,requested,self.limits,
            torch.minimum(self.scales,self.max_target_delta),self.command_velocity_limits,self.control_dt)
        self.robot.set_joint_position_target(self.targets,joint_ids=self.ids)
        self.contact_impulse[:]=0.;self.table_impulse=0.;self.robot_table_impulse=0.
        for _ in range(self.decimation):self._physics()
        self.episode_length_buf+=1;self._render()
        state=tensor(self.cube.data.root_state_w)[0];pos=state[:3]
        # Palm-center offset is represented geometrically; exact contact is required for lifting reward.
        distances=[torch.linalg.norm(self.palm_pos(s)[0]-pos).item() for s in ('left','right')]
        height=float(pos[2])-(self.cfg['table_z']+.0285)
        drop=float(pos[2])<self.cfg['table_z']-.12 or float(torch.linalg.norm(pos[:2]-torch.tensor(self.cfg['cube_xyz'][:2])))>.45
        timeout=int(self.episode_length_buf[0])>=self.max_episode_length
        velocity_limits=torch.tensor([3.1416 if n.startswith('Joint') else 10. for n in self.cfg['joint_names']])
        reward,terms,self.hold,success,bilateral=pickup_reward(self.cfg,distances,height,self.contact_impulse,self.table_impulse,
            float(state[7:10].norm()),float(state[10:13].norm()),self.hold,float((self.last_actions-self.previous_actions).square().mean()),
            float((tensor(self.robot.data.joint_vel)[:,self.ids]/velocity_limits).square().clamp(max=1.).mean()),drop,
            previous_potential=self.previous_potential,timeout=timeout)
        self.previous_potential=terms['potential']
        done=success or drop or timeout
        self.metrics={'height_m':height,'bilateral_contact':bilateral,'contact_impulse':self.contact_impulse.tolist(),'table_impulse':self.table_impulse,'robot_table_impulse':self.robot_table_impulse,
                      'hold_steps':self.hold,'success':success,'drop':drop,'reward':reward,'terms':terms,'cube_position':pos.tolist(),
                      'linear_speed':float(state[7:10].norm()),'angular_speed':float(state[10:13].norm())}
        actual=tensor(self.robot.data.joint_pos)[:,self.ids]
        velocity=tensor(self.robot.data.joint_vel)[:,self.ids]
        arm=[i for i,n in enumerate(self.cfg['joint_names']) if n.startswith('Joint')]
        hard_hit=(actual<self.hard_limits[...,0]-1e-4)|(actual>self.hard_limits[...,1]+1e-4)
        actual_soft_hit=(actual<self.limits[...,0]-1e-4)|(actual>self.limits[...,1]+1e-4)
        self.metrics.update({'requested_target':requested[0].tolist(),'applied_target':self.targets[0].tolist(),
            'soft_limit_hit':soft_hit[0].tolist(),'arm_soft_limit_hit':bool(soft_hit[:,arm].any()),
            'actual_arm_soft_limit_violation':bool(actual_soft_hit[:,arm].any()),
            'hard_limit_hit':hard_hit[0].tolist(),'joint_velocity':velocity[0].tolist(),
            'joint_position_error':(self.targets-actual)[0].tolist(),
            'cube_table_contact':bool(self.table_impulse>=1e-5)})
        if done:
            self.episodes+=1;self.successes+=int(success)
            if self.auto_reset:self.reset()
        obs=self.get_observations()
        return obs,torch.tensor([reward]),torch.tensor([done]),{}  # time limits treated as finite-horizon task ends

    def save_rgb(self,path):
        from PIL import Image
        assert self.camera is not None
        Image.fromarray(tensor(self.camera.data.output['rgb'])[0,:,:,:3].cpu().numpy()).save(path)
        if self.overview:Image.fromarray(tensor(self.overview.data.output['rgb'])[0,:,:,:3].cpu().numpy()).save(Path(path).with_name('overview_'+Path(path).name))
        calibration={'source':'Isaac CameraData, simulation calibration only','K':tensor(self.camera.data.intrinsic_matrices)[0].cpu().tolist(),
                     'position_world_m':tensor(self.camera.data.pos_w)[0].cpu().tolist(),'orientation_wxyz_ros':quaternion_from_native(tensor(self.camera.data.quat_w_ros))[0].cpu().tolist(),
                     'shape':list(tensor(self.camera.data.output['rgb']).shape),'parent':'/World/Robot/world','root_position_m':[0.,0.,self.cfg['robot_z']]}
        Path(path).with_suffix('.camera.json').write_text(json.dumps(calibration,indent=2,allow_nan=False))

    def close(self):
        self.sub=None;self.sim.stop()
