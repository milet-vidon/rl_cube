"""Use the unchanged 56-DOF safety environment with a free 27-body cube.

The scoped factory replacement affects initialization only. The certified
locked-cube environment and its immutable source hash remain unchanged.
"""
import json,os
from pathlib import Path
import torch
from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdShade
from isaaclab.assets import RigidObject as RealRigidObject,RigidObjectCfg
from isaac_compat import tensor,quaternion_from_native
from dynamic_cube import spawn_dynamic_cube
import environment as base


class DynamicCubeHandle:
    def __init__(self,stage,path,position,friction,hand_friction=None):
        self.model=spawn_dynamic_cube(stage,path,position);self.allow_pose_reset=False;self.pose_reset_count=0
        material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/Material'))
        self.model['source_friction']=[material.GetStaticFrictionAttr().Get(),material.GetDynamicFrictionAttr().Get()]
        material.CreateStaticFrictionAttr(friction[0]);material.CreateDynamicFrictionAttr(friction[1])
        self.model.update(runtime_model_version='dynamic_grasp_v3',runtime_friction=list(friction),friction_basis='same 0.9/0.8 values as certified locked-cube grasp; not real-robot calibration')
        if hand_friction is not None:
            hand=UsdShade.Material.Define(stage,'/World/RobotHandContactMaterial')
            physics=UsdPhysics.MaterialAPI.Apply(hand.GetPrim());physics.CreateStaticFrictionAttr(hand_friction[0]);physics.CreateDynamicFrictionAttr(hand_friction[1]);physics.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(hand.GetPrim()).CreateFrictionCombineModeAttr('max')
            bound=[]
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(('/World/Robot/left_','/World/Robot/right_')) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    binding=UsdShade.MaterialBindingAPI.Apply(prim);binding.Bind(hand,UsdShade.Tokens.strongerThanDescendants,'physics')
                    resolved,_=binding.ComputeBoundMaterial('physics')
                    if resolved.GetPath()!=hand.GetPath():raise RuntimeError('Hand contact material binding did not resolve')
                    bound.append(str(prim.GetPath()))
            if not bound:raise RuntimeError('No hand collision shapes found for contact material')
            self.model.update(runtime_model_version='dynamic_grasp_pair_materials_v4',runtime_hand_friction=list(hand_friction),hand_friction_combine_mode='max',hand_material_colliders=bound,friction_basis='independently configured cube/cube and hand/cube friction; uncalibrated simulation development; values recorded explicitly',material_api_reference='https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html')
        self.objects={}
        for name in self.model['bodies']:
            prim=stage.GetPrimAtPath(name);PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            self.objects[prim.GetName()]=RealRigidObject(RigidObjectCfg(prim_path=name,spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position))))
        self.core=self.objects['Core'];self.pending_update=0.
    @property
    def data(self):return self.core.data
    def reset(self):
        for obj in self.objects.values():obj.reset()
    def write_root_state_to_sim(self,state):
        if not self.allow_pose_reset:raise RuntimeError('Cube pose writes are allowed only during episode reset')
        # All source body origins are the same cube center; geometry is offset in each mesh.
        for obj in self.objects.values():obj.write_root_state_to_sim(state.clone())
        self.pose_reset_count+=1
    def write_data_to_sim(self):
        # No cube forces, torques, drives or pose targets are submitted.
        pass
    def update(self,dt):
        self.core.update(dt);self.pending_update+=dt
    def states(self):
        result={}
        for name,obj in self.objects.items():
            if name!='Core':obj.update(self.pending_update)
            s=tensor(obj.data.root_state_w)[0]
            result[name]={'position_m':s[:3].tolist(),'quaternion_wxyz':quaternion_from_native(s[3:7]).tolist(),'linear_velocity_m_s':s[7:10].tolist(),'angular_velocity_rad_s':s[10:13].tolist()}
        self.pending_update=0.
        return result


class DynamicPickupEnv(base.CubePickupEnv):
    def __init__(self,run_dir,friction=(.9,.8),hand_friction=None,**kwargs):
        self.run_dir=Path(run_dir);self.friction=friction
        self.dynamic_source=Path(os.environ.get('ISAAC_MUJOCO_CUBE_ROOT',str(base.ROOT.parent/'isaac_mujoco_cube')))/'cube_model.usda'
        old_builder,old_factory=base.build_locked_cube,base.RigidObject
        base.build_locked_cube=lambda:self.dynamic_source
        base.RigidObject=lambda cfg:DynamicCubeHandle(self.stage,self.dynamic_source,cfg.init_state.pos,self.friction,hand_friction)
        try:super().__init__(**kwargs)
        finally:base.build_locked_cube,base.RigidObject=old_builder,old_factory
        self.cfg['cube_mode']='free_core_27_body_contact_turn';self.contract['cube_mode']=self.cfg['cube_mode']
        self.contract['dynamic_cube']=self.cube.model
    def reset(self):
        self.cube.allow_pose_reset=True
        try:super().reset()
        finally:self.cube.allow_pose_reset=False
    def export_scene(self):
        # Keep the dynamic cube and original joint relationships in this run's scene.
        self.stage.GetRootLayer().Export(str(self.run_dir/'dynamic_training_scene.usda'))
