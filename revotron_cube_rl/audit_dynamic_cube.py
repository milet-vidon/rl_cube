"""Passive free-core fall test; not a robot grasp or layer-turn success test."""
from pathlib import Path
import os,json,hashlib,time
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import close_application,prepare_single_world,simulation_config,physx_config
app=AppLauncher(headless=True,enable_cameras=False,kit_args=platform_kit_args('')).app
try:
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom
    import omni.physx
    from dynamic_cube import spawn_dynamic_cube
    root=Path(__file__).resolve().parent
    out=root/'runs/dynamic_cube_passive_v2';out.mkdir(exist_ok=False,parents=True)
    config={'dt':1/240,'steps':480,'position':[.5,0.,1.1],'table_z':.9,'core_kinematic':False}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    sim=sim_utils.SimulationContext(simulation_config(dt=config['dt'],device='cpu',use_fabric=False,physx=physx_config(solver_type=1,min_position_iteration_count=32,max_position_iteration_count=64,min_velocity_iteration_count=4)))
    stage=sim.stage
    path=Path(os.environ['ISAAC_MUJOCO_CUBE_ROOT'])/'cube_model.usda'
    model=spawn_dynamic_cube(stage,path,config['position'])
    table=sim_utils.CuboidCfg(size=(.8,.9,.04),collision_props=sim_utils.CollisionPropertiesCfg())
    table.func('/World/Table',table,translation=(.5,0.,.88))
    sim.set_setting('/physics/updateToUsd',True)
    prepare_single_world(sim);sim.reset()
    trajectory=[];started=time.monotonic()
    for step in range(config['steps']):
        sim.step(render=False)
        omni.physx.get_physx_interface().update_transformations(False,True,True,False)
        cache=UsdGeom.XformCache()
        positions={p:list(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation()) for p in model['bodies']}
        trajectory.append({'step':step,'positions':positions})
    z=[r['positions']['/World/Core'][2] for r in trajectory]
    final=trajectory[-1]['positions'];core=final['/World/Core']
    import numpy as np
    radius=max(float(np.linalg.norm(np.array(v)-core)) for k,v in final.items() if k!='/World/Core')
    passed=(len(model['active_physics_scenes'])==1 and z[0]-z[-1]>.1 and .91<z[-1]<.96 and max(z[-60:])-min(z[-60:])<.002 and radius<.07)
    result={'kind':'passive_dynamic_cube_asset_test','passed':passed,'model':model,'seed':None,'seed_reason':'deterministic passive asset test; no learned policy or randomized reset','steps':config['steps'],'initial_core_z':z[0],'final_core_z':z[-1],'final_core_jitter_m':max(z[-60:])-min(z[-60:]),'max_cubelet_radius_m':radius,'elapsed_s':time.monotonic()-started,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime_override_sha256':hashlib.sha256((root/'dynamic_cube.py').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),'robot_present':False,'turn_success_claim':False,'checkpoint':None}
    (out/'trajectory.json').write_text(json.dumps(trajectory));(out/'result.json').write_text(json.dumps(result,indent=2));print('DYNAMIC_CUBE_AUDIT',json.dumps(result),flush=True)
    if not passed:raise RuntimeError('Passive free-core cube validation failed')
finally:
    close_application(app)
