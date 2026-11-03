from pathlib import Path
import json,sys
ROOT=Path('/home/brainco/rl_cube/github_project/revotron_cube_rl');sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app=AppLauncher(headless=True,enable_cameras=False,kit_args=platform_kit_args('')).app
env=None
try:
 from articulated_cube_env import DynamicPickupEnv
 from isaac_compat import tensor
 from pxr import UsdPhysics
 out=ROOT/'runs/cube_physics_property_probe_20261103';out.mkdir(exist_ok=False)
 env=DynamicPickupEnv(out,seed=9501,vision=False,preview=False,auto_reset=False,arm_damping=8.)
 scene=env.stage.GetPrimAtPath('/physicsScene')
 report={'physics_hz':env.cfg['physics_hz'],'control_hz':env.cfg['control_hz'],'scene':{},'bodies':{}}
 for name in ('solverType','minPositionIterationCount','maxPositionIterationCount','minVelocityIterationCount','maxVelocityIterationCount','enableExternalForcesEveryIteration','enableStabilization'):
  report['scene'][name]=scene.GetAttribute('physxScene:'+name).Get()
 for name,obj in env.cube.objects.items():
  if name not in ('Core','nZ','nZ_pX','nX_nY_nZ'):continue
  body={}
  for field in ('body_mass','body_inertia','default_mass','default_inertia'):
   try:body[field]=tensor(getattr(obj.data,field)).tolist()
   except (AttributeError,TypeError) as e:body[field]=str(e)[:120]
  report['bodies'][name]=body
 (out/'result.json').write_text(json.dumps(report,indent=2));print('PHYSICS_PROPERTY_PROBE',json.dumps(report),flush=True)
finally:
 from isaac_compat import close_application
 if env is not None:env.close()
 close_application(app)
