"""Parse the operator scene's composed, mounted physics topology on CPU."""
import json
from pathlib import Path


def check_demo_usd(source):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source = Path(source)
    scene = source/'dynamic_training_scene.usda'
    if not scene.is_file():
        raise ValueError(f'Missing demo scene at {scene}')
    root = Sdf.Path('/World/envs/env_0000')
    robot = root.AppendChild('Robot')
    anchor = robot.AppendChild('root_joint')
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, root.GetParentPath())
    mounted = UsdGeom.Xform.Define(stage, root).GetPrim()
    mounted.GetReferences().AddReference(str(scene.resolve()), '/World')
    robot_prim = stage.GetPrimAtPath(robot)
    if not robot_prim:
        raise ValueError(f'No robot composed at {robot}')

    # The topology check reads links and joints. Skip rendering meshes and
    # vendor collision shapes that the generic USD parser cannot classify.
    geometry = [child.GetPath() for link in robot_prim.GetChildren() if link.HasAPI(UsdPhysics.RigidBodyAPI)
                for child in link.GetChildren() if child.GetName() in ('visuals', 'collisions')]
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root], geometry)
    kinds = UsdPhysics.ObjectType
    paths, robots = parsed.get(kinds.Articulation, ([], []))
    if list(paths) != [anchor] or not robots[0].isValid or list(robots[0].rootPrims) != [anchor]:
        raise ValueError(f'Expected one mounted USD articulation rooted at {anchor}, found {list(paths)}')
    joint_names = json.loads((source/'config.json').read_bytes())['joint_names']
    expected_joints = {robot.AppendPath('joints/'+name) for name in joint_names}
    if len(expected_joints) != 56 or set(robots[0].articulatedJoints) != expected_joints | {anchor}:
        raise ValueError('Mounted robot articulation does not contain the configured 56 joints')
    fixed = dict(zip(*parsed.get(kinds.FixedJoint, ([], [])), strict=True)).get(anchor)
    if (fixed is None or not fixed.isValid or not fixed.jointEnabled or fixed.body0
            or fixed.body1 != robot.AppendChild('world') or fixed.excludeFromArticulation):
        raise ValueError(f'Invalid robot world anchor at {anchor}')

    cube = root.AppendChild('Cube')
    core = root.AppendChild('Core')
    pieces = {cube.AppendChild('_'.join(sorted(filter(None, (x, y, z))))) for x in ('', 'pX', 'nX')
              for y in ('', 'pY', 'nY') for z in ('', 'pZ', 'nZ') if x or y or z}
    bodies = dict(zip(*parsed.get(kinds.RigidBody, ([], [])), strict=True))
    articulated = {p for p in robots[0].articulatedBodies if p}
    if {p for p in bodies if p.HasPrefix(robot)} != articulated:
        raise ValueError('Mounted robot has disconnected rigid links')
    cube_bodies = {p for p in bodies if p == core or p.HasPrefix(cube)}
    if cube_bodies != pieces | {core} or any(not bodies[p].isValid or not bodies[p].rigidBodyEnabled
                                             or bodies[p].kinematicBody for p in cube_bodies):
        raise ValueError('Expected a dynamic Core and all 26 dynamic cubelets in the mounted physics scene')
    return dict(articulation_path=str(anchor), robot_joints=len(expected_joints), cube_bodies=len(cube_bodies))
