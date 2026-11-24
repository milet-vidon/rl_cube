"""Read-only USD contact-frame lookup; safe before starting Isaac/PhysX.

The imported asset contains the same rubber-link basename under the physical
link, visuals and collisions. Calibration refers to the actual link frame, not
either geometry branch. Resolve the explicit link path and verify its body.
"""
import math

PAD_LINK = 'right_index_DIP_rubber_link'
PAD_BODY = 'right_index_DIP_link'


def contact_frame(stage, robot_root, source_link=PAD_LINK):
    from pxr import UsdGeom, UsdPhysics, Gf
    if source_link != PAD_LINK:
        raise ValueError('Unexpected calibrated contact link: '+str(source_link))
    body_path = str(robot_root).rstrip('/')+'/'+PAD_BODY
    frame_path = body_path+'/'+PAD_LINK
    body = stage.GetPrimAtPath(body_path)
    frame = stage.GetPrimAtPath(frame_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Missing calibrated contact rigid body: '+body_path)
    if not frame or not frame.IsA(UsdGeom.Xform) or frame.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Missing/invalid calibrated link frame: '+frame_path)
    parent = frame.GetParent()
    if parent != body:
        raise ValueError('Calibrated link frame is not attached to its expected rigid body')
    # Keep the full fixed transform, rather than assuming a zero offset simply
    # because the current URDF happens to have one. Never mutate the USD stage.
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(frame)*cache.GetLocalToWorldTransform(body).GetInverse()
    transform = Gf.Transform(matrix)
    if any(not math.isfinite(x) or abs(x-1.)>1e-7 for x in transform.GetScale()):
        raise ValueError('Calibrated contact frame must have unit scale')
    position = list(transform.GetTranslation())
    q = transform.GetRotation().GetQuat()
    quat = [q.GetReal(),*q.GetImaginary()]
    if not all(math.isfinite(x) for x in position+quat):
        raise ValueError('Nonfinite contact frame transform')
    return PAD_BODY, position, quat


def check_source_scene(source):
    """Resolve the SAME frame in the frozen input scene, without simulation."""
    from pxr import Usd
    path=source/'dynamic_training_scene.usda'
    stage=Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError('Cannot open frozen scene for contact-frame preflight: '+str(path))
    body,offset,quat=contact_frame(stage,'/World/Robot')
    return dict(body=body,offset=offset,quaternion_wxyz=quat)
