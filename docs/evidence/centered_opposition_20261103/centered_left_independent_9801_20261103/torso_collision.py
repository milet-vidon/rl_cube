"""Export fixed USD torso colliders missing from the simplified arm URDF."""
import numpy as np

def export_fixed_colliders(stage,root_path):
 from pxr import Usd,UsdGeom,UsdPhysics
 from scipy.spatial import ConvexHull
 root=stage.GetPrimAtPath(root_path);cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(root).GetInverse();items=[];seen=set()
 for prim in Usd.PrimRange(root):
  if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
  body=prim
  while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
  if not body or body.GetName()!='world':continue
  for mesh in ([prim]if prim.IsA(UsdGeom.Mesh)else [m for m in Usd.PrimRange(prim)if m.IsA(UsdGeom.Mesh)]):
   if str(mesh.GetPath())in seen:continue
   seen.add(str(mesh.GetPath()));transform=cache.GetLocalToWorldTransform(mesh)*inverse;points=np.array([transform.Transform(p)for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]);hull=ConvexHull(points);points=points[hull.vertices]
   items.append({'prim':str(mesh.GetPath()),'points_root_m':points.tolist()})
 if not any('/waist_pitch_Link/'in item['prim']for item in items):raise ValueError('Actual USD torso collider missing')
 return items

def collision_objects(items):
 import coal
 from scipy.spatial import ConvexHull
 result=[]
 for item in items:
  points=np.array(item['points_root_m']);vertices=coal.StdVec_Vec3s()
  for p in points:vertices.append(p)
  triangles=coal.StdVec_Triangle()
  for a,b,c in ConvexHull(points).simplices:triangles.append(coal.Triangle(int(a),int(b),int(c)))
  mesh=coal.BVHModelOBBRSS();mesh.beginModel(len(triangles),len(points));mesh.addSubModel(vertices,triangles);mesh.endModel();mesh.buildConvexRepresentation(True);center=points.mean(0)
  result.append(('usd_torso:'+item['prim'],mesh.convex,np.eye(4),center,float(np.linalg.norm(points-center,axis=1).max())))
  # The runtime uses cooked mesh colliders. Keep a conservative waist envelope
  # because a measured PhysX Link5/world collision escaped the raw mesh hull.
  if '/waist_pitch_Link/'in item['prim']:
   low=points.min(0);high=points.max(0);pose=np.eye(4);pose[:3,3]=(low+high)/2
   result.append(('usd_torso:waist_envelope',coal.Box(*(high-low)),pose,np.zeros(3),float(np.linalg.norm((high-low)/2))))
 return result
