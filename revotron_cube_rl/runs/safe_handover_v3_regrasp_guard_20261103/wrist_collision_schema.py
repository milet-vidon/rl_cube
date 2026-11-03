"""Correct the two wrist MeshCollisionAPI attachments, preserving all meshes.

The imported asset authors these APIs on plain Xforms without a mesh-merge
schema. Author them on their existing Mesh children instead. No mesh points,
indices, transforms, joint limits, or filtered collision pairs are changed.
Reference: https://nvidia.github.io/simready-foundation/2026.04.1/capabilities/physics_bodies/physics_rigid_bodies/requirements/physx-collider-mesh.html
"""
from pathlib import Path
import hashlib,json
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Sdf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(prim):
    mesh=UsdGeom.Mesh(prim)
    return {'points_sha256':hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get()).tobytes()).hexdigest(),
      'indices_sha256':hashlib.sha256(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).tobytes()).hexdigest(),
      'counts_sha256':hashlib.sha256(np.asarray(mesh.GetFaceVertexCountsAttr().Get()).tobytes()).hexdigest(),
      'world_transform':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).tolist()}


def build(project):
    project=Path(project).resolve();folder=project.parent/'revotron_revo3/source/revotron_system/usd'
    original=folder/'revotron_bimanual_revo3_56dof.usd'
    output=folder/'revotron_bimanual_revo3_wrist_mesh_v1.usda'
    if output.exists():raise FileExistsError(output)
    layer=Sdf.Layer.CreateNew(str(output));layer.subLayerPaths=[original.name]
    stage=Usd.Stage.Open(layer);stage.SetDefaultPrim(stage.GetPrimAtPath('/revotron_bimanual'))
    changes=[]
    for suffix,side in [('L','left'),('R','right')]:
        path=f'/revotron_bimanual/Link7_{suffix}/collisions/{side}_connector_wrist_link/node_STL_BINARY_'
        parent=stage.GetPrimAtPath(path);mesh=stage.GetPrimAtPath(path+'/mesh')
        assert parent.GetTypeName()=='Xform' and mesh.IsA(UsdGeom.Mesh)
        assert parent.HasAPI(UsdPhysics.CollisionAPI) and parent.HasAPI(UsdPhysics.MeshCollisionAPI)
        before=geometry(mesh);approx=UsdPhysics.MeshCollisionAPI(parent).GetApproximationAttr().Get()
        values=[(a.GetName(),a.GetTypeName(),a.Get()) for a in parent.GetAttributes()
                if a.GetName().startswith(('physics:','physxCollision:')) and a.Get() is not None]
        parent.RemoveAPI(UsdPhysics.MeshCollisionAPI);parent.RemoveAPI(UsdPhysics.CollisionAPI)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approx)
        for name,typ,value in values:mesh.CreateAttribute(name,typ).Set(value)
        assert geometry(mesh)==before
        changes.append({'old_collider':path,'new_collider':str(mesh.GetPath()),'approximation':str(approx),
                        'geometry_before':before,'geometry_after':geometry(mesh),'collision_enabled':True})
    stage.GetRootLayer().Save()
    manifest={'kind':'USD_collision_schema_attachment_fix','original_asset':str(original.relative_to(project.parent)),
      'original_asset_sha256':sha(original),'override_asset':str(output.relative_to(project.parent)),
      'override_asset_sha256':sha(output),'changes':changes,'mesh_geometry_unchanged':True,
      'filtered_collision_pairs_changed':False,'physics_validation_pending':True}
    (project/'wrist_collision_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'override_asset':str(output),'sha256':sha(output),'corrected_colliders':len(changes)}))


if __name__=='__main__':build(Path(__file__).resolve().parent)
