"""Compare recorded DIP-local contact points with the two authored collision meshes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class Surface:
    def __init__(self, vertices, faces):
        self.faces = vertices[faces]
        centers = self.faces.mean(axis=1)
        self.tree = cKDTree(centers)
        self.radius = np.linalg.norm(self.faces - centers[:, None], axis=-1).max(axis=1)
        self.largest = self.radius.max()

    def distance(self, point):
        # Every triangle lies inside its centroid ball. Starting with a real
        # vertex gives a valid upper bound for the closest surface point.
        _, near = self.tree.query(point)
        bound = np.linalg.norm(self.faces[near] - point, axis=1).min()
        candidates = self.tree.query_ball_point(point, bound + self.largest + 1e-12)
        candidates = np.asarray(candidates, dtype=np.intp)
        lower = np.linalg.norm(self.tree.data[candidates] - point, axis=1) - self.radius[candidates]
        triangles = self.faces[candidates[lower <= bound + 1e-12]]
        edges = np.roll(triangles, -1, axis=1) - triangles
        delta = point - triangles
        lengths_squared = (edges * edges).sum(-1)
        frac = np.clip((edges * delta).sum(-1) / np.maximum(lengths_squared, 1e-30), 0, 1)
        to_edges = delta - frac[:, :, None] * edges
        min_sq = (to_edges * to_edges).sum(-1).min(axis=1)
        normal = np.cross(edges[:, 0], -edges[:, 2])
        area_squared = (normal * normal).sum(-1)
        inside = ((np.cross(edges, delta) * normal[:, None]).sum(-1) >= 0).all(axis=1)
        inside &= area_squared > 1e-32
        min_sq[inside] = (delta[inside, 0] * normal[inside]).sum(-1)**2 / area_squared[inside]
        return float(np.sqrt(min_sq.min()))


def load_surfaces(scene):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(scene))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Expected a meter-unit frozen training scene')
    body = stage.GetPrimAtPath('/World/Robot/right_index_DIP_link')
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Missing calibrated right index rigid body')
    cache = UsdGeom.XformCache()
    inv = cache.GetLocalToWorldTransform(body).GetInverse()
    shapes = {}
    for label, link in [('rubber', 'right_index_DIP_rubber_link'), ('body', 'right_index_DIP_link')]:
        root = stage.GetPrimAtPath(f'{body.GetPath()}/collisions/{link}')
        if not root:
            raise ValueError('Missing collision branch: ' + link)
        colliders = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1 or colliders[0].GetAttribute('physics:approximation').Get() != 'convexDecomposition':
            raise ValueError('Unexpected collision approximation: ' + link)
        meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(colliders[0]) if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1 or set(meshes[0].GetFaceVertexCountsAttr().Get()) != {3}:
            raise ValueError('Expected a triangulated collision source: ' + link)
        mesh = meshes[0]
        transform = cache.GetLocalToWorldTransform(mesh.GetPrim()) * inv
        vertices = np.array([transform.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        shapes[label] = Surface(vertices, faces)
    return shapes


def mesh_distances(shapes, point_pad_body_m):
    point = np.asarray(point_pad_body_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected one finite DIP-local position in meters')
    return {name: surface.distance(point) for name, surface in shapes.items()}


def summarize(scene, contact_file):
    shapes = load_surfaces(scene)
    groups = {}
    with contact_file.open() as stream:
        for line in stream:
            row = json.loads(line)
            distances = mesh_distances(shapes, row['point_pad_body_m'])
            label = min(distances, key=distances.get)
            group = groups.setdefault(row['seed'], dict(contacts=0, nearer=Counter(), first=None))
            group['contacts'] += 1
            group['nearer'][label] += 1
            first = group['first']
            if first is None or (row['step'], row['substep'], -row['impulse_Ns']) < (
                    first['step'], first['substep'], -first['impulse_Ns']):
                group['first'] = dict(step=row['step'], substep=row['substep'], cubelet=row['cubelet'],
                                      impulse_Ns=row['impulse_Ns'], distance_m=distances,
                                      anchor_distance_m=mesh_distances(shapes, row['anchor_pad_body_m']))
    return dict(scene=str(scene), source='exact point-to-authored-triangle distances',
                physical_shape_identity='unresolved: native contact identifies the shared rigid body; '
                                        'PhysX cooks convex decomposition from both meshes',
                seeds=groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True, help='Frozen dynamic_training_scene.usda')
    parser.add_argument('--contacts', type=Path, required=True, help='native_contact_geometry.jsonl')
    opts = parser.parse_args()
    print(json.dumps(summarize(opts.scene, opts.contacts), indent=2, allow_nan=False))
