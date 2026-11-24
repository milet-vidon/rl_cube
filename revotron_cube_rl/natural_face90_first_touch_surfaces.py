"""Compare first-touch approach directions with the authored DIP and rubber collision meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from natural_face90_first_touch_audit import DEFAULT, audit, checked_json
from natural_face90_geometry_probe_surfaces import load_surfaces, mesh_distances


def inward_dip(point):
    # Two same-substep vectors expressed in both frames determine their rotation:
    # the native normal and the vector from the calibration anchor to the contact.
    def basis(normal, offset):
        length = np.linalg.norm(normal)
        distance = np.linalg.norm(offset)
        if not .99 < length < 1.01 or not distance > 1e-8:
            raise ValueError('Degenerate or invalid first-contact geometry')
        n = normal / length
        t = offset / distance
        cross = np.cross(n, t)
        sine = np.linalg.norm(cross)
        if not sine > 1e-4:
            raise ValueError('Native vectors are too close to parallel to recover orientation')
        b = cross / sine
        return np.column_stack((n, np.cross(b, n), b))

    delta_core = np.subtract(point['point_core_m'], point['pad_core_m'])
    delta_dip = np.subtract(point['point_pad_body_m'], point['anchor_pad_body_m'])
    nc, nd = (np.asarray(point[k]) for k in ('normal_core', 'normal_pad_body'))
    core_to_dip = basis(nd, delta_dip) @ basis(nc, delta_core).T
    if (not np.allclose(core_to_dip @ delta_core, delta_dip, atol=1e-8, rtol=0)
            or not np.allclose(core_to_dip @ nc, nd, atol=1e-7, rtol=0)):
        raise ValueError('Core and DIP records disagree on native contact geometry')
    return core_to_dip[:, 2]  # Inward from the nZ face is Core +Z.


def ahead_mm(vertices, anchor, direction):
    """Positive means the mesh extends beyond the calibrated anchor toward the face."""
    return {name: 1000 * float(((v-anchor) @ direction).max()) for name, v in vertices.items()}


def first_pad_facing_angle(vertices, anchor, current, candidate):
    """Sample the shortest arc to the hull direction at spacing of at most one degree."""
    angle = float(np.arccos(np.clip(current @ candidate, -1., 1.)))
    if np.pi-angle < 1e-8:
        return None  # No unique short arc for opposite directions.
    transverse = candidate - np.cos(angle)*current
    if angle > 1e-8:
        transverse /= np.linalg.norm(transverse)
    for theta in np.linspace(0., angle, 1 + int(np.ceil(np.rad2deg(angle)))):
        direction = np.cos(theta)*current + np.sin(theta)*transverse
        ahead = ahead_mm(vertices, anchor, direction)
        if abs(ahead['rubber']) <= 6. and ahead['rubber'] > ahead['body']:
            return float(np.rad2deg(theta))
    return None


def surface_analysis(shapes, samples):
    vertices = {name: np.unique(shape.faces.reshape(-1, 3), axis=0) for name, shape in shapes.items()}
    hull = ConvexHull(np.vstack(list(vertices.values())))
    anchor = np.asarray(samples[0]['anchor_pad_body_m'])
    # scipy hull planes have unit outward normals; choose the face normal along
    # which the anchor has the largest projection beyond the combined support.
    planes = hull.equations
    best = np.argmax(planes[:, :3] @ anchor + planes[:, 3])
    candidate = planes[best, :3]
    rows = []
    for sample in samples:
        if not np.allclose(anchor, sample['anchor_pad_body_m'], atol=1e-8, rtol=0):
            raise ValueError('Calibration anchor changed between samples')
        inward = inward_dip(sample)
        distances = mesh_distances(shapes, sample['point_pad_body_m'])
        rows.append(dict(seed=sample['seed'], step=sample['step'], substep=sample['substep'],
                         cubelet=sample['cubelet'], inward_dip=inward.tolist(),
                         native_point_to_authored_mesh_mm={k: 1000*v for k, v in distances.items()},
                         mesh_ahead_of_anchor_mm=ahead_mm(vertices, anchor, inward),
                         first_sampled_pad_facing_deg=first_pad_facing_angle(vertices, anchor, inward, candidate),
                         angle_to_candidate_deg=float(np.rad2deg(np.arccos(np.clip(inward @ candidate, -1, 1))))))
    return dict(candidate_dip_direction=candidate.tolist(),
                candidate_mesh_ahead_of_anchor_mm=ahead_mm(vertices, anchor, candidate),
                scope='DIP authored collision meshes only; native shared body identity and whole-hand feasibility unresolved',
                per_seed=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_bytes())
    report = checked_json(Path(summary['candidate_report']), summary['candidate_report_sha256'])
    receipt = report['domain_receipt']
    scene = Path(receipt['scene_source'])/'dynamic_training_scene.usda'
    with scene.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['scene_sha256']:
            raise ValueError('Frozen training scene differs from the candidate report')
    rows = audit(args.summary, include_native=True)['fixed']
    samples = [r['native_point'] for r in rows if 'native_point' in r]
    if not samples:
        parser.error('The fixed cohort has no first index-only contact with native evidence')
    result = surface_analysis(load_surfaces(scene), samples)
    if args.json:
        print(json.dumps(result, allow_nan=False, indent=2))
        return
    print('seed     first_step  native distance (mm)   mesh ahead of anchor (mm)   arc: pad-facing / candidate')
    print('                     rubber     DIP body    rubber           DIP body   degrees')
    for row in result['per_seed']:
        dist = row['native_point_to_authored_mesh_mm']
        ahead = row['mesh_ahead_of_anchor_mm']
        first = row['first_sampled_pad_facing_deg']
        first_label = 'n/a' if first is None else f'{first:.1f}'
        print(f'{row["seed"]} {row["step"]:10d}  {dist["rubber"]:6.2f} {dist["body"]:12.2f}'
              f'    {ahead["rubber"]:6.2f} {ahead["body"]:18.2f} {first_label:>7} / {row["angle_to_candidate_deg"]:.1f}')
    print('Geometric candidate inward unit vector in DIP frame', np.round(result['candidate_dip_direction'], 4))
    print('Candidate mesh ahead of anchor (mm)', {k: round(v, 2) for k, v in result['candidate_mesh_ahead_of_anchor_mm'].items()})
    print('Pad-facing sample: rubber leads hard DIP and its tangent plane is within 6 mm of the calibrated point.')
    print(result['scope'])


if __name__ == '__main__':
    main()
