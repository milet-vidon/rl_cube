"""Emit the two measured-frame DIP collision hulls for an unloaded orientation probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from natural_face90_first_touch_audit import DEFAULT, audit, checked_json
from natural_face90_geometry_probe_surfaces import load_surfaces


def geometry_payload(shapes, anchor_dip_m):
    """Hull support equals full-mesh support in every direction. Input and output are meters."""
    anchor = np.asarray(anchor_dip_m, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError('Invalid measured DIP-frame calibration point')
    hulls, authored = {}, []
    for name in ('rubber', 'body'):
        vertices = np.unique(shapes[name].faces.reshape(-1, 3), axis=0)
        authored.append(vertices)
        hulls[name] = vertices[ConvexHull(vertices).vertices]

    # Score the outward facet normals by the distance the calibrated patch projects
    # beyond the combined meshes. Unit normals point from the finger into free space.
    # Use all original vertices (the identical ordering used by the surface audit)
    # so triangulation ties across coplanar faces cannot change which facet wins.
    equations = ConvexHull(np.vstack(authored)).equations
    best = np.argmax(equations[:, :3] @ anchor + equations[:, 3])
    return dict(hard_vertices_dip_m=hulls['body'].tolist(), rubber_vertices_dip_m=hulls['rubber'].tolist(),
                anchor_dip_m=anchor.tolist(), outward_axis_dip=equations[best, :3].tolist())


def from_summary(path):
    summary = json.loads(path.read_bytes())
    report = checked_json(Path(summary['candidate_report']), summary['candidate_report_sha256'])
    receipt = report['domain_receipt']
    scene = Path(receipt['scene_source']) / 'dynamic_training_scene.usda'
    with scene.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['scene_sha256']:
            raise ValueError('Frozen training scene differs from the approach-anchor report')
    rows = audit(path, include_native=True)['fixed']
    anchors = [row['native_point']['anchor_pad_body_m'] for row in rows if 'native_point' in row]
    if not anchors or not np.allclose(anchors, anchors[0], atol=1e-9, rtol=0):
        raise ValueError('Expected a common measured DIP-frame calibration point across first contacts')
    return geometry_payload(load_surfaces(scene), anchors[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT)
    print(json.dumps(from_summary(parser.parse_args().summary), allow_nan=False, separators=(',', ':')))
