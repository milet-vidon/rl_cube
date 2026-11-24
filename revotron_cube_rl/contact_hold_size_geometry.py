"""Build a separate, fixed-mass scaled version of the pinned 27-body cube USD.

No Isaac imports or stage writes are needed to build or verify the file. All
26 hulls use points already expressed relative to the shared cube center; their
body origins and the 26 pairs of joint anchors must remain at zero. A scale on
the parent Xform is deliberately avoided because PhysX consumes the bodies
independently.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'isaac_mujoco_cube/cube_model.usda'
SOURCE_SHA256 = 'c41baa3cd75db5a497c5878d0e18bb6b3316124e0f1935867de8305b44e4eb11'
BASELINE = ROOT / 'runs/native_v20_raw_replay_off_actual_v3_20261104/passive_cube_baseline.json'
BASELINE_SHA256 = '0bc715f5f009a9dcf6b90c7d4f5f97dc22043354aebe403aa5a2ad94ed9615c8'
BASE_EDGE = .057
TEXTURE = 'upstream/mujoco_cube-main/assets/sticker.png'
POINTS = re.compile(r'^(\s*point3f\[\] points = )(\[[^\n]+\])$', re.M)
CUBELET = re.compile(r'^        def Xform "(\w+)" \(', re.M)
MASS = re.compile(r'^(\s*float physics:mass = )([\d.eE+-]+)$', re.M)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _require(ok, msg):
    if not ok:
        raise ValueError(msg)


def _baseline():
    raw = BASELINE.read_bytes()
    _require(_hash(raw) == BASELINE_SHA256, 'Wrong reference mass measurement')
    b = json.loads(raw)
    _require(b['status'] == 'rigid_baseline_measured' and len(b['rigid_baseline']) == 27
             and SOURCE_SHA256 in b['matching_fingerprint']['source_sha256'].values(),
             'Invalid ordinary rigid-body baseline')
    return b['rigid_baseline']


def _inspect(src):
    names = CUBELET.findall(src)
    points = [ast.literal_eval(m[2]) for m in POINTS.finditer(src)]
    joints = re.findall(r'def Physics(Revolute|Spherical)Joint "(\w+)"', src)
    _require(len(points) == len(names) == len(set(names)) == 26 and
             set(names) == {name for _,name in joints} and len(joints) == 26,
             'Unexpected joint or cubelet coverage')
    _require([t for t,_ in joints].count('Revolute') == 6 and
             src.count('rel physics:body0 = </World/Core>') == 26 and
             sorted(re.findall(r'rel physics:body1 = </World/Cube/(\w+)>', src)) == sorted(names),
             'Unexpected joint topology')
    _require(src.count('physics:localPos0 = (0, 0, 0)') == 26 and
             src.count('physics:localPos1 = (0, 0, 0)') == 26 and
             src.count('xformOp:translate = (0, 0, 0)') == 27,
             'Asset does not share one rotation center across all joints/bodies')
    _require(src.count('physics:approximation = "convexHull"') == 26 and
             src.count('"PhysicsCollisionAPI"') == 26 and
             'xformOp:scale' not in src and 'extent =' not in src,
             'Unexpected collider schema/transform')
    for name,pts in zip(names, points):
        _require(len(pts) >= 4 and all(len(p) == 3 and all(math.isfinite(x) for x in p) for p in pts),
                 f'Invalid points for {name}')
    masses = [float(m[2]) for m in MASS.finditer(src)]
    _require(len(masses) == 27 and all(x > 0 for x in masses), 'Missing explicit masses')
    return names, points, masses


def _render(source, edge_m):
    _require(isinstance(edge_m, (float, int)) and math.isfinite(edge_m) and .04 <= edge_m <= .12,
             'Expected cube edge in meters (0.04 .. 0.12)')
    source = Path(source).resolve()
    raw = source.read_bytes()
    _require(_hash(raw) == SOURCE_SHA256, 'Wrong source USD; this generator only supports the audited cube')
    src = raw.decode()
    names, old, masses = _inspect(src)
    k = edge_m / BASE_EDGE
    def fmt(v):
        return format(v, '.12g')
    def scale(match):
        pts = ast.literal_eval(match[2])
        return match[1] + '[' + ', '.join('(' + ', '.join(fmt(v*k) for v in p) + ')' for p in pts) + ']'
    result = POINTS.sub(scale, src)
    # The core has no collision shape, so it has no geometry from which PhysX
    # could recalculate inertia. Scale its measured baseline tensor explicitly.
    # Each cubelet's COM and inertia are computed from its new hull and the
    # original authored mass. Dimensions of the collision skin stay in meters.
    ref = _baseline()
    ci = ref['/World/Core']['inertia']
    _require(ci[0] == ci[4] == ci[8] and all(ci[i] == 0 for i in (1,2,3,5,6,7)), 'Anisotropic Core')
    c = fmt(k*k*ci[0])
    core_mass = MASS.search(result)
    _require('def Xform "Core"' in result[:core_mass.start()] and 'def Xform "Cube"' not in result[:core_mass.start()], 'Core mass order changed')
    result = result[:core_mass.end()] + '\n        float3 physics:diagonalInertia = ('+ c + ', ' + c + ', ' + c + ')' + result[core_mass.end():]
    tex = source.parent / TEXTURE
    _require(src.count('@'+TEXTURE+'@') == 1 and tex.is_file() and '@' not in str(tex), 'Missing or unsupported sticker texture')
    result = result.replace('@'+TEXTURE+'@', '@'+str(tex)+'@')
    new_names,new,new_masses = _inspect(result)
    _require(new_names == names and new_masses == masses, 'Topology or mass drift')
    lo = [min(p[i] for cloud in new for p in cloud) for i in range(3)]
    hi = [max(p[i] for cloud in new for p in cloud) for i in range(3)]
    _require(all(abs(a+edge_m/2)<1e-9 and abs(b-edge_m/2)<1e-9 for a,b in zip(lo,hi)), 'Outer hull bounds do not equal requested edge')
    for name,cloud,prev in zip(names,new,old):
        _require(all(len(x)==len(y) and all(abs(b-a*k) < 1e-11 for a,b in zip(x,y)) for x,y in zip(prev,cloud)), f'Unscaled vertex {name}')
    report = dict(schema='fixed_mass_cube_mesh_resize_v1', source_path=str(source), source_sha256=SOURCE_SHA256,
                  baseline_mass_properties_sha256=BASELINE_SHA256, source_edge_m=BASE_EDGE, edge_m=edge_m,
                  half_extent_m=edge_m/2, scale=k, bodies=27, joints=26, revolute=6, spherical=20,
                  collision_meshes=26,vertices=sum(map(len,new)), local_bounds_m=[lo,hi], total_mass_kg=sum(masses),
                  mass_policy='preserve all 27 authored masses; scale collision vertices relative to the common origin; fixed material and offsets',
                  central_inertia_kg_m2=float(c), cubelet_inertia_policy='PhysX computes from new convex hull and fixed mass',
                  center_inertia_policy='shape-less Core: measured baseline isotropic inertia times scale squared',
                  asset_sha256=_hash(result.encode()))
    return result.encode(), report


def prepare_asset(source_path, out_path, edge_m=0.090):
    """Write a new isolated cube_model.usda; refuses to replace any existing file."""
    source, dest = Path(source_path).resolve(), Path(out_path).resolve()
    _require(dest != source and dest.name == 'cube_model.usda', 'Choose a separate .../cube_model.usda destination')
    raw, report = _render(source,edge_m)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:
        f.write(raw)
    return report


def verify_asset(out_path, receipt):
    """Check the exact bytes and reproducibility from the pinned input."""
    raw, expected = _render(receipt['source_path'],receipt['edge_m'])
    _require(expected == receipt and Path(out_path).read_bytes() == raw,'Size asset or receipt mismatch')
    return expected


def verify_runtime_mass_properties(env, edge_m=.090):
    """Read actual initialized PhysX mass, principal inertia & COM; never writes state."""
    import numpy as np
    from isaac_compat import tensor
    scale = edge_m / BASE_EDGE
    ref = _baseline()
    measured={}
    for name,obj in env.cube.objects.items():
        key='/World/Core' if name=='Core' else '/World/Cube/'+name
        expected=ref[key]
        def arr(value):
            v=tensor(value)
            return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v)
        view=obj.root_view
        mass=float(arr(view.get_masses()).reshape(()))
        inertia=arr(view.get_inertias()).reshape(3,3)
        com=arr(view.get_coms()).reshape(7)
        # For isotropic cubelets different equivalent principal-axis frames can
        # result from convex cooking. Compare all three eigenvalues and COM xyz.
        want=np.linalg.eigvalsh(np.array(expected['inertia']).reshape(3,3))*scale**2
        actual=np.linalg.eigvalsh(inertia)
        _require(np.isclose(mass,expected['mass'],rtol=2e-5,atol=0)
                 and np.allclose(com[:3],np.array(expected['local_com'][:3])*scale,rtol=1e-4,atol=2e-6)
                 and np.allclose(actual,want,rtol=2e-3,atol=0), f'PhysX scaled mass/inertia/COM mismatch: {key} mass={mass} COM={com[:3]} inertia={actual} expected={want}')
        measured[key]=dict(mass_kg=mass,local_com_m=com[:3].tolist(),principal_inertia_kg_m2=actual.tolist())
    _require(set(measured)==set(ref), 'Missing measured PhysX bodies')
    return dict(verified=True,edge_m=edge_m,baseline_sha256=BASELINE_SHA256,body_properties=measured,
                total_mass_kg=sum(x['mass_kg'] for x in measured.values()))
