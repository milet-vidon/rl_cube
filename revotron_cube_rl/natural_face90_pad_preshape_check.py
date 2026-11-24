"""Static finger-preshape and wrist-direction check at the first learned control interval."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from kinematics import RobotKinematics
from natural_face90_first_touch_audit import DEFAULT, checked_frames, checked_json
from natural_face90_pad_geometry import from_summary
from natural_face90_turn_guide import TurnGuide, leading_axis

# Prespecified exploratory offsets from measured finger joints (degrees).
# Each digit group changes its flexion joints by the same amount.
FLEX = ('MCP', 'PIP', 'DIP')
VARIANTS = {
    'measured': {},
    **{f'index_{d:+d}': {'index': d} for d in (-9, -6, -3, 3, 6, 9)},
    **{f'middle_{d:+d}': {'middle': d} for d in (3, 6, 9)},
    **{f'middle_thumb_{d:+d}': {'middle': d, 'thumb': d} for d in (3, 6, 9)},
}


def handoffs(summary_path):
    summary = json.loads(summary_path.read_bytes())
    path = Path(summary['candidate_report'])
    report = checked_json(path, summary['candidate_report_sha256'])
    plan = summary['plan']
    sweep = checked_json(Path(plan['sweep_dir'])/'contact_point_summary.json', plan['sweep_summary_sha256'])
    seeds = report['seeds']
    first = {}
    for row in checked_frames(path.parent/'turn_contact_evidence.jsonl', report['expert_learning']['turn_evidence_sha256'], seeds):
        first.setdefault(row['seed'], row['step'])
    snapshots = {}
    for row in checked_frames(path.parent/'controller_states.jsonl', report['controller_evidence_sha256'], seeds):
        if row.get('learned_policy_active') and row['event'] == 'periodic' and row['step'] == first.get(row['seed']):
            if row['seed'] in snapshots:
                raise ValueError('Duplicate learned handoff snapshot')
            snapshots[row['seed']] = row
    expected = {r['seed'] for r in summary['candidate']['per_seed'] if r['learned_frames']}
    if set(snapshots) != set(first) or set(first) != expected:
        raise ValueError('Missing measured first learned interval or its post-step joint snapshot')
    return [snapshots[s] for s in seeds if s in snapshots], sweep['fixed_C_no_contact_horizon_seeds']


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def unit(v):
    return v / np.linalg.norm(v)


class ShapeCheck:
    def __init__(self, geometry):
        self.fk = RobotKinematics()
        self.anchor = np.array(geometry['anchor_dip_m'])
        self.hard = np.asarray(geometry['hard_vertices_dip_m']) - self.anchor
        self.rubber = np.asarray(geometry['rubber_vertices_dip_m']) - self.anchor
        self.outward = np.asarray(geometry['outward_axis_dip'])
        # Read exactly the frozen planner's landmark names, offsets, radii and
        # direction samples, without creating an Isaac articulation.
        names = list(self.fk.fk({}))
        robot = SimpleNamespace(shared_metatype=SimpleNamespace(link_names=names))
        reference = torch.zeros(1, 7, dtype=torch.float64)
        guide = TurnGuide(SimpleNamespace(rp=SimpleNamespace(ix=list(range(7))), e=SimpleNamespace(robot=robot),
                                          right_ref=reference, route_rotation=torch.eye(3)[None]))
        self.names = [names[i] for i in guide.links]
        self.offsets = guide.offsets.numpy()
        self.radii = guide.radii
        self.samples = guide.samples

    def joint_pose(self, original, variant):
        q = original.copy()
        for finger, degrees in variant.items():
            for joint in FLEX:
                name = f'right_{finger}_{joint}_joint'
                value = original[name] + np.deg2rad(degrees)
                lo, hi = self.fk.limits[name]
                if not lo <= value <= hi:
                    return None
                q[name] = value
        return self.fk.fk(q)

    def evaluate(self, model, inward):
        dip = model['right_index_DIP_link']
        palm = model['right_palm_link'][:3, :3]
        tip = dip[:3, :3] @ self.anchor + dip[:3, 3]
        passive = np.array([model[n][:3, :3] @ off + model[n][:3, 3] for n, off in zip(self.names, self.offsets)])
        vectors = (tip - passive) @ palm
        protrusion = unit((tip - dip[:3, 3]) @ palm)
        base_palm = leading_axis(torch.from_numpy(vectors)[None], self.radii, torch.from_numpy(protrusion)[None],
                                 torch.from_numpy(inward @ palm)[None], self.samples)[0].numpy()
        base = unit(base_palm @ palm.T @ dip[:3, :3])
        tangent = self.outward - (base @ self.outward)*base
        if np.linalg.norm(tangent) < 1e-8:
            ref = np.eye(3)[np.argmin(abs(base))]
            tangent = ref - (base @ ref)*base
        tangent = unit(tangent)
        perpendicular = np.cross(base, tangent)
        # Cover the entire cone, including directions off the shortest arc.
        tilts = np.deg2rad(np.arange(31))[:, None, None]
        azimuth = np.deg2rad(np.arange(0, 360, 4))[None, :, None]
        rays = np.cos(tilts)*base + np.sin(tilts)*(np.cos(azimuth)*tangent + np.sin(azimuth)*perpendicular)
        rubber = (rays @ self.rubber.T).max(axis=-1)
        lead = rubber - (rays @ self.hard.T).max(axis=-1)
        palm_rays = rays @ dip[:3, :3].T @ palm
        passive_lead = (palm_rays @ vectors.T - self.radii.numpy())
        margin = passive_lead.min(axis=-1)
        # The existing 2 mm passive lead, .45 index protrusion and 6 mm
        # contact window are combined with the experimental 0.25 mm rubber
        # lead design margin. This is neither a collision nor a success test.
        mesh_ok = (np.abs(rubber) <= .006) & (lead >= .00025) & (rays @ unit(self.anchor) >= .45)
        good = mesh_ok & (margin >= .002)
        if good.any():
            i = int(np.flatnonzero(good.any(axis=1))[0])
            j = int(np.argmax(np.where(good[i], margin[i], -np.inf)))
            return dict(tilt_deg=i, azimuth_deg=4*j, passive_lead_mm=1000*float(margin[i, j]),
                        rubber_lead_mm=1000*float(lead[i,j]))
        if mesh_ok.any():
            i, j = np.unravel_index(np.argmax(np.where(mesh_ok, margin, -np.inf)), mesh_ok.shape)
            return dict(tilt_deg=None, best_passive_lead_mm=1000*float(margin[i, j]),
                        limiting_landmark=self.names[int(passive_lead[i, j].argmin())])
        return dict(tilt_deg=None, limiting_landmark='no_pad_facing_sample')

    def check(self, row):
        measured = dict(zip(row['joint_names'], row['q'], strict=True))
        if len(measured) != 56:
            raise ValueError('Expected all 56 measured joint positions')
        bodies = dict(zip(row['body_names'], map(pose_matrix, row['body_poses']), strict=True))
        inward = bodies['world'][:3,:3].T @ Rotation.from_quat(row['core'][3:]).as_matrix()[:,2]
        check = self.joint_pose(measured, {})
        for name in ('Link7_R', 'right_index_DIP_link'):
            delta = np.linalg.inv(bodies[name]) @ bodies['world'] @ check[name]
            if np.linalg.norm(delta[:3,3]) > .0001 or Rotation.from_matrix(delta[:3,:3]).magnitude() > .001:
                raise ValueError(f'{name} FK disagrees with measured handoff for {row["seed"]}')
        result = {}
        for name, change in VARIANTS.items():
            pose = self.joint_pose(measured, change)
            result[name] = self.evaluate(pose, inward) if pose is not None else dict(tilt_deg=None, joint_limit=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rows, fixed = handoffs(args.summary)
    check = ShapeCheck(from_summary(args.summary))
    results = {r['seed']: check.check(r) for r in rows}
    counts = {}
    for variant in VARIANTS:
        usable = {s:r[variant]['tilt_deg'] for s,r in results.items() if r[variant]['tilt_deg'] is not None}
        counts[variant] = dict(all=len(usable), fixed=len(set(usable)&set(fixed)),
                               tilt_range_deg=[min(usable.values()),max(usable.values())] if usable else None)
    if args.json:
        print(json.dumps(dict(counts=counts, finger_flexion_offsets_deg=VARIANTS, per_seed=results),
                         indent=2, allow_nan=False))
        return
    print(f'Static FK at first learned post-step snapshot: {len(rows)} handoffs, {len(fixed)} fixed seeds')
    print('pose                   all    fixed     planned tilt range (deg)')
    for name, row in counts.items():
        print(f'{name:22s} {row["all"]:3d} {row["fixed"]:8d}     {row["tilt_range_deg"]}')
    print('Offsets in degrees are applied to MCP/PIP/DIP. Wrist-direction grid: 0..30 degrees by 1; azimuth by 4.')
    print('Counts represent sampled hand geometry. Arm IK, contact, support and signed rotation require a physics trial.')


if __name__ == '__main__':
    main()
