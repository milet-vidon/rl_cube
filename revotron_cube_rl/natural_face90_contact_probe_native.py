"""Passive substep geometry of the existing PhysX DIP/nZ contact channel."""
import hashlib
import json
from pathlib import Path

import torch

from natural_face90_recipe_frames import contact_frame
from natural_face90_turn_guide import _point
from natural_face90_v5_metrics import conj, mul, rotate, unit, wrap


def local_point(pose, point):
    return rotate(conj(unit(pose[..., 3:])), point - pose[..., :3])


def substep_marker(core, target_poses, initial_relative_q, radial):
    """Compute v4's measured median nZ layer angle at this substep, without advancing its observer."""
    relative = mul(conj(unit(core[:, None, 3:])), unit(target_poses[:, :, 3:]))
    dq = mul(relative, conj(initial_relative_q))
    dq = torch.where(dq[..., 3:] < 0, -dq, dq)
    angles = wrap(torch.rad2deg(2 * torch.atan2(-dq[..., 2], dq[..., 3])))
    angles = angles[:, :1] + wrap(angles - angles[:, :1])
    face = angles.median(1).values
    return face, _point(radial, face, torch.zeros_like(face))


class NativeGeometry:
    """Read positive native points for every traced learning frame, including late recontacts."""

    def __init__(self, phase, trace_seeds, out):
        self.phase = phase
        self.base = phase.index_tracker
        self.raw = self.base.raw
        self.seeds = list(phase.req['seeds'])
        self.selected = list(trace_seeds)
        if (not self.selected or len(set(self.selected)) != len(self.selected)
                or not set(self.selected).issubset(self.seeds) or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('Trace seeds must correspond to distinct lanes of the original validation')
        self.device = self.raw.lanes.device
        self.lanes = torch.tensor([self.seeds.index(s) for s in self.selected], device=self.device)
        self.target_names = [name for name in phase.monitor.names if 'nZ' in name.split('_')]
        self.name_ids = {name: i for i, name in enumerate(self.target_names)}
        paths = list(self.raw.v.sensor_paths)
        self.sensors = self.base.sensors[torch.isin(self.raw.lanes[self.base.sensors], self.lanes)]
        if len(self.sensors) != 9 * len(self.lanes):
            raise ValueError('Expected nine target cubelets per selected lane')
        self.sensor_piece = torch.zeros_like(self.raw.lanes)
        self.sensor_piece[self.sensors] = torch.tensor(
            [self.name_ids[paths[i].rsplit('/', 1)[-1]] for i in self.sensors.tolist()], device=self.device)
        _, pos, wxyz = contact_frame(phase.e.stage, '/World/envs/env_0000/Robot')
        self.rubber_in_dip = self.base.pad_offset.new_tensor([*pos, *wxyz[1:], wxyz[0]])
        self.anchor_rubber = local_point(self.rubber_in_dip, self.base.pad_offset).tolist()
        self.path = Path(out) / 'native_contact_geometry.jsonl'
        self.log = self.path.open('x', buffering=1)
        self.traced_frames = dict.fromkeys(self.selected, 0)
        self.contact_frames = dict.fromkeys(self.selected, 0)
        self.points = 0
        self.step = -1
        self._watch = False
        self.enabled = torch.zeros_like(self.base.index_impulse, dtype=torch.bool)
        self.previous = torch.zeros_like(self.base.index_impulse)
        self.total = self.previous.clone()

    def begin_step(self, step, active):
        if step != self.step + 1:
            raise ValueError('Missing or repeated diagnostic control step')
        self.step = step
        self.enabled.zero_()
        self.enabled[self.lanes] = active[self.lanes]
        self._watch = bool(self.enabled.any())

    def clear(self, scene):
        self.base.clear(scene)
        self.previous.zero_()
        self.total.zero_()

    def update(self, scene):
        self.base.update(scene)
        if not self._watch:
            return
        cumulative = self.base.index_impulse
        changed = self.enabled & (cumulative > self.previous)
        if not changed.any():
            return
        self.previous.copy_(cumulative)
        # Tensor API RigidContactView.get_raw_contact_data: unfiltered contacts
        # are indexed by SENSOR. dt=1 returns solver impulses in N s.
        forces, points, normals, separations, counts, starts, actors = self.raw.v.get_raw_contact_data(1.)
        sensors = self.sensors[changed[self.raw.lanes[self.sensors]]]
        count = counts.flatten().long()[sensors]
        n = int(count.sum())
        parent = torch.repeat_interleave(sensors, count, output_size=n)
        prior = torch.repeat_interleave(count.cumsum(0) - count, count, output_size=n)
        idx = starts.flatten().long()[parent] + torch.arange(n, device=self.device) - prior
        if (idx < 0).any() or (idx >= forces.numel()).any():
            raise ValueError('Out-of-range native contact span')
        force = forces.reshape(-1)[idx].double()
        keep = torch.isin(actors.reshape(-1).view(torch.int64)[idx], self.base._distal_tensor) & (force.abs() > 0)
        sensor, idx, force = parent[keep], idx[keep], force[keep]
        if len(idx) == 0:
            raise ValueError('Distal impulse increased without a positive native contact record')
        lane = self.raw.lanes[sensor]
        self.total.scatter_add_(0, lane, force.abs())
        # All poses and contact records belong to the SAME completed physics
        # substep, in a scene whose tensor subspace root is /.
        unique, lookup = lane.unique(return_inverse=True)
        core = scene.views['Core'].get_transforms()[unique].double()
        target = torch.stack([scene.views[name].get_transforms()[unique] for name in self.target_names], 1).double()
        layer = self.phase.monitor.layer
        face, marker = substep_marker(core, target, layer.q0[unique][:, layer.select],
                                      self.phase.controller.radial[unique])
        cubelet_id = self.sensor_piece[sensor]
        piece = target[lookup, cubelet_id]
        core = core[lookup]
        dip = scene.robot.get_link_transforms()[lane, self.base.pad_body].double()
        pad_world = dip[:, :3] + rotate(unit(dip[:, 3:]), self.base.pad_offset)
        p = points.reshape(-1, 3)[idx].double()
        normal = normals.reshape(-1, 3)[idx].double()
        sep = separations.reshape(-1)[idx].double()
        p_core = local_point(core, p)
        pad_core = local_point(core, pad_world)
        marker = marker[lookup]
        p_dip = local_point(dip, p)
        norm_core = rotate(conj(unit(core[:, 3:])), normal)
        norm_dip = rotate(conj(unit(dip[:, 3:])), normal)
        length = norm_core.norm(dim=1)
        if (length < .99).any() or (length > 1.01).any():
            raise ValueError('Non-unit native contact normal')
        # Preserve the native normal's sign. Use its unsigned inclination to
        # compare surface planes; it is not an outward rubber-surface normal.
        vectors = dict(point_world_m=p, normal_world=normal, point_core_m=p_core, normal_core=norm_core,
                       marker_core_m=marker, pad_core_m=pad_core, point_cubelet_m=local_point(piece, p),
                       normal_cubelet=rotate(conj(unit(piece[:, 3:])), normal), point_pad_body_m=p_dip,
                       normal_pad_body=norm_dip, point_rubber_m=local_point(self.rubber_in_dip, p_dip),
                       normal_rubber=rotate(conj(unit(self.rubber_in_dip[3:])), norm_dip))
        scalars = dict(impulse_Ns=force.abs(), signed_native_scalar_Ns=force, separation_m=sep,
                       face_deg=face[lookup], normal_nz_abs_cos=norm_core[:, 2].abs()/length,
                       contact_marker_lateral_m=(p_core[:, :2]-marker[:, :2]).norm(dim=1),
                       contact_face_plane_abs_m=(p_core[:, 2]-marker[:, 2]).abs(),
                       anchor_contact_distance_m=(p_core-pad_core).norm(dim=1))
        packet = torch.cat([v for v in vectors.values()] + [v[:, None] for v in scalars.values()], 1)
        if not packet.isfinite().all():
            raise ValueError('Nonfinite physical contact geometry')
        anchor = self.base.pad_offset.tolist()
        for i, cube, values in zip(lane.tolist(), cubelet_id.tolist(), packet.tolist()):
            row = dict(step=self.step, substep=self.base.substeps, lane=i, seed=self.seeds[i],
                       cubelet=self.target_names[cube], anchor_pad_body_m=anchor, anchor_rubber_m=self.anchor_rubber)
            for j, key in enumerate(vectors):
                row[key] = values[3*j:3*j+3]
            row.update(zip(scalars, values[3*len(vectors):], strict=True))
            self.log.write(json.dumps(row, allow_nan=False) + '\n')
        self.points += len(idx)

    def validate(self):
        self.base.validate()
        if self._watch:
            torch.testing.assert_close(self.total[self.enabled], self.base.index_impulse[self.enabled],
                                       rtol=1e-8, atol=1e-12)
            for lane in self.enabled.nonzero(as_tuple=True)[0].tolist():
                seed = self.seeds[lane]
                self.traced_frames[seed] += 1
                self.contact_frames[seed] += int(self.total[lane] > 0)

    def close(self):
        self.log.close()
        with self.path.open('rb') as handle:
            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        return dict(schema='native_dip_contact_geometry_v1', file=self.path.name, sha256=sha,
                    selected_seeds=self.selected, points=self.points, frames_by_seed=self.contact_frames,
                    active_frames_by_seed=self.traced_frames)


def install_probe(phase, trace_seeds, out):
    if phase.e.reducer is not phase.index_tracker:
        raise ValueError('Probe must wrap the original v4 native index tracker')
    probe = NativeGeometry(phase, trace_seeds, out)
    phase.e.reducer = probe
    return probe
