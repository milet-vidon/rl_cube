"""CPU-only native-buffer attribution and coordinate regression checks."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import natural_face90_contact_probe_native as probe
from natural_face90_v5_metrics import rotate


class ProbeGeometry(unittest.TestCase):
    def test_sparse_indices_signed_normal_and_original_lane(self):
        names = ['nZ'] + [f'edge{i}_nZ' for i in range(8)]
        lanes = torch.arange(2).repeat_interleave(9)
        sensor_paths = [f'/World/envs/env_{k:04d}/Cube/{name}' for k in range(2) for name in names]
        starts = torch.zeros(18, dtype=torch.int32)
        counts = starts.clone()
        starts[9], counts[9] = 4, 2  # two bodies at a nonzero native buffer offset
        force = torch.tensor([0., 0., 0., 0., .125, .25, 0.])
        actors = torch.tensor([0, 0, 0, 0, 7, 99, 0])
        empty = torch.zeros(len(force))
        s = math.sqrt(.5)
        pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1.], [7., 2., 3., s, 0., 0., s]], dtype=torch.float64)
        world_point = torch.tensor([[.03, 0., -.045]], dtype=torch.float64)
        points = torch.zeros(len(force), 3, dtype=torch.float64)
        points[4] = pose[1, :3] + rotate(pose[1, 3:], world_point[0])
        normals = torch.zeros_like(points)
        normals[4] = rotate(pose[1, 3:], torch.tensor([0., 0., -1.], dtype=torch.float64))
        record = (force, points, normals, empty, counts, starts, actors)
        v = SimpleNamespace(sensor_paths=sensor_paths, get_raw_contact_data=lambda dt: record)
        index = torch.zeros(2, dtype=torch.float64)
        def clear(scene):
            index.zero_()
            base.substeps = 0
        def update(scene):
            base.substeps += 1
            index[1] += .125
        base = SimpleNamespace(raw=SimpleNamespace(lanes=lanes, v=v), sensors=torch.arange(18),
            index_impulse=index, pad_offset=torch.tensor([.03, 0., -.05], dtype=torch.float64),
            pad_body=0, _distal_tensor=torch.tensor([7]), clear=clear, update=update, validate=lambda: None)
        view = SimpleNamespace(get_transforms=lambda: pose)
        scene = SimpleNamespace(stage=None, reducer=base, views={name:view for name in ['Core']+names},
                                robot=SimpleNamespace(get_link_transforms=lambda: pose[:, None]))
        reference_q = torch.tensor([0., 0., 0., 1.], dtype=torch.float64).expand(2,9,4).clone()
        phase = SimpleNamespace(index_tracker=base, e=scene, req={'seeds':[123, 456]},
            monitor=SimpleNamespace(names=names, layer=SimpleNamespace(q0=reference_q, select=torch.ones(9,dtype=torch.bool))),
            controller=SimpleNamespace(radial=torch.tensor([[1., 0.]]*2, dtype=torch.float64)))
        fixed = ('right_index_DIP_link', [.003, 0., 0.], [1.,0.,0.,0.])
        with TemporaryDirectory() as path, patch.object(probe, 'contact_frame', return_value=fixed):
            p = probe.install_probe(phase, [456], path)
            p.begin_step(0, torch.tensor([False, True]))
            p.clear(scene)
            for _ in range(2):
                p.update(scene)
            p.validate()
            meta = p.close()
            rows = [json.loads(line) for line in (Path(path)/meta['file']).read_text().splitlines()]
        self.assertEqual(meta['points'], 2)
        self.assertEqual(meta['frames_by_seed'], {456:1})
        self.assertEqual([r['substep'] for r in rows], [1, 2])
        for r in rows:
            self.assertEqual((r['lane'], r['seed'], r['cubelet']), (1, 456, 'nZ'))
            self.assertAlmostEqual(r['impulse_Ns'], .125)
            self.assertAlmostEqual(r['contact_marker_lateral_m'], 0.)
            self.assertAlmostEqual(r['contact_face_plane_abs_m'], 0.)
            self.assertAlmostEqual(r['normal_core'][2], -1.)
            self.assertAlmostEqual(r['normal_nz_abs_cos'], 1.)
            self.assertAlmostEqual(r['point_rubber_m'][0], .027)
            self.assertAlmostEqual(r['anchor_contact_distance_m'], .005)

    def test_marker_tracks_nine_piece_median_at_sample_time(self):
        core = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)
        pieces = core[:,None].repeat(1,9,1)
        reference = pieces[:,:,3:].clone()
        angle = torch.deg2rad(torch.tensor([[25., 11., 11., 11., 11., 11., 11., 11., 11.]]))
        pieces[:,:,5] = (-angle/2).sin()
        pieces[:,:,6] = (angle/2).cos()
        face, marker = probe.substep_marker(core, pieces, reference, torch.tensor([[1., 0.]]))
        self.assertAlmostEqual(float(face[0]), 11., places=5)
        self.assertAlmostEqual(float(marker[0, 1]), -.03 * math.sin(math.radians(11)), places=6)


if __name__ == '__main__':
    unittest.main()
