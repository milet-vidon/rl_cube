import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

import natural_face90_pad_preshape_policy as p
import test_natural_face90_pad_orientation_policy as w
import test_natural_face90_turn_guide as fixtures


class PreshapeTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.ContactTurn()
        f.setUp()
        self.f = f
        c = p.PreshapeController(f.c.e, torch.zeros(2,6), 15)
        self.c = c
        yes = f.yes
        self.obs = dict(f.obs, natural_ready=yes, right_contact=~yes, index_contact=~yes,
                        other_digit_contact=~yes, right_total_impulse=f.q.new_zeros(2))
        self.gaps = f.q.new_full((2,), .040)
        c.live_pad_clearance = lambda: self.gaps
        self.records = []
        c.record_standby = lambda staging, values: self.records.append((staging.clone(), values.clone()))
        self.q = f.q.clone()
        self.q[:,c.fIDs] = c.fGoal
        c._q = lambda: self.q.clone()
        c._core = lambda: (f.q.new_tensor([[.45,.10,1.37]]*2),
                           torch.eye(3,dtype=f.q.dtype).expand(2,-1,-1))
        c.on[:] = True
        c.phase[:] = 1
        c.clock[:] = 45

    def test_normal_entry_limits_staging_holds_contact_and_preserves_other_fingers(self):
        c = self.c
        base = self.q.clone()
        self.obs['right_total_impulse'][1] = 1e-9
        target = base.clone()
        for _ in range(20):
            old = target.clone()
            target, _, failed, _ = c.action(old, self.f.yes, self.obs)
            self.assertFalse(failed.any())
            self.q = target.clone()
            actual = target[:,c.index_ids] - old[:,c.index_ids]
            self.assertTrue((actual.abs() <= .01100001).all())
            self.assertTrue((actual[0] <= 0.).all())
            self.assertTrue((actual[1] == 0.).all())
        torch.testing.assert_close(target[0,c.index_ids], c.index_goal[0])
        torch.testing.assert_close(target[1,c.index_ids], base[1,c.index_ids])
        self.assertEqual(c.right_ready.tolist(), [True,False])
        self.assertGreaterEqual(len(self.records), 10)
        others = [j for j in c.fIDs if j not in c.index_ids]
        torch.testing.assert_close(target[:,others], base[:,others])
        self.assertTrue((c.index_goal - (c.fGoal[c.slots] - math.radians(6))).abs().max() < 1e-9)

    def test_engaged_hand_does_not_begin_a_shape_change_and_clearance_is_required(self):
        c = self.c
        c.phase[0] = 0
        c.initial = self.q.clone()
        self.gaps[1] = .019
        target, _, _, _ = c.action(self.q, self.f.yes, self.obs)
        torch.testing.assert_close(target[:,c.index_ids],self.q[:,c.index_ids])
        self.assertFalse(c.right_ready.any())
        self.assertEqual(self.records[-1][0].tolist(), [False, True])
        self.assertEqual(self.records[-1][1][:,14].tolist(), [True, False])

    def test_real_standby_records_reconcile_and_sensor_mismatch_is_rejected(self):
        c = self.c
        req = dict(seeds=[61,62], parameters=[[0.]*6]*2, pad_preshape_offset_deg=-6)
        orientation = {k+'_by_lane': [0,0] for k in p.wrist.FIELDS}
        self.obs['right_total_impulse'][1] = 1e-8
        with TemporaryDirectory() as directory:
            trace = Path(directory) / 'turn_contact_evidence.jsonl'
            trace.write_text('')
            path = trace.with_name(p.TRACE)
            data = SimpleNamespace(req=req, monitor=SimpleNamespace(last_step=0),
                                   shape_frames=[0,0], ready_steps=[None,None])
            with path.open('w') as data.shape_trace:
                c.record_standby = lambda lanes, values: p.PadPreshapePhase.record_standby(data, lanes, values)
                target = self.q.clone()
                for step in range(25):
                    data.monitor.last_step = step
                    target, _, _, _ = c.action(target, self.f.yes, self.obs)
                    self.q = target.clone()
            receipt = dict(offset_deg=-6, joints=list(p.JOINTS), frames_by_lane=data.shape_frames,
                           ready_step_by_lane=data.ready_steps, orientation=orientation,
                           azimuth_deg_by_lane=[0,0], handoff_error_rad_by_lane=[0,0],
                           trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            p.verify_receipt(receipt, trace, req)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['right_impulse_Ns'] = 1e-8
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            receipt['trace_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'eligibility'):
                p.verify_receipt(receipt, trace, req)

    def test_sphere_uses_measured_eligibility_and_minimum_tilt(self):
        z = torch.tensor([[0.,0.,1.]]*2,dtype=torch.float64)
        dip = z.new_tensor([[0.,0.,0.,1.]]*2)
        palm = torch.eye(3,dtype=z.dtype).expand(2,-1,-1)
        geometry = p.wrist.geometry_tensors(w.geometry(), z)
        chosen, m = p.sphere_choice(z, dip, palm, .05*z[:,None], z.new_tensor([.01]),
                                    geometry, torch.tensor([True,False]))
        self.assertEqual(m['adjusted'].tolist(), [1.,0.])
        self.assertGreater(m['angle_deg'][0],0)
        self.assertLessEqual(m['angle_deg'][0],22)
        self.assertGreaterEqual(m['rubber_lead_mm'][0],.25)
        self.assertGreaterEqual(m['passive_lead_mm'][0],2.)
        self.assertLessEqual(abs(m['predicted_gap_mm'][0]),6.)
        torch.testing.assert_close(chosen[1], z[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
