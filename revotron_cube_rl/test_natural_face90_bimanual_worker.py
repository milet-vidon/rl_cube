"""Lane attribution, uninterrupted drive, native evidence replay and checkpoint isolation, on CPU."""
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import patch
import tempfile
from pathlib import Path

import torch

import natural_face90_bimanual_worker as worker
from natural_face90_bimanual_contacts import GripContacts
from natural_face90_bimanual_policy import BimanualExpert
from natural_face90_recipe_policy import PPO


class Packets:
    def __init__(self):
        self.cache = {}
        for lane in (0, 1):
            for digit in range(3):
                self.cache[lane*3+digit] = (lane, 1)
        self.code = torch.tensor(list(self.cache.values()))
        self.keys = torch.tensor(list(self.cache))
        self.acc = torch.zeros(2, 5, dtype=torch.double)
        self.target = torch.zeros(52, dtype=torch.bool)
        self.target[:9] = self.target[26:35] = True
        self.lanes = torch.arange(2).repeat_interleave(26)
        targets = ['pX_nZ','pY_nZ','nY_nZ','nZ','nX_nZ','pX_pY_nZ','pX_nY_nZ','nX_pY_nZ','nX_nY_nZ']
        self.sensor_paths = [f'/World/envs/env_{i:04}/Cube/{name}' for i in (0,1)
                             for name in targets+['rest']*17]
        self.v = self
        self.capacity = 8192
        self.count = 0

    def get_raw_contact_data(self, dt):
        return self.packet

    def emit(self, records):
        actors, values, counts, starts = [], [], [], []
        sub = torch.zeros_like(self.acc)
        for sensor in range(52):
            found = [(a, v) for s, a, v in records if s == sensor]
            starts.append(len(actors))
            counts.append(len(found))
            for actor, value in found:
                lane = sensor//26
                actors.append(actor)
                values.append(value)
                if self.target[sensor]:
                    sub[lane, 0] += value
        self.acc += sub
        self.count += 1
        self.packet = (torch.tensor(values, dtype=torch.double), None, None, None,
                       torch.tensor(counts), torch.tensor(starts), torch.tensor(actors, dtype=torch.long))


class BilateralContactsTests(TestCase):
    def test_two_lanes_same_substep_support_wrong_layer_and_other_digit(self):
        raw = Packets()
        original = NS(base=raw, clear=lambda e:raw.acc.zero_(), update=lambda e:None, validate=lambda:None)
        scene = NS(n=2, dec=8, home=torch.zeros(2,56), reducer=original)
        observer = GripContacts(scene, NS(r=raw))
        names = ['thumb', 'index', 'middle']
        actor_path = lambda e, ids: [f'/World/envs/env_{i//3:04}/Robot/right_{names[i%3]}_DIP_link' for i in ids]
        with patch('natural_face90_bimanual_contacts._actor_paths', side_effect=actor_path):
            observer.clear(scene)
            for sub in range(8):
                # First lane has two digits on nZ in the same step. Second lane touches with
                # the two digits across different substeps; both frame sums exceed threshold.
                rec = [(0, 0, .001), (4, 1, .002)]
                if sub == 0:
                    rec += [(26, 3, .003)]
                if sub == 1:
                    rec += [(30, 4, .004)]
                if sub == 2:
                    rec += [(26, 3, .003),(26, 4, .004)]  # same cubelet, never qualifies
                raw.emit(rec)
                raw.acc[:,3] += .04
                observer.update(scene)
        observer.validate()
        yes = torch.tensor([True, True])
        m = dict(support=yes, natural_ready=yes, structure_ok=yes, allowed_layer_state=yes, wrong_contact=~yes)
        x = observer.read(m)
        self.assertEqual(x['bilateral'].tolist(), [True, False])
        self.assertEqual(x['overlap'].tolist(), [8, 1])
        self.assertEqual(x['opposed_substeps'].tolist(), [8, 0])
        self.assertEqual(x['thumb'].tolist(), [True, True])
        self.assertEqual(x['index'].tolist(), [True, True])
        raw.acc[0,4] += .01
        self.assertFalse(observer.read(m)['bilateral'][0])
        raw.acc[0,4] = 0
        raw.acc[0,0] += .01
        self.assertFalse(observer.read(m)['bilateral'][0])


class DriveTests(TestCase):
    def test_gap_or_wrong_direction_cannot_be_counted(self):
        d = worker.GripDrive(1, 'cpu')
        yes = torch.tensor([True])
        previous = 0.
        def step(angle, contact=True, orbit=.01, safe=True):
            nonlocal previous
            before, face = torch.tensor([previous]).double(), torch.tensor([angle]).double()
            r = d.update(yes, torch.tensor([safe]), yes, torch.tensor([contact]), torch.tensor([orbit]),
                         before, face, 2, yes)
            previous = angle
            return float(r)
        self.assertEqual(step(1.), 0.)
        self.assertEqual(step(1.5), 1.)
        self.assertEqual(step(3., contact=False), 0.)
        self.assertFalse(d.reached.item())
        self.assertEqual(step(6., orbit=-.01), 0.)
        self.assertEqual(step(7.), 0.)
        self.assertEqual(step(8.5), 1.)
        self.assertEqual(step(8.6), 2.5)
        self.assertTrue(d.reached.item())
        self.assertEqual(step(20., safe=False), 0.)

    def test_trace_rebuild_rejects_false_contact_and_angle(self):
        replay = worker.LaneEvidence()
        native = dict(safe=True, structure_ok=True, allowed_layer_state=True, wrong_contact=False,
                      natural_ready=True, support=True, left_static_impulse=.32, left_target_impulse=0.,
                      target_impulse=.024, other_impulse=0., retract=False, success=False,
                      contact_goal_reached=True)
        def packet(before, face, credit, reached):
            return dict(step=int(before*2), angle_before=before, angle=face, handoff_deg=0.,
                        substeps_Ns=[[.001,.002,.04]]*8, side_masks=[[1,2]]*8, other_target_Ns=0., left_target_Ns=0.,
                        wrong_Ns=0., pre_bilateral=True, supported=True, clean=True, orbit_rad=.01,
                        retreat=False, native_goal=True, native_held=False, reached=reached,
                        held=False, credit_deg=credit, reward=0.)
        r = packet(0.,.5,0.,False)
        replay.read(r,dict(native,face_deg=.5),2)
        self.assertEqual(replay.overlap, 8)
        with self.assertRaisesRegex(ValueError,'observer'):
            replay.read(dict(r,step=1),dict(native,face_deg=1.,left_static_impulse=0.),2)
        with self.assertRaisesRegex(ValueError,'rotation'):
            replay.read(packet(.5,1.,3.,True),dict(native,face_deg=1.),2)

    def test_incompatible_policy_rejected_and_frozen_eval_keeps_optimizer(self):
        policy = BimanualExpert()
        ppo = PPO(policy)
        contract = {'schema':'grip','goals':[2,15,90]}
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)/'checkpoint.pt'
            torch.save(dict(policy=policy.state_dict(), contract=contract, iteration=2), f)
            req = dict(checkpoint=str(f),checkpoint_sha=worker.recipe.sha(f), learn=False, iteration=2, target_deg=15)
            worker.restore(policy, ppo.optimizer, contract, req)  # eval does not touch optimizer
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                worker.restore(policy, ppo.optimizer, {'schema':'single-finger'}, req)
            with self.assertRaisesRegex(ValueError,'fresh'):
                worker.restore(policy, ppo.optimizer, contract, dict(req,checkpoint=None))


if __name__ == '__main__':
    main()
