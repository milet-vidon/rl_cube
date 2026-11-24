"""Offline regressions for the isolated R70 touches, drive attribution and source transfer."""
import unittest
from unittest.mock import Mock

import torch

import natural_face90_v6_long as old
import natural_face90_v6_worker as old_worker
import natural_face90_v7_long as new
import natural_face90_v7_worker as worker
import test_natural_face90_turn_guide as fixtures
from natural_face90_turn_guide_v4 import TurnGuide


class ContactCentering(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ContactTurn()
        fixture.setUp()
        self.g = TurnGuide(fixture.c)
        self.obs = fixture.obs
        self.yes = fixture.yes
        self.obs.update(natural_ready=self.yes.clone(), other_digit_contact=~self.yes)
        # First non-recovering qualified sample for seeds 66002221 and 66002215.
        # Restore the marker direction from the old goal XY and its commanded angle.
        xy = torch.tensor([[.0299909, -.0007403], [.0299329, -.0020054]], dtype=torch.double)
        angle = torch.deg2rad(torch.tensor([-.6868992, -1.1962708], dtype=torch.double))
        x, y = xy.unbind(-1)
        self.radial = torch.stack([angle.cos()*x-angle.sin()*y, angle.sin()*x+angle.cos()*y], -1)
        self.radial /= self.radial.norm(dim=1, keepdim=True)
        self.obs['face_deg'] = torch.tensor([-.7657469, -1.2729418], dtype=torch.double)
        self.pad = torch.tensor([[.0220407, -.0084192, -.0508032],
                                 [.0242366, -.0097068, -.0509025]], dtype=torch.double)

    def plan(self, *, fresh=False):
        return self.g._path(self.yes if fresh else ~self.yes, self.yes, self.radial,
                            self.pad, self.yes, self.obs)

    def test_round70_isolated_touches_center_with_bounded_steps(self):
        goal = self.plan(fresh=True)
        evidence = self.g.control_evidence()
        self.assertTrue(evidence['pre_centering'].all())
        self.assertTrue((evidence['pre_lateral_error_m'] > .004).all())
        delta = goal - self.pad
        self.assertTrue((delta[:, :2].norm(dim=1) <= .000751).all())
        self.assertTrue((delta[:, :2].norm(dim=1) > 0).all())
        torch.testing.assert_close(goal[:, 2], self.pad[:, 2])
        self.assertFalse(self.g.drive_contact.any())
        self.obs['index_contact'] = self.obs['right_contact'] = ~self.yes
        self.pad[:, 2] -= .0005  # just outside the original 6 mm normal band
        for _ in range(6):
            reacquire = self.plan()
            self.assertTrue((reacquire[:, 2] > -.055).all())
            self.assertTrue((self.g.streak == 0).all())
            self.assertFalse(self.g.drive_contact.any())
        self.assertTrue((self.plan()[:, 2] <= -.075).all())

class SourceContract(unittest.TestCase):
    def test_frozen_compare_and_train_can_use_the_same_v3_checkpoint(self):
        origin = dict(iteration=70, schema=old.SCHEMA, checkpoint='/example/R70.pt',
                      sha='a'*64, software={'old.py': 'b'*64})
        contract = dict(schema=new.SCHEMA, action_mode=worker.ACTION_MODE, software={'v4.py': 'c'*64})
        ck = dict(iteration=70, policy={}, optimizer={'old_moment': 1},
                  contract=dict(contract, schema=old.SCHEMA, software=origin['software'],
                                action_mode=old_worker.ACTION_MODE))
        model, optimizer = Mock(), Mock()
        for role in ('validate', 'train'):
            learning = role == 'train'
            req = dict(iteration=70+int(learning), role=role, learn=learning,
                       paired_validation=not learning, checkpoint=origin['checkpoint'],
                       checkpoint_sha=origin['sha'], transfer=origin)
            worker.restore_expert(model, optimizer, ck, contract, req)
            optimizer.load_state_dict.assert_not_called()
        self.assertEqual(model.load_state_dict.call_count, 2)
        with self.assertRaises(ValueError):
            worker.restore_expert(model, optimizer, ck, contract, dict(req, checkpoint_sha='0'*64))


if __name__ == '__main__':
    unittest.main()
