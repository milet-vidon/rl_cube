import unittest
from types import SimpleNamespace

import torch
from contact_hold_student_eval import StudentEvaluation


class SpyModel:
    def __call__(self, features):
        self.input = features.clone()
        return torch.full((1,28), .9)


class EvaluationActionTests(unittest.TestCase):
    def test_left_only_causal_action_and_clock(self):
        s = StudentEvaluation.__new__(StudentEvaluation)
        s.left = list(range(0,56,2))
        s.cap = torch.full((1,28), .025)
        s.model, s.phase = SpyModel(), None
        env = SimpleNamespace(scales=torch.full((1,56), .05), targets=torch.zeros(1,56))
        row = dict(q=[.1]*56, joint_velocity=[0.]*56, applied_target=[0.]*56,
                   cube_position=[0.,0.,1.], cube_quaternion_wxyz=[1.,0.,0.,0.], finger_contact_points=[])
        teacher = torch.linspace(-1.,1.,56).reshape(1,56)
        a, record = s.action(env, 'ready', 1, row, teacher)
        torch.testing.assert_close(a[:,1::2],teacher[:,1::2],rtol=0,atol=0)
        torch.testing.assert_close(a[:,::2],torch.full((1,28),.5))
        self.assertEqual(record['phase_elapsed'],0.)
        self.assertEqual(s.model.input.shape,(1,198))
        teacher[:,::2] = -10.
        b, record = s.action(env,'ready',2,row,teacher)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertAlmostEqual(record['phase_elapsed'],1/300.)
        _, record = s.action(env,'preshape',3,row,teacher)
        self.assertEqual(record['phase_elapsed'],0.)


if __name__ == '__main__':
    unittest.main()
