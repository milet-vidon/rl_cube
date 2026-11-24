import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from contact_hold_dataset import SCHEMA, audit_demo, sha, training_pairs
from contact_hold_policy import ContactHoldPolicy, restored


def row(step, target, phase='ready'):
    return dict(step=step, phase=phase, q=[float(step)]*56, joint_velocity=[0.]*56,
                applied_target=[target]*56, cube_position=[0., 0., 1.],
                cube_quaternion_wxyz=[1., 0., 0., 0.], finger_contact_points=[])


class DatasetTests(unittest.TestCase):
    def test_post_state_is_not_used_to_predict_its_own_action(self):
        x, y, _ = training_pairs([row(0, .1), row(1, .12)], list(range(28)))
        np.testing.assert_array_equal(x[0, :56], 0.)
        np.testing.assert_allclose(y[0], .12)
        self.assertAlmostEqual(float(x[0, -1]), 1/300.)

    def test_clock_resets_without_future_phase_length(self):
        x, _, _ = training_pairs([row(0, .1), row(1, .12, 'lower'), row(2, .13, 'lower')], list(range(28)))
        self.assertEqual(x[0, -1], 0.)
        self.assertAlmostEqual(float(x[1, -1]), 1/300.)

    def test_unsafe_target_step_rejected(self):
        with self.assertRaisesRegex(ValueError, 'maximum step'):
            training_pairs([row(0, 0), row(1, .05)], list(range(28)))

    def test_failed_teacher_never_enters_training(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run/'result.json').write_text(json.dumps(dict(full_episode=True, dynamic_grasp_passed=False, failure='tilt_hold_timeout')))
            (run/'artifact_manifest.json').write_text(json.dumps({'sha256': {'result.json': sha(run/'result.json')}}))
            with self.assertRaisesRegex(ValueError, 'accepted teacher'):
                audit_demo(run)

    def test_changed_result_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run/'result.json').write_text('{}')
            (run/'artifact_manifest.json').write_text(json.dumps({'sha256': {'result.json': '0'*64}}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                audit_demo(run)

    def test_output_limits_runtime_cap_and_reload(self):
        model = ContactHoldPolicy(np.zeros(198), np.ones(198), np.full(28, -.2), np.full(28, .2), np.zeros(28), np.ones(28), 16)
        with torch.no_grad():
            model.net[-1].bias.fill_(100.)
        x = torch.zeros(3, 198)
        self.assertTrue((model(x) <= .2).all())
        self.assertTrue((model.targets(x, torch.zeros(3, 28), torch.full((1,28), .025)) <= .025).all())
        copy = restored({'metadata': {'schema': SCHEMA, 'width': 16}, 'model': model.state_dict()})
        self.assertTrue(torch.equal(model(x), copy(x)))
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            restored({'metadata': {'schema': 'v11'}})


if __name__ == '__main__':
    unittest.main()
