from copy import deepcopy
import unittest
import numpy as np
import torch
try:import joint_safety
except ImportError:import baseline_joint_safety as joint_safety
from test_phase_gated_residual_v6 import model,frame
from train_phase_gated_residual_v6 import inspect,assert_reachable

def episode(masked_label=1e-7,active_label=.005):
    labels=np.zeros((2,28),dtype=np.float32);labels[0,0]=masked_label;labels[1,0]=active_label
    parameters={'scales':[[.05]*56],'limits':[[[-2.,2.]]*56],'step_limit':[[.04]*56],
        'command_velocity_limits':[[1.]*56],'control_dt':1/30}
    return {'x':torch.cat([frame('squeeze',71/84),frame('squeeze',72/84)],dim=1)[0].numpy(),
        'previous_left':np.zeros((2,28),dtype=np.float32),'labels':labels,'mask':np.ones(2,dtype=bool),
        'parameters':parameters,'left_indices':list(range(28)),'projector':joint_safety.project_target,
        'joint_names':['fixture_'+str(i)for i in range(56)],'executed_sources':['student','student'],'episode_id':'synthetic_host_only'}

class ReachabilityTests(unittest.TestCase):
    def test_masked_float32_floor_is_reported_and_labels_stay_unchanged(self):
        data=episode();old=data['labels'].copy();report=inspect([data],model())
        assert_reachable(report)
        self.assertGreater(report['masked_original_label_floor_max_rad'],0.)
        self.assertEqual(report['masked_original_nonzero_label_elements'],1)
        self.assertEqual(report['active_max_unreachable_distance_rad'],0.)
        self.assertTrue(report['all_labels_retained_in_loss'])
        np.testing.assert_array_equal(data['labels'],old)

    def test_real_masked_correction_is_rejected_without_clipping(self):
        data=episode(masked_label=.005);old=data['labels'].copy();report=inspect([data],model())
        with self.assertRaisesRegex(ValueError,'mask hides'):assert_reachable(report)
        np.testing.assert_array_equal(data['labels'],old)

    def test_active_unreachable_label_is_rejected_strictly(self):
        report=inspect([episode(active_label=.007)],model())
        self.assertGreater(report['active_max_unreachable_distance_rad'],0.)
        with self.assertRaisesRegex(ValueError,'strictly reachable'):assert_reachable(report)

if __name__=='__main__':unittest.main()
