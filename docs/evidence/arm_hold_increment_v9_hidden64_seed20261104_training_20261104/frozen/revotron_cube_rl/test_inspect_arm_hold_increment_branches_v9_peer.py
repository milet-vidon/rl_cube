"""Independent coverage counterexamples; synthetic labels do not qualify data."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import torch
import inspect_arm_hold_increment_branches_v9 as inspect
import arm_hold_increment_policy_v9 as policy
from test_arm_hold_increment_policy_v9_peer import model,params
from test_zero_gated_residual_policy_v7 import frame,joint_safety


def fixture(length=2):
    parameters=params();x=frame().repeat(1,length,1)[0].numpy()
    return {'x':x,'previous_left':np.zeros((length,28),np.float32),'labels':np.zeros((length,28),np.float32),
        'mask':np.ones(length,bool),'episode_id':'synthetic_not_physics','parameters':parameters,
        'left_indices':list(range(28)),'joint_names':parameters['joint_names'],'projector':joint_safety.project_target}


class IndependentV9Coverage(unittest.TestCase):
    def test_actual_j4_cap_clipped_witness_does_not_clip_label(self):
        data=fixture(1);data['previous_left'][0,3]=-2.207127332687378;data['labels'][0,3]=-2.2321274280548096
        raw=data['labels'].copy();delta=np.float32(raw[0,3]-data['previous_left'][0,3]);self.assertGreater(abs(delta),np.float32(.025))
        r=inspect.coverage(data,model())
        self.assertEqual(r['active_unreachable_elements'],0);self.assertEqual(r['active_no_float32_branch_witness_elements'],0)
        self.assertEqual(r['active_classes']['UPDATE'],1);self.assertEqual(r['per_joint']['Joint4_L']['UPDATE'],1)
        self.assertTrue(np.array_equal(raw.view(np.int32),data['labels'].view(np.int32)))

    def test_interval_containment_without_projected_bit_witness_is_reported_missing(self):
        data=fixture(1);data['labels'][0,3]=.007123
        def quantized(previous,requested,limits,step,velocity,dt):
            applied,hit=joint_safety.project_target(previous,requested,limits,step,velocity,dt)
            return torch.round(applied/.005)*.005,hit
        data['projector']=quantized
        r=inspect.coverage(data,model())
        self.assertEqual(r['active_unreachable_elements'],0)
        self.assertEqual(r['active_no_float32_branch_witness_elements'],1)
        self.assertEqual(r['per_joint']['Joint4_L']['no_float32_branch_witness_count'],1)

    def test_label_beyond_union_is_never_silently_clipped(self):
        data=fixture(1);data['labels'][0,3]=.03;before=data['labels'].tobytes()
        r=inspect.coverage(data,model());self.assertGreater(r['active_max_union_gap_rad'],0.)
        self.assertEqual(r['active_unreachable_elements'],1);self.assertEqual(r['active_no_float32_branch_witness_elements'],1)
        self.assertEqual(data['labels'].tobytes(),before)

    def test_base_hold_priority_and_ambiguous_hashes_equal_frozen_classifier(self):
        data=fixture(1);data['previous_left'][0,1:5]=.01
        data['labels'][0,1]=.01;data['labels'][0,2]=.02;data['labels'][0,3]=.0000005;data['labels'][0,4]=np.float32(.01)+np.float32(.0000005)
        data['labels'][0,5]=-0.;raw=data['labels'].tobytes();m=model();r=inspect.coverage(data,m)
        x,prev,labels,_=inspect.trainer.frozen.tensor_episode(data,'cpu')
        project=lambda v:policy.project_left_targets(v,prev,data['parameters'],data['left_indices'],data['projector'])
        c=policy.classify_branch_labels(labels,project(m.baseline_targets(x,prev)),project(prev))
        for name,key in (('BASE','exact_base'),('HOLD','exact_hold'),('UPDATE','confident_update'),('ambiguous','ambiguous')):
            self.assertEqual(r['classification_boolean_array_sha256'][name],hashlib.sha256(c[key][0].numpy().tobytes()).hexdigest())
        self.assertEqual(r['active_classes'],{'BASE':23,'HOLD':1,'UPDATE':1,'ambiguous':3})
        self.assertEqual(data['labels'].tobytes(),raw)

    def test_each_dataset_original_cap_is_verified_without_mutation(self):
        data=fixture();prior=deepcopy(data['parameters'])
        with patch.object(policy,'verify_projection_arm_bounds',wraps=policy.verify_projection_arm_bounds)as check:
            inspect.coverage(data,model());self.assertEqual(check.call_count,1)
        self.assertEqual(data['parameters'],prior)
        changed=deepcopy(data);changed['parameters']['step_limit'][0][0]=.024
        with self.assertRaisesRegex(ValueError,'step limit'):inspect.coverage(changed,model())

    def test_pre_feedback_correction_is_reported_not_made_representable(self):
        data=fixture(2);data['x'][0]=frame('squeeze',71/84)[0,0].numpy();data['labels'][0,3]=.005
        r=inspect.coverage(data,model())
        self.assertEqual(r['active_frames'],1);self.assertEqual(r['masked_corrections_gt_1e_6'],1)
        self.assertGreater(r['masked_max_base_difference_rad'],.0049)

    def test_excluded_transition_cannot_raise_reported_eligible_count(self):
        data=fixture(2);data['mask'][1]=False;data['labels'][1,3]=1.
        r=inspect.coverage(data,model());self.assertEqual(r['eligible_frames'],1)
        self.assertEqual(r['active_unreachable_elements'],0);self.assertEqual(r['active_no_float32_branch_witness_elements'],0)

    def test_exact_nine_run_identity_excludes_exception_seed05(self):
        self.assertEqual(len(inspect.v8_trainer.RUN_NAMES)+len(inspect.NEW_RUNS),9)
        self.assertEqual(inspect.NEW_RUNS,('left_hold_increment_v8_seed20261104_eval_9501_20261104',))
        self.assertNotIn('left_hold_increment_v8_seed20261105_eval_9501_20261104',inspect.NEW_RUNS)
        self.assertFalse(inspect.CONTRACT['original_environment_projection_or_safety_gates_changed'])


if __name__=='__main__':unittest.main()
