from copy import deepcopy
import unittest
import numpy as np
import torch
from phase_gated_residual_policy_v6 import (PhaseGatedResidualPolicy,GATE_CONTRACT,
    candidate_bounds,EXPECTED_LEFT_NAMES,RECOVERY_JOINTS,features,FEATURES)
from test_phase_residual_v5 import baseline,observation

def model():
    return PhaseGatedResidualPolicy(baseline(),np.zeros(FEATURES),np.ones(FEATURES),.006,hidden_size=8)
def frame(phase='ready',progress=0.):
    obs=observation();obs.update(phase=phase,phase_progress=progress)
    return torch.from_numpy(features(obs)).reshape(1,1,-1)

class PublicPhaseGateTests(unittest.TestCase):
    def test_all_pre_feedback_phases_are_bitwise_base_with_nonzero_head(self):
        m=model();m.head.bias.data.fill_(2.)
        for phase in ('ready','preshape','lower','approach'):
            for progress in (0.,1.):
                x=frame(phase,progress);previous=torch.zeros(1,1,28)
                output,hidden,residual=m(x,previous)
                self.assertTrue(torch.equal(output.view(torch.int32),m.baseline_targets(x,previous).view(torch.int32)))
                self.assertEqual(torch.count_nonzero(residual).item(),0)
                self.assertGreater(torch.count_nonzero(hidden).item(),0)

    def test_actual_feedback_tick_boundary_is_public_72_over_84_float32(self):
        m=model();threshold=np.float32(72/84)
        self.assertFalse(m.residual_active(frame('squeeze',71/84)).item())
        self.assertFalse(m.residual_active(frame('squeeze',float(np.nextafter(threshold,np.float32(-np.inf))))).item())
        self.assertTrue(m.residual_active(frame('squeeze',72/84)).item())
        self.assertTrue(m.residual_active(frame('squeeze',1.)).item())
        self.assertTrue(m.residual_active(frame('primary_lift',0.)).item())

    def test_gru_masked_history_is_retained_and_can_affect_later_active_output(self):
        torch.manual_seed(7);m=model();m.head.weight.data.normal_(0.,.3)
        first=frame('ready');changed=first.clone();changed[...,0]=1.
        _,hidden1,zero1=m(first,torch.zeros(1,1,28))
        _,hidden2,zero2=m(changed,torch.zeros(1,1,28))
        self.assertEqual(torch.count_nonzero(zero1).item()+torch.count_nonzero(zero2).item(),0)
        self.assertFalse(torch.equal(hidden1,hidden2))
        active=frame('primary_lift')
        out1,_,_=m(active,torch.zeros(1,1,28),hidden1)
        out2,_,_=m(active,torch.zeros(1,1,28),hidden2)
        self.assertFalse(torch.equal(out1,out2))

    def test_active_recovery_output_respects_named_bounds(self):
        names=list(EXPECTED_LEFT_NAMES);bounds=candidate_bounds(names)
        self.assertEqual(sum(value==.006 for value in bounds),11)
        self.assertEqual(sum(value==.0005 for value in bounds),7)
        self.assertEqual(sum(value==.000001 for value in bounds),10)
        for name,bound in zip(names,bounds):
            if name in RECOVERY_JOINTS:self.assertEqual(bound,.006)
        reversed_names=names[::-1]
        self.assertEqual(candidate_bounds(reversed_names),bounds[::-1])
        with self.assertRaises(ValueError):candidate_bounds(names[:-1]+['left_unknown_joint'])

    def test_unreviewed_gate_metadata_is_rejected(self):
        altered=deepcopy(GATE_CONTRACT);altered['squeeze_active_progress_numerator']=73
        with self.assertRaises(ValueError):
            PhaseGatedResidualPolicy(baseline(),np.zeros(FEATURES),np.ones(FEATURES),.006,8,altered)

    def test_hidden_expert_fields_and_execution_identity_do_not_change_gate_or_features(self):
        first=observation();second=deepcopy(first)
        second.update(expert_supervision={'target':[999.]*28},arrival={'initialized':True},previous_executed_source='teacher')
        np.testing.assert_array_equal(features(first),features(second))

    def test_masked_base_preserves_negative_zero_bits(self):
        m=model();negative_zero=torch.full((1,1,28),-0.)
        m.baseline_targets=lambda x,previous:negative_zero
        output,_,_=m(frame(),torch.zeros(1,1,28))
        self.assertTrue(torch.equal(output.view(torch.int32),negative_zero.view(torch.int32)))

    def test_original_shape_and_finite_guards_are_preserved(self):
        m=model();x=frame();previous=torch.zeros(1,1,28)
        with self.assertRaises(ValueError):m(x[0],previous)
        with self.assertRaises(ValueError):m(x,previous[0])
        x[...,0]=float('nan')
        with self.assertRaises(ValueError):m(x,previous)

if __name__=='__main__':unittest.main()
