"""Synthetic host contracts only; no physical-performance or dataset claim."""
from copy import deepcopy
import unittest

import numpy as np
import torch
try:
    import joint_safety
except ImportError:
    import baseline_joint_safety as joint_safety

from test_phase_residual_v5 import baseline, observation
from zero_gated_residual_policy_v7 import (
    ZeroGatedResidualPolicy, ZERO_GATE_CONTRACT, FEATURES, features,
    EpisodeActor, project_left_targets, supervision_losses,
    candidate_bounds, EXPECTED_LEFT_NAMES,
    classify_correction_labels,
)


def policy():
    return ZeroGatedResidualPolicy(baseline(), np.zeros(FEATURES), np.ones(FEATURES), .006, hidden_size=8)


def frame(phase='squeeze', progress=1.):
    obs=observation();obs.update(phase=phase,phase_progress=progress)
    return torch.from_numpy(features(obs)).reshape(1,1,FEATURES)


PARAMETERS={'scales':[[.05]*56],'limits':[[[-2.,2.]]*56],'step_limit':[[.04]*56],
            'command_velocity_limits':[[1.]*56],'control_dt':1/30}


class LearnedZeroContracts(unittest.TestCase):
    def test_named_candidate_bounds_expand_only_the_reviewed_seven_arms(self):
        names=list(EXPECTED_LEFT_NAMES);bounds=candidate_bounds(names)
        self.assertEqual(bounds[:7],[.0248,.0243,.0249,.0076,.0257,.0256,.0244])
        self.assertEqual(sum(x==.006 for x in bounds),11)
        self.assertEqual(sum(x==.000001 for x in bounds),10)
        self.assertEqual(candidate_bounds(names[::-1]),bounds[::-1])
        with self.assertRaises(ValueError):candidate_bounds(names[:-1]+[names[0]])

    def test_zero_branch_is_bitwise_base_even_with_nonzero_residual_head(self):
        m=policy();m.head.bias.data.fill_(5.)
        x=frame();previous=torch.full((1,1,28),.123)
        proposed,hidden,residual,diag=m.forward_with_diagnostics(x,previous)
        base=m.baseline_targets(x,previous)
        self.assertTrue(torch.equal(proposed.view(torch.int32),base.view(torch.int32)))
        self.assertFalse(diag['predicted_correction_class'].any())
        self.assertEqual(torch.count_nonzero(residual),0)
        self.assertGreater(torch.count_nonzero(hidden),0)
        # Same native normalized action and original projection, not just target proximity.
        action=(proposed-previous)/.05;base_action=(base-previous)/.05
        self.assertTrue(torch.equal(action.view(torch.int32),base_action.view(torch.int32)))
        a=project_left_targets(proposed,previous,PARAMETERS,list(range(28)),joint_safety.project_target)
        b=project_left_targets(base,previous,PARAMETERS,list(range(28)),joint_safety.project_target)
        self.assertTrue(torch.equal(a.view(torch.int32),b.view(torch.int32)))

    def test_per_joint_gate_changes_only_chosen_joints_and_respects_bounds(self):
        m=policy();m.head.bias.data.fill_(100.)
        m.correction_gate.bias.data.fill_(-1.);m.correction_gate.bias.data[[2,17]]=1.
        x=frame();previous=torch.zeros(1,1,28)
        proposed,_,residual,diag=m.forward_with_diagnostics(x,previous)
        base=m.baseline_targets(x,previous)
        active=diag['predicted_correction_class'][0,0]
        self.assertEqual(torch.nonzero(active).flatten().tolist(),[2,17])
        self.assertTrue(torch.equal(proposed[0,0,~active].view(torch.int32),base[0,0,~active].view(torch.int32)))
        self.assertTrue((residual.abs()<=.006).all())
        self.assertEqual(tuple(proposed.shape),(1,1,28))

    def test_public_feedback_boundary_overrides_learned_gate_but_history_advances(self):
        m=policy();m.correction_gate.bias.data.fill_(5.);m.head.bias.data.fill_(1.)
        for phase,progress in [('ready',1.),('preshape',1.),('lower',1.),('approach',1.),('squeeze',71/84)]:
            x=frame(phase,progress);out,h,_,d=m.forward_with_diagnostics(x,torch.zeros(1,1,28))
            self.assertFalse(d['predicted_correction_class'].any())
            self.assertGreater(torch.count_nonzero(h),0)
            self.assertTrue(torch.equal(out,m.baseline_targets(x,torch.zeros(1,1,28))))
        _,_,_,d=m.forward_with_diagnostics(frame('squeeze',72/84),torch.zeros(1,1,28))
        self.assertTrue(d['predicted_correction_class'].all())

    def test_gate_decision_does_not_depend_on_residual_amplitude(self):
        m=policy();m.correction_gate.bias.data.fill_(1.)
        m.head.bias.data.fill_(1e-7)
        _,_,residual,diag=m.forward_with_diagnostics(frame(),torch.zeros(1,1,28))
        self.assertTrue(diag['predicted_correction_class'].all())
        self.assertTrue((residual>0).all())
        m.correction_gate.bias.data.fill_(-1.);m.head.bias.data.fill_(100.)
        _,_,residual,diag=m.forward_with_diagnostics(frame(),torch.zeros(1,1,28))
        self.assertFalse(diag['predicted_correction_class'].any())
        self.assertEqual(torch.count_nonzero(residual),0)

    def test_exact_zero_preserves_negative_zero_and_is_not_previous_hold(self):
        m=policy();negative_zero=torch.full((1,1,28),-0.)
        m.baseline_targets=lambda x,previous:negative_zero
        previous=torch.full((1,1,28),.125)
        out,_,_=m(frame(),previous)
        self.assertTrue(torch.equal(out.view(torch.int32),negative_zero.view(torch.int32)))
        self.assertFalse(torch.equal(out,previous))

    def test_zero_output_frames_still_advance_history_for_later_decisions(self):
        torch.manual_seed(17);m=policy()
        a=frame('ready');b=a.clone();b[...,0]+=1.
        _,ha,_,_=m.forward_with_diagnostics(a,torch.zeros(1,1,28))
        _,hb,_,_=m.forward_with_diagnostics(b,torch.zeros(1,1,28))
        self.assertFalse(torch.equal(ha,hb))
        m.correction_gate.weight.data.normal_(0,.2)
        _,_,_,da=m.forward_with_diagnostics(frame(),torch.zeros(1,1,28),ha)
        _,_,_,db=m.forward_with_diagnostics(frame(),torch.zeros(1,1,28),hb)
        self.assertFalse(torch.equal(da['correction_logits'],db['correction_logits']))

    def test_continuous_labels_include_small_floor_and_classification_has_gradients(self):
        m=policy();x=frame();previous=torch.zeros(1,1,28)
        out,_,_,diag=m.forward_with_diagnostics(x,previous)
        labels=torch.zeros(1,1,28);labels[...,0]=1e-7;labels[...,1]=.005
        labels_before=labels.clone();classification=classify_correction_labels(labels,torch.zeros_like(labels))
        losses=supervision_losses(out,diag,previous,labels,torch.ones(1,1,dtype=torch.bool),classification['correction_target'],
                                  classification['classification_mask'],PARAMETERS,list(range(28)),joint_safety.project_target)
        self.assertEqual(losses['continuous_label_count'],28)
        self.assertEqual(losses['classification_label_count'],27)
        self.assertEqual(losses['ambiguous_continuous_label_count'],1)
        self.assertEqual(losses['gate_false_negative_count'],1)
        expected=labels.double().square().mean()
        self.assertEqual(losses['regression_mse_rad2'].item(),expected.item())
        self.assertTrue(torch.equal(labels,labels_before))
        (losses['regression_mse_rad2']+losses['gate_binary_cross_entropy']).backward()
        self.assertGreater(m.head.bias.grad.abs().sum(),0)
        self.assertGreater(m.correction_gate.bias.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.baseline.parameters()))

    def test_publicly_masked_quantization_floor_stays_in_loss_without_classification(self):
        m=policy();previous=torch.zeros(1,1,28);out,_,_,diag=m.forward_with_diagnostics(frame('ready'),previous)
        labels=torch.full_like(out,1.1920928955078125e-7);classification=classify_correction_labels(labels,torch.zeros_like(labels))
        losses=supervision_losses(out,diag,previous,labels,torch.ones(1,1,dtype=torch.bool),classification['correction_target'],
                                  classification['classification_mask'],PARAMETERS,list(range(28)),joint_safety.project_target)
        self.assertEqual(losses['continuous_label_count'],28)
        self.assertEqual(losses['classification_label_count'],0)
        self.assertGreater(losses['regression_mse_rad2'],0)

    def test_labels_and_expert_memory_do_not_enter_actor_and_episode_reset_is_explicit(self):
        obs=observation();obs.update(phase='squeeze',phase_progress=1.)
        changed=deepcopy(obs);changed.update(expert_supervision={'target':[999.]*56},teacher_hidden={'arrival':True},previous_executed_source='teacher')
        np.testing.assert_array_equal(features(obs),features(changed))
        actor=EpisodeActor(policy(),list(range(28)))
        with self.assertRaises(RuntimeError):actor.act(obs,episode_id='a',step=0)
        actor.reset('a');first=actor.act(obs,episode_id='a',step=0)
        with self.assertRaises(RuntimeError):actor.act(obs,episode_id='a',step=2)
        actor.reset('b');np.testing.assert_array_equal(actor.act(changed,episode_id='b',step=0),first)

    def test_invalid_shapes_nonfinite_history_and_architecture_are_rejected(self):
        m=policy();x=frame();previous=torch.zeros(1,1,28)
        with self.assertRaises(ValueError):m(x[0],previous)
        with self.assertRaises(ValueError):m(x,previous,torch.zeros(1,1,9))
        with self.assertRaises(ValueError):m(x,previous,torch.full((1,1,8),float('nan')))
        x[...,0]=float('nan')
        with self.assertRaises(ValueError):m(x,previous)
        contract=deepcopy(ZERO_GATE_CONTRACT);contract['residual_amplitude_deadband']=.0001
        with self.assertRaises(ValueError):ZeroGatedResidualPolicy(baseline(),np.zeros(FEATURES),np.ones(FEATURES),.006,8,zero_gate_contract=contract)

    def test_checkpoint_cannot_change_gates_bounds_normalization_or_baseline(self):
        m=policy();saved=deepcopy(m.state_dict())
        m.load_state_dict(saved)
        names=['correction_logit_threshold','squeeze_active_progress','residual_phase_active',
               'residual_bound','mean','std','training_class_confidence_threshold_rad',next(name for name in saved if name.startswith('baseline.'))]
        for name in names:
            altered=deepcopy(saved)
            if altered[name].dtype==torch.bool:altered[name].view(-1)[0]=~altered[name].view(-1)[0]
            else:altered[name].view(-1)[0]+=1.
            with self.assertRaisesRegex(ValueError,'frozen model contract'):m.load_state_dict(altered)
        with self.assertRaises(ValueError):m.load_state_dict(saved,strict=False)
        altered=deepcopy(saved);altered['correction_gate.bias'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite checkpoint'):m.load_state_dict(altered)

    def test_gate_and_gru_weights_roundtrip_without_changing_actor_contract(self):
        torch.manual_seed(91);m=policy();m.correction_gate.weight.data.normal_(0,.2);m.head.weight.data.normal_(0,.2)
        saved=deepcopy(m.state_dict());restored=policy();restored.load_state_dict(saved)
        x=torch.cat([frame('ready'),frame('squeeze',71/84),frame('squeeze',72/84),frame('primary_lift')],dim=1)
        previous=torch.zeros(1,4,28)
        a,ha,ra,da=m.forward_with_diagnostics(x,previous);b,hb,rb,db=restored.forward_with_diagnostics(x,previous)
        for left,right in [(a,b),(ha,hb),(ra,rb),(da['correction_logits'],db['correction_logits'])]:
            self.assertTrue(torch.equal(left,right))

    def test_reviewed_zero_nonzero_ambiguous_definition_is_not_ulp_heuristic(self):
        base=torch.zeros(1,28);expert=base.clone()
        expert[0,1]=1e-7;expert[0,2]=1e-6;expert[0,3]=float(np.nextafter(np.float32(1e-6),np.float32(np.inf)))
        base[0,4]=-0.;expert[0,4]=0.
        cls=classify_correction_labels(expert,base)
        self.assertTrue(cls['exact_zero'][0,0])
        self.assertTrue(cls['ambiguous'][0,1]and cls['ambiguous'][0,2]and cls['ambiguous'][0,4])
        self.assertTrue(cls['correction_target'][0,3])
        self.assertFalse(cls['classification_mask'][0,1])
        with self.assertRaises(ValueError):classify_correction_labels(expert.double(),base.double())

    def test_ambiguous_label_has_no_BCE_but_keeps_continuous_head_gradient(self):
        m=policy();previous=torch.zeros(1,1,28);out,_,_,diag=m.forward_with_diagnostics(frame(),previous)
        labels=torch.full_like(out,3e-7);cls=classify_correction_labels(labels,torch.zeros_like(labels))
        loss=supervision_losses(out,diag,previous,labels,torch.ones(1,1,dtype=torch.bool),cls['correction_target'],
                               cls['classification_mask'],PARAMETERS,list(range(28)),joint_safety.project_target)
        self.assertEqual(loss['classification_label_count'],0)
        self.assertEqual(loss['ambiguous_continuous_label_count'],28)
        self.assertGreater(loss['regression_mse_rad2'],0)
        (loss['regression_mse_rad2']+loss['gate_binary_cross_entropy']).backward()
        self.assertGreater(m.head.bias.grad.abs().sum(),0)
        self.assertEqual(m.correction_gate.bias.grad.abs().sum(),0)
        bad=torch.ones_like(cls['classification_mask'])
        with self.assertRaisesRegex(ValueError,'reviewed projected-space'):
            supervision_losses(out,diag,previous,labels,torch.ones(1,1,dtype=torch.bool),cls['correction_target'],
                               bad,PARAMETERS,list(range(28)),joint_safety.project_target)


if __name__=='__main__':unittest.main()
