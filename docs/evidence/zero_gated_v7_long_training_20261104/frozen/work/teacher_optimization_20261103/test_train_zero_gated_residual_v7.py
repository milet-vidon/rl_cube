"""Host training/provenance contracts; synthetic inputs do not prove behavior."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import numpy as np
import torch
import train_zero_gated_residual_v7 as trainer
from test_zero_gated_residual_policy_v7 import policy, frame, PARAMETERS, joint_safety
from test_phase_residual_v5 import baseline


def fixture(length=502):
    m=policy()
    x=frame().repeat(1,length,1)
    x[:,:499]=frame('squeeze',71/84)
    previous=torch.zeros(1,length,28)
    with torch.no_grad():
        labels=trainer.policy.project_left_targets(m.baseline_targets(x,previous),previous,PARAMETERS,list(range(28)),joint_safety.project_target)
    data={'x':x[0].numpy(),'previous_left':previous[0].numpy(),'labels':labels[0].numpy(),
          'mask':np.ones(length,dtype=bool),'parameters':PARAMETERS,'left_indices':list(range(28)),
          'projector':joint_safety.project_target,'episode_id':'synthetic','executed_sources':['student']*length,
          'evidence':{'directory':trainer.inspection.RUN_NAMES[0]}}
    return m,data


class V7TrainingContracts(unittest.TestCase):
    def test_bitwise_frozen_tensor_comparison_includes_signed_zero_and_scalars(self):
        self.assertFalse(trainer.bits_equal(torch.tensor(0.),torch.tensor(-0.)))
        self.assertTrue(trainer.bits_equal(torch.tensor(True),torch.tensor(True)))
        self.assertFalse(trainer.bits_equal(torch.tensor([0.]),torch.tensor([0.],dtype=torch.float64)))

    def test_selection_prefers_boundary_then_classification_then_decoded_error(self):
        def metrics(passed,fp,fn,rmse):
            return {'step499_reviewed_zero_histories_bitwise_baseline':passed,'strict_zero_gate_false_positive_count':fp,
                    'certain_nonzero_gate_false_negative_count':fn,'hard_decoded_projected_fit':{'rmse_rad':rmse}}
        self.assertLess(trainer.selection_key(metrics(True,100,200,9)),trainer.selection_key(metrics(False,0,0,0)))
        self.assertLess(trainer.selection_key(metrics(True,1,2,9)),trainer.selection_key(metrics(True,0,4,0)))
        self.assertLess(trainer.selection_key(metrics(True,1,2,.1)),trainer.selection_key(metrics(True,2,1,.2)))

    def test_actual_full_history_assessment_detects_first_added_residual_and_label(self):
        m,data=fixture();m.head.bias.data.fill_(1.);m.correction_gate.bias.data.fill_(1.)
        report=trainer.assess([data],m,streaming=True)
        self.assertFalse(report['step499_reviewed_zero_histories_bitwise_baseline'])
        first=report['runs'][0]['first_nonbaseline_proposal']
        self.assertEqual(first['step'],499)
        self.assertEqual(first['phase'],'squeeze')
        self.assertEqual(len(first['expert_projected_label_rad']),28)
        self.assertEqual(report['strict_zero_gate_false_positive_count'],3*28)
        self.assertEqual(report['certain_nonzero_gate_false_negative_count'],0)

    def test_streamed_boundary_advances_500_frames_from_reset(self):
        m,data=fixture();count=[];original=m.forward_with_diagnostics
        def call(x,previous,h=None):
            count.append(h is None)
            return original(x,previous,h)
        with patch.object(m,'forward_with_diagnostics',side_effect=call):
            report=trainer.assess([data,data],m,streaming=True,boundary_only=True)
        self.assertEqual(len(count),1000)
        self.assertEqual([i for i,x in enumerate(count)if x],[0,500])
        self.assertTrue(report['step499_reviewed_zero_histories_bitwise_baseline'])
        self.assertTrue(all(r['frames_processed']==500 for r in report['runs']))

    def test_missing_boundary_cannot_be_claimed_passed(self):
        m,data=fixture(499)
        with self.assertRaisesRegex(ValueError,'boundary'):
            trainer.assess([data],m,streaming=True)

    def test_failed_v5_step499_recovery_is_not_required_to_equal_baseline(self):
        m,data=fixture();other=deepcopy(data)
        other['evidence']['directory']=trainer.inspection.RUN_NAMES[1]
        other['labels'][499,16:24]+=.005
        report=trainer.assess([data,other],m,streaming=True)
        self.assertTrue(report['step499_reviewed_zero_histories_bitwise_baseline'])
        self.assertFalse(report['runs'][1]['step499_zero_regression_required'])
        self.assertEqual(report['certain_nonzero_gate_false_negative_count'],8)

    def test_ambiguous_labels_stay_in_regression_and_not_classifier(self):
        m,data=fixture();x,previous,labels,mask=trainer.frozen.tensor_episode(data,'cpu')
        labels[0,500,2]+=5e-7
        out,_,_,diag=m.forward_with_diagnostics(x,previous,None)
        base=trainer.policy.project_left_targets(diag['baseline_proposal'],previous,PARAMETERS,data['left_indices'],data['projector'])
        c=trainer.policy.classify_correction_labels(labels,base)
        losses=trainer.policy.supervision_losses(out,diag,previous,labels,mask,c['correction_target'],c['classification_mask'],PARAMETERS,data['left_indices'],data['projector'])
        self.assertGreater(float(losses['regression_mse_rad2'].detach()),0)
        self.assertEqual(losses['ambiguous_continuous_label_count'],1)
        self.assertEqual(losses['classification_label_count'],3*28-1)
        losses['regression_mse_rad2'].backward()
        self.assertGreater(float(m.head.bias.grad[2].abs()),0)

    def test_epoch_zero_never_saves_a_model(self):
        m,_=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            path=Path(td)/'model.pt'
            with self.assertRaisesRegex(ValueError,'Untrained'):
                trainer.save_candidate(path,m,{},0,0,{}, {'inference':'sequential_one_frame_hidden_state'})
            self.assertFalse(path.exists())

    def test_device_probe_never_optimizes_or_changes_model(self):
        m,data=fixture();before={k:v.clone()for k,v in m.state_dict().items()}
        report=trainer.verify_device_definition([data],m,'cpu')
        self.assertFalse(report['optimizer_created'])
        self.assertTrue(report['all_classification_arrays_bitwise_identical'])
        self.assertTrue(all(trainer.bits_equal(before[k],v)for k,v in m.state_dict().items()))

    def test_candidate_initially_explicitly_unqualified_and_clones_weights(self):
        m,_=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            path=Path(td)/'model.pt'
            trainer.save_candidate(path,m,{},1,5,{}, {'inference':'sequential_one_frame_hidden_state'})
            saved=torch.load(path,weights_only=True)
            self.assertIs(saved['metadata']['offline_candidate_qualified'],False)
            self.assertEqual(saved['metadata']['optimizer_steps'],5)

    def test_root_authorization_pins_exact_loss_weights_and_source(self):
        config={'regression_weight_rad2':1e6,'binary_cross_entropy_weight':1.}
        auth={'schema':trainer.AUTH_SCHEMA,'approved_for_development_bc_training':True,
              'qualification_sha256':'q','source_sha256':{'synthetic':'hash'},'training_config':config,
              'selection_contract':trainer.SELECTION_CONTRACT,'eligible_for_rl_episode_training':False,
              'independent_generalization_claim':False}
        text=json.dumps(auth);sha=trainer.digest_bytes(text.encode())
        with patch.object(trainer,'source_hashes',return_value={'synthetic':'hash'}):
            self.assertEqual(trainer.validate_authorization(text,sha,'q',config),auth)
            with self.assertRaises(ValueError):trainer.validate_authorization(text,sha,'q',config|{'binary_cross_entropy_weight':2.})
            with self.assertRaises(ValueError):trainer.validate_authorization(text,'bad','q',config)

    def test_checkpoint_external_qualification_and_authorization_must_match_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            td=Path(td);q=td/'q.json';a=td/'a.json';cp=td/'model.pt'
            q.write_text('{}');a.write_text('{}')
            torch.save({'metadata':{'qualification_text':'different','qualification_sha256':trainer.sha(q)},'model':{}},cp)
            with self.assertRaisesRegex(ValueError,'External reviewed evidence'):
                trainer.load_checkpoint(cp,'unused',[],expected_physics_family=trainer.FAMILY_SHA,
                    expected_checkpoint_sha256=trainer.sha(cp),qualification_path=q,expected_qualification_sha256=trainer.sha(q),
                    authorization_path=a,expected_authorization_sha256=trainer.sha(a))

    def test_checkpoint_bytes_hash_checked_before_unpickling(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            cp=Path(td)/'bad.pt';cp.write_bytes(b'not a torch model')
            with self.assertRaisesRegex(ValueError,'checkpoint bytes'):
                trainer.load_checkpoint(cp,'unused',[],expected_physics_family=trainer.FAMILY_SHA,
                    expected_checkpoint_sha256='bad',qualification_path='unused',expected_qualification_sha256='unused',
                    authorization_path='unused',expected_authorization_sha256='unused')

    def test_load_checkpoint_rejects_signed_zero_changes_in_frozen_baseline(self):
        m,_=fixture();base=baseline()
        metadata={'qualification_text':'{}','qualification_sha256':trainer.digest_bytes(b'{}'),
            'root_authorization_text':'{}','root_authorization_sha256':trainer.digest_bytes(b'{}'),
            'offline_candidate_qualified':True,'normalization_mean':m.mean.tolist(),'normalization_std':m.std.tolist(),
            'residual_bound_rad':m.residual_bound.tolist(),'hidden_size':m.hidden_size,
            'gate_contract':trainer.policy.PUBLIC_GATE_CONTRACT,'zero_gate_contract':trainer.policy.ZERO_GATE_CONTRACT}
        state={k:v.clone()for k,v in m.state_dict().items()}
        state['baseline.heads.0.coefficients'][0,0]=-0.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            td=Path(td);cp=td/'model.pt';q=td/'q.json';a=td/'a.json'
            q.write_text('{}');a.write_text('{}');torch.save({'model':state,'metadata':metadata},cp)
            with (patch.object(trainer.frozen,'load_baseline',return_value=(base,'synthetic')),
                  patch.object(trainer,'verify_metadata')):
                with self.assertRaisesRegex(ValueError,'Frozen baseline tensor differs bitwise'):
                    trainer.load_checkpoint(cp,'unused',[],expected_physics_family=trainer.FAMILY_SHA,
                        expected_checkpoint_sha256=trainer.sha(cp),qualification_path=q,expected_qualification_sha256=trainer.sha(q),
                        authorization_path=a,expected_authorization_sha256=trainer.sha(a))

    def test_metadata_rejects_actual_joint_order_and_physics_family_changes(self):
        m,_=fixture();base=baseline()
        minimal={'schema':trainer.policy.SCHEMA,'feature_schema':trainer.policy.FEATURE_SCHEMA,
            'source_sha256':{'synthetic':'h'},'physics_family_sha256':trainer.FAMILY_SHA,
            'frozen_baseline_checkpoint_sha256':trainer.BASELINE_SHA,'joint_names':base['metadata']['joint_names'],
            'output_indices':list(reversed(base['metadata']['output_indices']))}
        with patch.object(trainer,'source_hashes',return_value={'synthetic':'h'}):
            with self.assertRaisesRegex(ValueError,'joint order'):
                trainer.verify_metadata(minimal,base,trainer.BASELINE_SHA,[],trainer.FAMILY_SHA)
            with self.assertRaisesRegex(ValueError,'physical family'):
                trainer.verify_metadata(minimal,base,trainer.BASELINE_SHA,[],'different')

    def test_rejects_unreviewed_history_order_before_reading(self):
        with self.assertRaisesRegex(ValueError,'five reviewed histories'):
            trainer.load_histories(list(reversed(trainer.inspection.RUN_NAMES)),{},'bad')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
