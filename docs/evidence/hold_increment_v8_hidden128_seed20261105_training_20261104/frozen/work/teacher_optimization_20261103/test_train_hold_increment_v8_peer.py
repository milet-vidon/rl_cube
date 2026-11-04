"""Independent host contracts for V8 trainer; synthetic data is not behavior evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_hold_increment_v8 as trainer
from test_hold_increment_policy_v8 import force_modes
from test_zero_gated_residual_policy_v7 import frame, PARAMETERS, joint_safety
from test_phase_residual_v5 import baseline

torch.set_num_threads(1)


def fixtures(length=546):
    saved = baseline()
    names = list(trainer.policy.EXPECTED_LEFT_NAMES) + ['right_joint_'+str(i) for i in range(28)]
    saved['metadata']['joint_names'] = names
    model = trainer.policy.HoldIncrementPolicy(saved, np.zeros(284), np.ones(284),
        trainer.policy.candidate_bounds(names[:28]), hidden_size=8, left_joint_names=names[:28])
    parameters = deepcopy(PARAMETERS); parameters['joint_names'] = names
    episodes = []
    ring = [names.index(n) for n in trainer.HOLD_BOUNDARY_JOINTS]
    for run in trainer.RUN_NAMES:
        x = frame().repeat(1, length, 1)
        x[:, :499] = frame('squeeze', 71/84)
        previous = torch.zeros(1, length, 28)
        if length > 545 and run in trainer.HOLD_BOUNDARY_RUNS:
            previous[0, 545, ring] = .02
        with torch.no_grad():
            labels = trainer.policy.project_left_targets(model.baseline_targets(x, previous), previous,
                parameters, list(range(28)), joint_safety.project_target)
        if length > 545 and run in trainer.HOLD_BOUNDARY_RUNS:
            labels[0, 545, ring] = previous[0, 545, ring]
        if length > 499 and run not in trainer.ZERO_BOUNDARY_RUNS:
            labels[0, 499, 16:24] += .005
        episodes.append({'x':x[0].numpy(), 'previous_left':previous[0].numpy(), 'labels':labels[0].numpy(),
            'mask':np.ones(length, dtype=bool), 'parameters':parameters, 'left_indices':list(range(28)),
            'joint_names':names, 'projector':joint_safety.project_target, 'episode_id':run,
            'executed_sources':['student']*length, 'evidence':{'directory':run}})
    return saved, model, episodes


def metadata_fixture(saved,model,episodes):
    sources={'synthetic':'not_real_source_evidence'}
    for i,d in enumerate(episodes):
        d['consumer_schema']='mixture' if i==0 else ('legacy' if i<5 else 'registry')
    q={'schema':trainer.QUAL_SCHEMA,'training_authorized':False,'source_sha256':sources,
        'eligible_for_development_bc_transition_supervision':True,
        'physics_family_sha256':trainer.FAMILY_SHA,'frozen_baseline_checkpoint_sha256':trainer.BASELINE_SHA,
        'public_gate_contract':trainer.policy.PUBLIC_GATE_CONTRACT,'action_contract':trainer.policy.ACTION_CONTRACT,
        'classification_contract':trainer.policy.CLASSIFICATION_CONTRACT,'selection_contract':trainer.SELECTION_CONTRACT,
        'datasets':[{'evidence':d['evidence']}for d in episodes],
        'dataset_consumers':[d['consumer_schema']for d in episodes],
        'left_joint_names':list(model.left_joint_names),'update_bound_rad':model.residual_bound.tolist(),
        'update_anchor_is_previous':model.update_anchor_is_previous.tolist(),
        'normalization_mean':model.mean.tolist(),'normalization_std':model.std.tolist(),
        **{k:False for k in ('eligible_for_rl_episode_training','pure_student_episode_success_claim',
                            'independent_generalization_claim','original_result_files_modified')}}
    text=json.dumps(q);digest=hashlib.sha256(text.encode()).hexdigest()
    with patch.object(trainer,'source_hashes',return_value=sources):
        m=trainer.checkpoint_metadata(model,episodes,q,text,digest,'{}',hashlib.sha256(b'{}').hexdigest(),
            {'epochs':2,'hidden_size':model.hidden_size})
    m.update(best_epoch=1,optimizer_steps=8,offline_candidate_qualified=True,
        initial_cpu_sequential_metrics={'classification_error_count':2,'hard_decoded_projected_fit':{'rmse_rad':.01}},
        final_cpu_sequential_metrics={'inference':'sequential_one_frame_hidden_state',
            'classification_error_count':1,'hard_decoded_projected_fit':{'rmse_rad':.001},
            'step499_reviewed_zero_histories_bitwise_baseline':True,
            'step545_reviewed_hold_histories_exact_projected_hold':True,
            'step499_regression_violation_count':0,'step545_regression_violation_count':0})
    return q,m,sources


class V8TrainerPeerContracts(unittest.TestCase):
    def test_eight_order_and_seven_zero_regressions_are_explicit(self):
        self.assertEqual(len(trainer.RUN_NAMES), 8)
        self.assertEqual(len(trainer.ZERO_BOUNDARY_RUNS), 7)
        self.assertNotIn('left_state_residual_v5_long_eval_9501_20261104_v2', trainer.ZERO_BOUNDARY_RUNS)
        with self.assertRaisesRegex(ValueError, 'eight ordered'):
            trainer.load_histories(trainer.RUN_NAMES[::-1], {}, 'unused')

    def test_assessment_confusion_preserves_v5_recovery_and_hold_labels(self):
        _, model, episodes = fixtures()
        report = trainer.assess(episodes, model, streaming=True)
        self.assertTrue(report['step499_reviewed_zero_histories_bitwise_baseline'])
        self.assertEqual(report['step499_regression_violation_count'], 0)
        self.assertEqual(report['step545_regression_violation_count'], 6)
        active = 8 * (546 - 499) * 28
        self.assertEqual(report['confusion_matrix_true_by_predicted'], [[active-14,0,0],[6,0,0],[8,0,0]])
        self.assertEqual(report['classification_error_count'], 14)
        self.assertFalse(report['runs'][1]['step499_zero_regression_required'])
        self.assertEqual(report['inference'], 'sequential_one_frame_hidden_state')

    def test_boundary_stream_restarts_hidden_per_episode_and_runs_through_545(self):
        _, model, episodes = fixtures(550)
        resets = []; original = model.forward_with_diagnostics
        def tracked(x, previous, hidden=None):
            resets.append(hidden is None)
            return original(x, previous, hidden)
        with patch.object(model, 'forward_with_diagnostics', side_effect=tracked):
            report = trainer.assess(episodes, model, streaming=True, boundary_only=True)
        self.assertEqual(len(resets), 8*546)
        self.assertEqual([i for i,v in enumerate(resets) if v], [i*546 for i in range(8)])
        self.assertTrue(all(r['frames_processed']==546 for r in report['runs']))
        with self.assertRaisesRegex(ValueError, 'stream'):
            trainer.assess(episodes, model, boundary_only=True)

    def test_hold_beyond_moving_base_bounds_is_supported_and_checked_by_actual_projection(self):
        _, model, episodes = fixtures()
        force_modes(model, trainer.policy.HOLD)
        report = trainer.assess(episodes, model, streaming=True)
        self.assertEqual(report['step545_regression_violation_count'], 0)
        self.assertTrue(report['step545_reviewed_hold_histories_exact_projected_hold'])
        for run in report['runs']:
            if run['step545_hold_regression_required']:
                self.assertEqual(run['step545']['projection_exact_HOLD'], [True]*3)
                self.assertEqual(run['step545']['predicted_classes'], [trainer.policy.HOLD]*3)

    def test_changed_reviewed_boundary_labels_fail_closed_not_relabelled(self):
        _, model, episodes = fixtures()
        episodes[0]['labels'][499,0] += .005
        with self.assertRaisesRegex(ValueError, '499 label'):
            trainer.assess(episodes, model)
        _, model, episodes = fixtures()
        i = trainer.RUN_NAMES.index(trainer.HOLD_BOUNDARY_RUNS[0])
        j = episodes[i]['joint_names'].index(trainer.HOLD_BOUNDARY_JOINTS[0])
        episodes[i]['labels'][545,j] += .005
        with self.assertRaisesRegex(ValueError, '545 expert'):
            trainer.assess(episodes, model)

    def test_excluded_hold_boundary_cannot_qualify(self):
        _,model,episodes=fixtures()
        i=trainer.RUN_NAMES.index(trainer.HOLD_BOUNDARY_RUNS[0])
        episodes[i]['mask'][545]=False
        with self.assertRaisesRegex(ValueError,'545'):
            trainer.assess(episodes,model)

    def test_selection_orders_actual_boundary_then_hold_then_confusion_then_decoded_fit(self):
        def score(a,b,c,d):
            return trainer.selection_key({'step499_regression_violation_count':a,
                'step545_regression_violation_count':b, 'classification_error_count':c,
                'hard_decoded_projected_fit':{'rmse_rad':d},
                'label_selected_projected_fit':{'rmse_rad':0.}})
        self.assertLess(score(0,99,99,99), score(1,0,0,0))
        self.assertLess(score(0,0,99,99), score(0,1,0,0))
        self.assertLess(score(0,0,0,99), score(0,0,1,0))
        self.assertLess(score(0,0,0,.1), score(0,0,0,.2))

    def test_named_anchor_and_bounds_cannot_be_silently_permuted(self):
        _, model, episodes = fixtures()
        trainer.verify_model_named_contract(model, episodes)
        model.update_anchor_is_previous[0] = True
        with self.assertRaisesRegex(ValueError, 'anchor mask'):
            trainer.verify_model_named_contract(model, episodes)
        _, model, episodes = fixtures()
        model.residual_bound[0] += 1e-7
        with self.assertRaisesRegex(ValueError, 'UPDATE bounds'):
            trainer.verify_model_named_contract(model, episodes)
        _, model, episodes = fixtures()
        model.left_joint_names = model.left_joint_names[::-1]
        with self.assertRaisesRegex(ValueError, 'left-name order'):
            trainer.verify_model_named_contract(model, episodes)

    def test_root_authorization_binds_sources_loss_selection_and_immutable_candidate(self):
        config={'regression_weight_rad2':1e6, 'cross_entropy_weight':1.}
        with patch.object(trainer, 'source_hashes', return_value={'synthetic':'not_real_physics_evidence'}):
            auth = trainer.authorization_template('q', config)
            text = json.dumps(auth); digest = hashlib.sha256(text.encode()).hexdigest()
            trainer.validate_authorization(text, digest, 'q', config)
            for changed in (config|{'cross_entropy_weight':2.}, config|{'regression_weight_rad2':1.}):
                with self.assertRaises(ValueError):
                    trainer.validate_authorization(text, digest, 'q', changed)
            with self.assertRaises(ValueError):
                trainer.validate_authorization(text, 'bad', 'q', config)
        candidate={'training_authorized':False, 'original_result_files_modified':False}
        text=json.dumps(candidate); digest=hashlib.sha256(text.encode()).hexdigest()
        trainer.validate_qualification(text,digest,candidate)
        with self.assertRaises(ValueError):
            trainer.validate_qualification(text,digest,candidate|{'training_authorized':True})

    def test_checkpoint_evidence_dispatch_uses_registry_for_all_new_v7_histories(self):
        saved,model,episodes=fixtures();_,metadata,sources=metadata_fixture(saved,model,episodes)
        calls=[]
        def capture(kind):
            return lambda m,paths:calls.append((kind,Path(paths[0]).name))
        with (patch.object(trainer,'source_hashes',return_value=sources),
              patch.object(trainer,'validate_authorization'),
              patch.object(trainer.prior.mixture_reader,'verify_saved_datasets',side_effect=capture('mixture')),
              patch.object(trainer.prior.reader,'verify_saved_datasets',side_effect=capture('legacy')),
              patch.object(trainer.registry,'verify_saved_datasets',side_effect=capture('registry'))):
            trainer.verify_metadata(metadata,saved,trainer.BASELINE_SHA,trainer.RUN_NAMES,trainer.FAMILY_SHA)
        self.assertEqual(calls,[(('mixture'if i==0 else'legacy'if i<5 else'registry'),n)
                                for i,n in enumerate(trainer.RUN_NAMES)])

    def test_checkpoint_hash_is_verified_before_deserialization(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            cp=Path(td)/'invalid.pt';cp.write_bytes(b'not a torch checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint bytes'):
                trainer.load_checkpoint(cp,'unused',[],expected_physics_family=trainer.FAMILY_SHA,
                    expected_checkpoint_sha256='incorrect',qualification_path='unused',expected_qualification_sha256='unused',
                    authorization_path='unused',expected_authorization_sha256='unused')

    def test_checkpoint_external_reviewed_bytes_not_only_embedded_claim(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            td=Path(td);cp=td/'model.pt';q=td/'q.json';auth=td/'a.json'
            q.write_text('{}');auth.write_text('{}')
            torch.save({'model':{},'metadata':{'qualification_text':'changed','qualification_sha256':trainer.sha(q)}},cp)
            with self.assertRaisesRegex(ValueError,'External reviewed evidence'):
                trainer.load_checkpoint(cp,'unused',[],expected_physics_family=trainer.FAMILY_SHA,
                    expected_checkpoint_sha256=trainer.sha(cp),qualification_path=q,expected_qualification_sha256=trainer.sha(q),
                    authorization_path=auth,expected_authorization_sha256=trainer.sha(auth))

    def test_checkpoint_rejects_changed_static_anchor_and_signed_zero_baseline(self):
        saved,model,episodes=fixtures();_,metadata,_=metadata_fixture(saved,model,episodes)
        for tensor_name in ('update_anchor_is_previous','baseline.heads.0.coefficients'):
            with self.subTest(tensor_name=tensor_name),tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
                td=Path(td);cp=td/'model.pt';q=td/'q.json';auth=td/'a.json'
                q.write_text(metadata['qualification_text']);auth.write_text(metadata['root_authorization_text'])
                state={k:v.clone()for k,v in model.state_dict().items()}
                if tensor_name=='update_anchor_is_previous':state[tensor_name][0]=not bool(state[tensor_name][0])
                else:state[tensor_name][0,0]=-0.
                torch.save({'model':state,'metadata':metadata},cp)
                with (patch.object(trainer,'verify_metadata'),
                      patch.object(trainer.prior.frozen,'load_baseline',return_value=(saved,trainer.BASELINE_SHA))):
                    with self.assertRaisesRegex(ValueError,'static tensor differs bitwise'):
                        trainer.load_checkpoint(cp,'unused',trainer.RUN_NAMES,expected_physics_family=trainer.FAMILY_SHA,
                            expected_checkpoint_sha256=trainer.sha(cp),qualification_path=q,expected_qualification_sha256=trainer.sha(q),
                            authorization_path=auth,expected_authorization_sha256=trainer.sha(auth))

    def test_epoch_zero_and_non_cpu_candidates_are_rejected(self):
        _,model,_=fixtures()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as td:
            path=Path(td)/'model.pt'
            with self.assertRaisesRegex(ValueError,'Untrained'):
                trainer.save_candidate(path,model,{},0,0,{}, {})
            with self.assertRaisesRegex(ValueError,'CPU sequential'):
                trainer.save_candidate(path,model,{},1,8,{'inference':'complete_causal_history_batch'},
                    {'inference':'sequential_one_frame_hidden_state'})
            self.assertFalse(path.exists())

    def test_metadata_architecture_epoch_and_step_count_match_exact_authorization(self):
        saved,model,episodes=fixtures();_,metadata,sources=metadata_fixture(saved,model,episodes)
        for key,value in (('hidden_size',model.hidden_size+1),('best_epoch',3),('optimizer_steps',7)):
            changed=deepcopy(metadata);changed[key]=value
            with (self.subTest(key=key),patch.object(trainer,'source_hashes',return_value=sources),
                  patch.object(trainer,'validate_authorization'),
                  patch.object(trainer.prior.mixture_reader,'verify_saved_datasets'),
                  patch.object(trainer.prior.reader,'verify_saved_datasets'),
                  patch.object(trainer.registry,'verify_saved_datasets')):
                with self.assertRaises(ValueError):
                    trainer.verify_metadata(changed,saved,trainer.BASELINE_SHA,trainer.RUN_NAMES,trainer.FAMILY_SHA)

    def test_device_definition_check_does_not_optimize_or_mutate_weights(self):
        _,model,episodes=fixtures()
        original={k:v.clone()for k,v in model.state_dict().items()}
        report=trainer.verify_device_definition(episodes,model,'cpu')
        self.assertFalse(report['optimizer_created']);self.assertFalse(report['training_started'])
        self.assertTrue(report['all_branch_and_class_arrays_bitwise_identical'])
        self.assertTrue(all(trainer.bits_equal(original[k],v)for k,v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
