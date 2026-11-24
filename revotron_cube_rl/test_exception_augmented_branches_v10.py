"""Host contracts for the new data combination; no simulator or data grant."""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import inspect_exception_augmented_branches_v10 as c
import test_inspect_arm_hold_increment_branches_v9_peer as old_tests
import test_arm_hold_increment_policy_v9_peer as model_tests


def prefix_fixture():
    name=tuple(c.PREFIX_REPORTS)[0]
    data={'x':np.zeros((2,284),np.float32),'previous_left':np.zeros((2,28),np.float32),
        'labels':np.zeros((2,28),np.float32),'mask':np.ones(2,np.bool_),
        'episode_id':'exact-path:'+name,'evidence':{'directory':'/p/'+name,'a':'fixed'},
        'excluded_labels':[{'step':2,'expert_label_present':False}],
        'audit':{'queries':2},'result':{'failure':'original_failure'},'parameters':{'unchanged':True},
        'joint_names':['same'],'left_indices':[0],'collection_source_sha256':'source',
        'physics_family_sha256':'family','consumer_schema':'candidate',
        'training_qualification_granted':False,'eligible_for_rl_episode_training':False,
        'current_V9_training_dataset_admission':False,'exception_query_after_state_known':False,
        'exception_query_label_fabricated':False}
    report={k:deepcopy(v)for k,v in data.items()if k not in ('x','previous_left','labels','mask')}
    report.update(selected_completed_candidate_transitions=2,arrays_sha256=c.array_hashes(data))
    return name,data,report


def histories():
    lengths=(1648,619,752,752,1654,752,752,752,1842,1725,1803)
    return [{'evidence':{'directory':'/p/'+name},'episode_id':'/p/'+name+':'+name,
             'joint_names':['same'],'left_indices':[0],'physics_family_sha256':c.v9.FAMILY_SHA,
             'mask':np.ones(n,np.bool_)}for name,n in zip(c.RUN_NAMES,lengths)]


class ElevenHistoryCandidateTests(unittest.TestCase):
    def test_unchanged_coverage_is_direct_function_and_exact_run_order(self):
        self.assertIs(c.coverage,c.original.coverage)
        self.assertIs(c.CONTRACT,c.original.CONTRACT)
        self.assertEqual(c.RUN_NAMES,c.v9.RUN_NAMES+tuple(c.prefix.RUNS))
        self.assertEqual(len(set(c.RUN_NAMES)),11)

    def test_prefix_array_bytes_and_full_evidence_are_required(self):
        name,data,report=prefix_fixture()
        with patch.object(c.prefix,'RUNS',{name:{'steps':2}}):
            c.validate_prefix_against_completed_report(data,report,name)
            for field in ('labels','previous_left','x'):
                tampered=deepcopy(data);tampered[field].flat[0]=-0.
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'array bytes'):
                    c.validate_prefix_against_completed_report(tampered,report,name)
            for field in ('result','audit','evidence','parameters'):
                tampered=deepcopy(data);tampered[field]['changed']=True
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'evidence differs'):
                    c.validate_prefix_against_completed_report(tampered,report,name)

    def test_exception_end_cannot_be_added_or_marked_as_success(self):
        name,data,report=prefix_fixture()
        with patch.object(c.prefix,'RUNS',{name:{'steps':2}}):
            for field in ('training_qualification_granted','eligible_for_rl_episode_training',
                          'current_V9_training_dataset_admission','exception_query_after_state_known',
                          'exception_query_label_fabricated'):
                d=deepcopy(data);r=deepcopy(report);d[field]=r[field]=True
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'promoted or fabricated'):
                    c.validate_prefix_against_completed_report(d,r,name)
            d=deepcopy(data);d['mask'][1]=False
            with self.assertRaisesRegex(ValueError,'length or mask'):c.validate_prefix_against_completed_report(d,report,name)
            d=deepcopy(data);r=deepcopy(report)
            d['excluded_labels'][0]['expert_label_present']=r['excluded_labels'][0]['expert_label_present']=True
            with self.assertRaisesRegex(ValueError,'exclusion changed'):c.validate_prefix_against_completed_report(d,r,name)

    def test_13051_complete_transitions_order_and_family(self):
        ds=histories();c.verify_history_contract(ds)
        changed=deepcopy(ds);changed[9],changed[10]=changed[10],changed[9]
        with self.assertRaisesRegex(ValueError,'ordered'):c.verify_history_contract(changed)
        changed=deepcopy(ds);changed[-1]['mask'][-1]=False
        with self.assertRaisesRegex(ValueError,'13051|9523'):c.verify_history_contract(changed)
        changed=deepcopy(ds);changed[-1]['physics_family_sha256']='different'
        with self.assertRaisesRegex(ValueError,'domain'):c.verify_history_contract(changed)

    def test_original_float32_cap_witness_preserves_raw_expert_label(self):
        data=old_tests.fixture(1);data['previous_left'][0,3]=-2.207127332687378
        data['labels'][0,3]=-2.2321274280548096;before=data['labels'].tobytes();model=model_tests.model()
        report=c.coverage(data,model)
        self.assertTrue(c.representation_pass([report]))
        checked=c.classify_and_check(data,model,report)
        self.assertEqual(checked['active_class_counts']['UPDATE'],1)
        self.assertEqual(data['labels'].tobytes(),before)

    def test_no_interval_or_masked_shortcut_to_representation_pass(self):
        base={'active_max_union_gap_rad':0.,'active_unreachable_elements':0,
              'active_no_float32_branch_witness_elements':0,'masked_corrections_gt_1e_6':0}
        self.assertTrue(c.representation_pass([base]))
        for key in base:
            bad=deepcopy(base);bad[key]=1e-9 if key=='active_max_union_gap_rad' else 1
            with self.subTest(key=key):self.assertFalse(c.representation_pass([base,bad]))

    def test_policy_class_hash_cannot_be_resealed_as_other_confident_class(self):
        data=old_tests.fixture(1);model=model_tests.model();report=c.coverage(data,model)
        report['classification_boolean_array_sha256']['BASE']='0'*64
        with self.assertRaisesRegex(ValueError,'class bits'):c.classify_and_check(data,model,report)

    def test_external_peer_evidence_hash_is_mandatory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory,patch.object(c,'WORK',Path(directory)):
            p=Path(directory)/'peer.json';p.write_text('{"passed":true}')
            self.assertEqual(c.read_pinned(('peer.json',c.sha(p))),{'passed':True})
            with self.assertRaisesRegex(ValueError,'Reviewed evidence changed'):c.read_pinned(('peer.json','0'*64))

    def test_changed_exception_source_rejected_before_old_graph(self):
        with patch.object(c,'sha',return_value='0'*64),patch.object(c.v9,'source_hashes')as old:
            with self.assertRaisesRegex(ValueError,'reader changed'):c.source_hashes()
            old.assert_not_called()

    def test_saved_exception_verifier_rejects_resealed_saved_evidence(self):
        name,data,report=prefix_fixture()
        with patch.object(c,'source_hashes'),patch.object(c,'reviewed_inputs',return_value=({}, {name:report})),\
             patch.object(c.prefix,'trust_run',return_value=(Path('/p')/name,{'steps':2})),\
             patch.object(c.prefix,'private_trust_registry')as make:
            altered=deepcopy(report['evidence']);altered['a']='resealed'
            with self.assertRaisesRegex(ValueError,'completed reviewed candidate'):
                c.verify_saved_exception_evidence(altered,Path('/p')/name)
            make.assert_not_called()

    def test_saved_verifier_uses_evidence_path_without_actor_forward(self):
        name,data,report=prefix_fixture();run=Path('/p')/name
        source='def verify_evidence(run):\n    return {"physics_family_sha256": "family"}\n'
        import hashlib
        current={'physics_family_sha256':'family','exception_prefix_contract_sha256':'pinned',
            'original_registry2_sha256':c.prefix.PRIOR_SHA,
            'private_engine_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'original_frozen_engine_sha256':c.prefix.prior.ENGINE_SHA,'exact_run_whitelist':name,
            'actor_proof':{'mode':'fresh_complete_single_actual_history_replay',
                'prior_completed_proof_available':False,'new_gru_calls':2},
            'original_assets_sha256':{str(c.ROOT/k):v for k,v in c.prefix.ASSETS.items()}}
        report['evidence']=current
        with ExitStack()as stack:
            stack.enter_context(patch.object(c,'source_hashes'))
            stack.enter_context(patch.object(c,'reviewed_inputs',return_value=({}, {name:report})))
            stack.enter_context(patch.object(c.prefix,'trust_run',return_value=(run,{'steps':2})))
            stack.enter_context(patch.object(c.prefix,'engine_source',return_value=source))
            stack.enter_context(patch.object(c.prefix,'scan_exact_boundary',return_value=report['excluded_labels'][0]))
            stack.enter_context(patch.object(c,'sha',return_value='pinned'))
            stack.enter_context(patch.object(c,'read_pinned',return_value=report))
            stack.enter_context(patch.object(c.v9,'FAMILY_SHA','family'))
            forbidden=stack.enter_context(patch.object(c.prefix.prior.policy.EpisodeActor,'act',side_effect=AssertionError('Extra GRU')))
            self.assertEqual(c.verify_saved_exception_evidence(current,run),current)
            forbidden.assert_not_called()


if __name__=='__main__':unittest.main()
