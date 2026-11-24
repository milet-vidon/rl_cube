"""CPU contracts only; synthetic fixtures never claim physical performance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
try:
    import joint_safety
except ImportError:
    import baseline_joint_safety as joint_safety

import phase_basis_policy_v4 as v4
import phase_contact_policy as contact
import dagger_projected_collection_v3 as reader
import dagger_collection_core_v1 as core
from dagger_collection_core_v1 import physics_fingerprint
from phase_residual_policy_v5 import PhaseResidualPolicy, EpisodeActor, features, FEATURES, SCHEMA, FEATURE_SCHEMA, project_left_targets
from dagger_residual_dataset_v5 import load_run, verify_evidence, REQUIRED_FILES, sha
import train_phase_residual_v5 as training

NAMES = ['left_joint_' + str(i) for i in range(28)] + ['right_joint_' + str(i) for i in range(28)]


def observation():
    return {'q': [0.] * 56, 'joint_velocity': [0.] * 56, 'applied_target': [0.] * 56,
            'cube_position': [.4, .1, 1.], 'cube_quaternion_wxyz': [1., 0., 0., 0.],
            'cube_linear_velocity': [0.] * 3, 'cube_angular_velocity': [0.] * 3,
            'finger_contact_impulses': {side: {finger: 0. for finger in contact.FINGERS} for side in ('left', 'right')},
            'finger_contact_points': [], 'phase': 'ready', 'phase_progress': 0., 'previous_executed_source': 'reset'}


def baseline():
    heads = [(torch.tensor([0., .5]), torch.zeros(8, 28)) for _ in contact.PHASES]
    model = v4.PhaseBasisPolicy([-2.] * 28, [2.] * 28, heads)
    return {'model': model.state_dict(), 'metadata': {'schema': v4.SCHEMA, 'joint_names': NAMES,
        'output_indices': list(range(28)), 'sources': [{'manifest_sha256': 'a' * 64}],
        'source_sha256': {Path(m.__file__).name: sha(m.__file__) for m in (v4, contact)}}}


def model(bound=.01):
    return PhaseResidualPolicy(baseline(), np.zeros(FEATURES), np.ones(FEATURES), bound, hidden_size=8)


def gate():
    return {'robot_safety': True, 'joint_limits': True, 'cube_integrity': True, 'no_pose_write': True,
            'stage_gate_qualified': True, 'original_audit_source_sha256': {'fixture': 'not_physics_evidence'}}


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.run = self.root / 'synthetic_complete_v3'; self.run.mkdir()
        self.baseline = baseline(); self.baseline_path = self.root / 'baseline.pt'; torch.save(self.baseline, self.baseline_path)
        self.base_hash = sha(self.baseline_path)
        self.write('config.json', {'joint_names': NAMES})
        self.write('runtime_protocol.json', {'source_manifest': {'manifest_sha256': 'a' * 64}})
        source = {'environment.py': 'b' * 64, 'joint_safety.py': sha(joint_safety.__file__),
                  'dagger_projected_collection_v3.py': sha(reader.__file__), 'dagger_collection_core_v1.py':sha(core.__file__)}
        physics = {'environment_class': 'articulated_cube_env.DynamicPickupEnv', 'config_sha256': sha(self.run/'config.json'),
                   'assets_sha256': {'fixture': 'not_real_assets'}, 'source_sha256': source,
                   'cube_constraint_topology': {'d6_damping_augmentation': False, 'native_articulation_facade': False},
                   'damping': {'fixture': 1.}, 'armature': {'fixture': 0.}, 'friction': {'fixture': .9},
                   'solver_options': {'fixture': 'not_simulation'}}
        self.full = {'student_checkpoint_sha256': self.base_hash, 'source_sha256': source,
                     'physics_identity': physics, 'physics_identity_sha256': physics_fingerprint(physics),
                     'capture_begins_at_episode_reset': True, 'policy_observation_scope': 'privileged_simulator_state_not_vision_actor'}
        self.ref = reader.persist_provenance(self.run, self.full)
        scales = torch.full((1,56), .05); cap = torch.full((1,56), .04)
        self.params = {'schema': 'original_baseline_project_target_v1', 'joint_names': NAMES,
                       'dtype': 'torch.float32', 'limits': [[[-2., 2.]] * 56], 'scales': scales.tolist(),
                       'max_target_delta': cap.tolist(), 'step_limit': torch.minimum(scales,cap).tolist(),
                       'command_velocity_limits': [[1.] * 56], 'control_dt': 1/30,
                       'source_sha256': {key:source[key] for key in ('environment.py','joint_safety.py')}}
        self.write('projection_parameters.json', self.params)
        self.records = [];self.raw_records=[]
        pre = observation()
        for index, source_name in enumerate(('teacher', 'student')):
            pre = deepcopy(pre)
            pre['phase_progress'] = index * .1
            old = torch.tensor(pre['applied_target']).reshape(1,56)
            action = torch.full((1,56), .1)
            requested = old + scales * action
            projected, hit = joint_safety.project_target(old, requested, torch.tensor(self.params['limits']), cap, torch.ones(1,56), 1/30)
            actual = projected[0].clone()
            if source_name == 'student': actual[:28] = old[0,:28] + .002
            post = deepcopy(pre); post['applied_target'] = actual.tolist(); post['previous_executed_source'] = source_name
            post['q'][0] += .001
            expert = {'failure_reason': None, 'oracle_env_inputs':{'episode_length':index,'max_episode_length':3500}, 'public_same_action_schedule': {'phase': pre['phase'], 'phase_progress': pre['phase_progress']},
                      'projection_previous_applied_target': old[0].tolist(), 'normalized_action': action[0].tolist(),
                      'requested_target': requested[0].tolist(), 'projected_target': projected[0].tolist(),
                      'projection_soft_limit_hit': hit[0].tolist(), 'projection_parameters_file': 'projection_parameters.json',
                      'projection_parameters_file_sha256': sha(self.run/'projection_parameters.json'),
                      'projection_source_sha256': self.params['source_sha256']}
            self.records.append({'schema':'left_contact_dagger_actual_state_transition_v1','supervision_schema':'baseline_left_projected_expert_v3',
                'episode':'synthetic', 'step':index, 'run_provenance':self.ref, 'actor_observation_pre':pre,
                'actor_observation_post':post,'actual_applied_target':actual.tolist(),'expert_supervision':expert,
                'executed_source':source_name, 'expert_projection_matches_teacher_actual_bitwise':True if source_name=='teacher' else None,
                'pre_gates':gate(),'post_gates':gate(),'phase_transition':False,'stage_complete':False,
                'failure_reason':None,'terminal':False})
            native=action.clone()
            if source_name=='student':native[:,:28]=(actual[:28]-old[0,:28])/scales[0,:28]
            self.raw_records.append({'step':index,'phase':pre['phase'],'success':False,'drop':False,
                'terms':{'terminal':0.,'timeout':0.},'executed_source':source_name,
                'native_action':native[0].tolist(),'applied_target':actual.tolist(),
                'termination':{'raw_done':False,'done_matches_original_causes':True,
                               'derived_timeout_truncated':False,'derived_drop_terminated':False}})
            pre = post
        self.records.append({'record_type':'abort','episode':'synthetic','step':2,'run_provenance':self.ref,'reason':'fixture_budget_end'})
        self.write('result.json', {'steps':2,'failure':'fixture_budget_end','eligible_training_episode':False})
        self.flush()

    def write(self, name, obj):
        (self.run/name).write_text(json.dumps(obj,allow_nan=False))

    def flush(self):
        (self.run/'transitions.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in self.records))
        (self.run/'trajectory.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in self.raw_records))
        self.write('artifact_manifest.json', {'sha256': {name:sha(self.run/name) for name in REQUIRED_FILES}})

    def load(self, family=None): return load_run(self.run,self.baseline,self.base_hash,family)


class ModelContracts(unittest.TestCase):
    def test_actor_ignores_labels_hidden_memory_and_execution_identity(self):
        obs = observation(); reference = features(obs)
        other = deepcopy(obs); other.update(expert_supervision={'projected_target':[999.]*56},teacher_hidden={'tick':9000},previous_executed_source='student')
        np.testing.assert_array_equal(reference, features(other))
        other['q'][0] = .02
        self.assertFalse(np.array_equal(reference,features(other)))

    def test_quaternion_sign_equivalence_and_contact_geometry_are_observed(self):
        a = observation(); b = deepcopy(a); b['cube_quaternion_wxyz']=[-1.,0.,0.,0.]
        np.testing.assert_array_equal(features(a),features(b))
        b['finger_contact_points']=[{'side':'left','finger':'index','impulse_Ns':.002,'position_world_m':[.41,.11,1.], 'normal_world':[1.,0.,0.]}]
        b['finger_contact_impulses']['left']['index']=.002
        self.assertFalse(np.array_equal(features(a),features(b)))

    def test_zero_residual_is_exact_frozen_baseline_and_gradients_stay_out(self):
        policy=model(); x=torch.from_numpy(features(observation())).reshape(1,1,FEATURES); previous=torch.zeros(1,1,28)
        predicted, _, residual=policy(x,previous)
        self.assertTrue(torch.equal(predicted,policy.baseline_targets(x,previous)))
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))
        ((predicted-.003)**2).mean().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in policy.baseline.parameters()))
        self.assertIsNotNone(policy.head.weight.grad)

    def test_residual_bound_and_left_only_output(self):
        policy=model(.004)
        with torch.no_grad(): policy.head.bias.fill_(100.)
        x=torch.from_numpy(features(observation())).reshape(1,1,FEATURES); previous=torch.zeros(1,1,28)
        prediction,_,residual=policy(x,previous)
        self.assertEqual(prediction.shape,(1,1,28))
        self.assertTrue((residual.abs()<=.004001).all())

    def test_actor_reset_and_recurrent_batch_step_equivalence(self):
        torch.manual_seed(10);policy=model()
        with torch.no_grad(): policy.head.weight.normal_(0,.1)
        obs=[observation() for _ in range(3)]
        for i,item in enumerate(obs):item['q'][0]=.1*i
        x=torch.from_numpy(np.stack([features(item)for item in obs])).unsqueeze(0)
        with torch.no_grad():whole,_,_=policy(x,torch.zeros(1,3,28))
        actor=EpisodeActor(policy,list(range(28)))
        with self.assertRaises(RuntimeError):actor.act(obs[0],episode_id='a',step=0)
        actor.reset('a')
        streamed=[actor.act(item,episode_id='a',step=i)for i,item in enumerate(obs)]
        np.testing.assert_allclose(streamed,whole[0].detach().numpy(),atol=1e-7,rtol=0)
        with self.assertRaises(RuntimeError):actor.act(obs[0],episode_id='b',step=0)
        actor.reset('b');first=actor.act(obs[0],episode_id='b',step=0)
        np.testing.assert_array_equal(first,streamed[0])


class DatasetContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent);self.addCleanup(self.tmp.cleanup)
        self.module_patch=patch.dict('sys.modules',{'joint_safety':joint_safety});self.module_patch.start();self.addCleanup(self.module_patch.stop)
        self.fixture=Fixture(self.tmp.name)

    def test_expert_label_not_student_actual_and_abort_is_not_a_sample(self):
        data=self.fixture.load()
        self.assertEqual(data['x'].shape,(2,FEATURES))
        self.assertEqual(data['labels'][1,0],np.float32(self.fixture.records[1]['expert_supervision']['projected_target'][0]))
        self.assertNotEqual(data['labels'][1,0],np.float32(self.fixture.records[1]['actual_applied_target'][0]))
        self.assertEqual(data['mask'].tolist(),[True,True])
        self.assertEqual(data['excluded_labels'][-1]['reason'],'terminal_abort_not_a_transition')

    def test_external_provenance_tamper_is_rejected_even_with_updated_manifest(self):
        path=self.fixture.run/'collection_provenance.json';path.write_text(path.read_text()+' ');self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):self.fixture.load()

    def test_wrong_physics_family_and_native_domain_rejected(self):
        with self.assertRaisesRegex(ValueError,'physics family'):self.fixture.load('0'*64)

    def test_forged_projected_label_rejected_after_manifest_rehash(self):
        self.fixture.records[1]['expert_supervision']['projected_target'][0]=.4;self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'does not reproduce'):self.fixture.load()

    def test_missing_projected_label_never_falls_back_to_actual(self):
        del self.fixture.records[1]['supervision_schema'];self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'actual action is not a label'):self.fixture.load()

    def test_teacher_bitwise_confirmation_and_actual_history_required(self):
        self.fixture.records[0]['expert_projection_matches_teacher_actual_bitwise']=False;self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'not verified'):self.fixture.load()
        self.fixture.records[0]['expert_projection_matches_teacher_actual_bitwise']=True
        self.fixture.records[1]['actor_observation_pre']['q'][0]=5.;self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'discontinuity'):self.fixture.load()

    def test_unsafe_final_transition_masked_and_never_changes_labels(self):
        self.fixture.records[1]['failure_reason']='robot_safety_gate';self.fixture.records[1]['post_gates']['robot_safety']=False
        self.fixture.records[1]['terminal']=True;self.fixture.records.pop()
        self.fixture.flush();data=self.fixture.load()
        self.assertEqual(data['mask'].tolist(),[True,False])
        self.assertAlmostEqual(float(data['labels'][1,0]),.01,places=7)

    def test_original_left_projection_matches_full_runtime_arithmetic(self):
        data=self.fixture.load();previous=torch.tensor(data['previous_left']).unsqueeze(0)
        proposed=previous+.08
        actual=project_left_targets(proposed,previous,data['parameters'],data['left_indices'],data['projector'])
        for i in range(2):
            old=torch.tensor(self.fixture.records[i]['actor_observation_pre']['applied_target']).reshape(1,56)
            target=old.clone();target[:,:28]=proposed[0,i]
            scales=torch.tensor(data['parameters']['scales']);action=(target-old)/scales
            full,_=joint_safety.project_target(old,old+scales*action,torch.tensor(data['parameters']['limits']),torch.tensor(data['parameters']['step_limit']),torch.tensor(data['parameters']['command_velocity_limits']),data['parameters']['control_dt'])
            self.assertTrue(torch.equal(actual[0,i].view(torch.int32),full[0,:28].view(torch.int32)))

    def test_bound_inspection_reports_misses_without_clipping(self):
        data=self.fixture.load();labels=data['labels'].copy()
        report=training.inspect([data],model(.0001))
        self.assertGreater(report['unreachable_label_elements'],0)
        self.assertFalse(report['labels_clipped_or_bound_misses_filtered'])
        np.testing.assert_array_equal(data['labels'],labels)

    def test_cli_inspect_does_not_train_and_unreachable_train_is_refused(self):
        output=Path(self.tmp.name)/'inspect'
        args=['train_phase_residual_v5.py','--run',str(self.fixture.run),'--baseline',str(self.fixture.baseline_path),'--output',str(output),'--residual-bound','.0001']
        with patch('sys.argv',args),patch.object(training,'train',side_effect=AssertionError('Must not train')):training.main()
        self.assertTrue((output/'inspection.json').exists());self.assertFalse((output/'model.pt').exists())
        args[args.index('--output')+1]=str(Path(self.tmp.name)/'refused_train');args.append('--train')
        with patch('sys.argv',args),patch.object(training,'train',side_effect=AssertionError('Must not train')):
            with self.assertRaisesRegex(ValueError,'cannot reach original labels'):training.main()

    def test_checkpoint_load_revalidates_external_data_family_and_frozen_baseline(self):
        data=self.fixture.load();policy=model()
        metadata={'schema':SCHEMA,'feature_schema':FEATURE_SCHEMA,'physics_family_sha256':data['physics_family_sha256'],
                  'source_sha256':training.source_hashes(),'frozen_baseline_checkpoint_sha256':self.fixture.base_hash,
                  'dataset_evidence':[data['evidence']],'residual_bound_rad':policy.residual_bound.tolist(),'hidden_size':8}
        path=Path(self.tmp.name)/'untrained_contract_only.pt';torch.save({'metadata':metadata,'model':policy.state_dict()},path)
        restored,_=training.load_checkpoint(path,self.fixture.baseline_path,[self.fixture.run],expected_physics_family=data['physics_family_sha256'])
        self.assertTrue(all(not p.requires_grad for p in restored.baseline.parameters()))
        with self.assertRaisesRegex(ValueError,'different physics family'):
            training.load_checkpoint(path,self.fixture.baseline_path,[self.fixture.run],expected_physics_family='0'*64)
        (self.fixture.run/'collection_provenance.json').write_text('{}')
        with self.assertRaises(ValueError):training.load_checkpoint(path,self.fixture.baseline_path,[self.fixture.run],expected_physics_family=data['physics_family_sha256'])

    def test_qualification_preserves_false_episode_flag_and_requires_pinned_review(self):
        data=self.fixture.load();document=training.qualification_manifest([data],self.fixture.base_hash,model())
        self.assertTrue(document['eligible_for_bc_transition_supervision'])
        self.assertFalse(document['eligible_for_rl_episode_training'])
        self.assertFalse(document['datasets'][0]['original_result_flags']['eligible_training_episode'])
        self.assertEqual(document['datasets'][0]['eligible_transition_steps'],[0,1])
        path=Path(self.tmp.name)/'qualification.json';path.write_text(json.dumps(document))
        digest=sha(path)
        training.require_reviewed_qualification(path,digest,document)
        with self.assertRaises(ValueError):training.require_reviewed_qualification(None,None,document)
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):training.require_reviewed_qualification(path,digest,document)

    def test_raw_done_cannot_be_ignored_and_unfinished_physics_rows_rejected(self):
        self.fixture.raw_records[0]['success']=True
        self.fixture.raw_records[0]['terms']['terminal']=1.
        self.fixture.raw_records[0]['termination']['raw_done']=True
        self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'done was ignored'):self.fixture.load()
        self.fixture.raw_records[0]['success']=False
        self.fixture.raw_records[0]['terms']['terminal']=0.
        self.fixture.raw_records[0]['termination']['raw_done']=False
        self.fixture.raw_records.append(deepcopy(self.fixture.raw_records[-1]));self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'Unfinished/unrecorded'):self.fixture.load()

    def test_right_native_action_mutation_is_rejected(self):
        self.fixture.raw_records[1]['native_action'][40]=.5;self.fixture.flush()
        with self.assertRaisesRegex(ValueError,'right-hand native action'):self.fixture.load()


if __name__=='__main__': unittest.main()
