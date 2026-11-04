"""V9 route/actual-cap checks. No Isaac or trained-policy qualification."""
import ast
from copy import deepcopy
import hashlib
import io,json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
import state_arm_hold_increment_eval_runtime_v9 as runtime
import arm_hold_increment_policy_v9 as policy
from test_arm_hold_increment_policy_v9 import policy as make_model,force_modes
from test_arm_hold_increment_policy_v9_peer import params
from test_state_arm_hold_increment_eval_v9 import FakeRuntime,joint_safety
from dagger_collection_core_v1 import PHYSICS_FIELDS

ROOT=Path(__file__).parent


class V9RouteContracts(unittest.TestCase):
    def test_entire_runtime_only_routes_and_adds_initial_actual_projection_check(self):
        before=(ROOT/'state_hold_increment_eval_runtime_v8.py').read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),'8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c')
        after=(ROOT/'state_arm_hold_increment_eval_runtime_v9.py').read_text()
        after=after.replace('from arm_hold_increment_policy_v9 import EpisodeActor, features as v9_features, verify_projection_arm_bounds',
                            'from hold_increment_policy_v8 import EpisodeActor, features as v8_features')
        start=after.index('        actual_parameters = self.read_projection_parameters()\n')
        end=after.index('        self.actor = ',start)
        added=after[start:end]
        self.assertIn('actual_parameters != self.projection_parameters',added)
        self.assertIn('verify_projection_arm_bounds(actual_parameters, self.left_indices)',added)
        self.assertIn('any(proof != actual_arm_proof for proof in reviewed_proofs)',added)
        after=after[:start]+after[end:]
        after=after.replace("== 'v9'","== 'v8'").replace("=='v9'","=='v8'").replace('v9_features(','v8_features(')
        self.assertEqual(after,before.decode())

    def test_main_execution_loop_cleanup_only_changes_route_and_default_budget(self):
        def main(file,normalize=False):
            text=(ROOT/file).read_text()
            if normalize:text=text.replace("'v9'","'v8'").replace('default=3500)','default=2000)')
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='main'),include_attributes=False)
        self.assertEqual(main('evaluate_state_arm_hold_increment_v9.py',True),main('evaluate_state_hold_increment_v8.py'))

    def make_initialization(self):
        obj=runtime.EvaluationRuntime.__new__(runtime.EvaluationRuntime);obj.projection_parameters=params()
        names=obj.projection_parameters['joint_names'];obj.left_indices=list(range(28))
        obj.policy_metadata=obj.baseline_metadata={'joint_names':names,'output_indices':obj.left_indices}
        obj.policy_model=make_model();obj.args=SimpleNamespace(policy='v9',name='mock_initialization')
        obj.physics_identity={key:'synthetic_not_actual_physics'for key in PHYSICS_FIELDS};obj.env=SimpleNamespace(cfg={'joint_names':names})
        proof=policy.verify_projection_arm_bounds(obj.projection_parameters,obj.left_indices)
        obj.evaluation_identity={'expected_physics_family_sha256':runtime.physics_fingerprint(obj.physics_identity),
            'reviewed_original_arm_projection_proofs':[deepcopy(proof)for _ in range(9)]}
        obj.read_projection_parameters=lambda:deepcopy(obj.projection_parameters);obj.write=Mock();obj.actor=None
        return obj

    def test_actual_cap_checked_after_baseline_initialize_before_actor_reset(self):
        obj=self.make_initialization();actor=Mock();events=[]
        with (patch.object(runtime.BaselineRuntime,'_initialize',side_effect=lambda:events.append('base_initialized')),
             patch.object(runtime,'verify_projection_arm_bounds',side_effect=lambda *a: (events.append('actual_cap_verified')or policy.verify_projection_arm_bounds(*a))),
             patch.object(runtime,'EpisodeActor',side_effect=lambda *a: (events.append('actor_constructed')or actor))):
            runtime.EvaluationRuntime._initialize(obj)
        self.assertEqual(events,['base_initialized','actual_cap_verified','actor_constructed']);actor.reset.assert_called_once_with(obj.args.name)
        self.assertEqual(obj.evaluation_identity['actual_original_arm_projection_proof'],obj.evaluation_identity['reviewed_original_arm_projection_proofs'][0])

    def test_bad_actual_or_reviewed_cap_blocks_actor_and_no_physics_step_exists(self):
        for kind in ('actual_cap','changed_parameters','reviewed_cap','missing_history'):
            obj=self.make_initialization()
            if kind=='actual_cap':
                obj.projection_parameters['scales'][0][3]=.024;obj.projection_parameters['step_limit'][0][3]=.024
            elif kind=='changed_parameters':
                obj.read_projection_parameters=lambda:dict(obj.projection_parameters,control_dt=.04)
            elif kind=='reviewed_cap':obj.evaluation_identity['reviewed_original_arm_projection_proofs'][0]['original_projection_cap_rad'][3]=.03
            else:obj.evaluation_identity['reviewed_original_arm_projection_proofs'].pop()
            with self.subTest(kind=kind),patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'EpisodeActor')as actor:
                with self.assertRaises(ValueError):runtime.EvaluationRuntime._initialize(obj)
                actor.assert_not_called();self.assertIsNone(obj.actor)

    def test_update_uses_own_history_one_query_once_and_original_projection(self):
        torch.manual_seed(19);m=make_model();force_modes(m,policy.UPDATE);m.head.bias.data.fill_(.5)
        run=FakeRuntime(success_step=3);run.teacher.name='squeeze';run.actor=policy.EpisodeActor(m,list(range(28)));run.actor.reset(run.args.name)
        def schedule(obs,before,after):
            run.current_schedule={'phase':after['name'],'phase_progress':(71/84,72/84,1.)[len(run.rows)]};return dict(run.current_schedule)
        run.schedule_resolver=schedule;journal=io.StringIO()
        with patch.dict('sys.modules',{'joint_safety':joint_safety}):summary=runtime.evaluate(run,journal,max_steps=10)
        rows=[r for r in map(json.loads,journal.getvalue().splitlines())if r['record_type']=='executed_transition']
        self.assertEqual(len(rows),3);self.assertEqual(summary['teacher_query_count'],3);self.assertEqual(run.actor.next_step,3)
        self.assertEqual(summary['left_teacher_position_action_steps'],0)
        replay=policy.EpisodeActor(m,list(range(28)));replay.reset(run.args.name)
        for t,r in enumerate(rows):
            obs=r['actor_observation_pre']|{'previous_executed_source':'student'}
            predicted=np.asarray(replay.act(obs,episode_id=run.args.name,step=t),np.float32)
            np.testing.assert_array_equal(predicted.view(np.int32),np.asarray(r['predicted_left_target'],np.float32).view(np.int32))
            self.assertTrue(r['execution']['right_native_bits_preserved']);self.assertTrue(r['execution']['actual_projection_bits_verified'])
            if t:self.assertEqual(obs['applied_target'],rows[t-1]['actor_observation_post']['applied_target'])
        self.assertFalse(summary['pure_student_episode_claim'])


if __name__=='__main__':unittest.main()
