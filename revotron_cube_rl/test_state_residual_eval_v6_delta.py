"""V6 policy routing adds no control/termination semantics to frozen V5."""
import ast
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from phase_gated_residual_policy_v6 import EpisodeActor,features
from test_phase_gated_residual_v6 import model
from test_state_residual_eval_v6 import FakeRuntime,joint_safety
from state_residual_eval_runtime_v6 import evaluate

ROOT=Path(__file__).resolve().parent

class VersionedEvaluationTests(unittest.TestCase):
    def test_runtime_control_audit_and_termination_are_exact_after_policy_rename(self):
        frozen=(ROOT/'state_residual_eval_runtime_v5.py').read_bytes()
        self.assertEqual(hashlib.sha256(frozen).hexdigest(),'90a1ef7d5d8cfdebd5e05df5ec598d48cea7ae7bcfaf145a357d61a5e7727d64')
        changed=(ROOT/'state_residual_eval_runtime_v6.py').read_text()
        changed=changed.replace('from phase_gated_residual_policy_v6 import EpisodeActor, features as v6_features',
                                'from phase_residual_policy_v5 import EpisodeActor, features as v5_features')
        changed=changed.replace("== 'v6'","== 'v5'").replace("=='v6'","=='v5'").replace('v6_features(', 'v5_features(')
        self.assertEqual(changed,frozen.decode())

    def test_entry_launch_and_cleanup_ast_preserved(self):
        def main(path):
            tree=ast.parse(path.read_text().replace("choices=('v6','v4'),default='v6'","choices=('v5','v4'),default='v5'"))
            return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='main'),include_attributes=False)
        self.assertEqual(main(ROOT/'evaluate_state_residual_v6.py'),main(ROOT/'evaluate_state_residual_v5.py'))

    def test_real_v6_actor_routes_masked_then_active_with_actual_history(self):
        torch.manual_seed(42);policy=model();policy.head.bias.data.fill_(.5)
        runtime=FakeRuntime(success_step=3);runtime.teacher.name='squeeze'
        runtime.actor=EpisodeActor(policy,list(range(28)));runtime.actor.reset(runtime.args.name)
        def schedule(obs,before,after):
            runtime.current_schedule={'phase':after['name'],'phase_progress':(71/84,72/84,1.)[len(runtime.rows)]}
            return dict(runtime.current_schedule)
        runtime.schedule_resolver=schedule
        journal=io.StringIO()
        with patch.dict('sys.modules',{'joint_safety':joint_safety}):
            summary=evaluate(runtime,journal,max_steps=10)
        rows=[json.loads(line)for line in journal.getvalue().splitlines()if json.loads(line)['record_type']=='executed_transition']
        self.assertEqual(len(rows),3);self.assertTrue(summary['original_wrist_gate_qualified'])
        self.assertEqual(summary['left_teacher_position_action_steps'],0)
        replay=EpisodeActor(policy,list(range(28)));replay.reset(runtime.args.name)
        active=[]
        for step,row in enumerate(rows):
            obs=row['actor_observation_pre']|{'previous_executed_source':'student'}
            x=torch.from_numpy(features(obs)).reshape(1,1,-1)
            active.append(bool(policy.residual_active(x).item()))
            prediction=replay.act(obs,episode_id=runtime.args.name,step=step)
            np.testing.assert_array_equal(np.asarray(prediction,dtype=np.float32).view(np.int32),
                                          np.asarray(row['predicted_left_target'],dtype=np.float32).view(np.int32))
            if step:
                self.assertEqual(obs['applied_target'],rows[step-1]['actor_observation_post']['applied_target'])
            else:
                base=policy.baseline_targets(x,torch.tensor(obs['applied_target'][:28]).reshape(1,1,28))
                np.testing.assert_array_equal(base.detach().numpy().reshape(28).view(np.int32),
                                              np.asarray(prediction,dtype=np.float32).view(np.int32))
        self.assertEqual(active,[False,True,True])
        self.assertEqual(replay.next_step,3)

if __name__=='__main__':unittest.main()
