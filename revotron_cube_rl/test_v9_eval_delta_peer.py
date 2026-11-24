"""Independent V9 evaluator delta checks; no simulator or trained checkpoint."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import hashlib,unittest
import state_arm_hold_increment_eval_runtime_v9 as runtime
import arm_hold_increment_policy_v9 as policy
from test_arm_hold_increment_policy_v9_peer import params


class V9EvaluationDeltaPeer(unittest.TestCase):
    def test_inverse_routing_and_readonly_init_block_restore_full_v8_bytes(self):
        raw=Path(runtime.__file__).read_bytes();source=raw.decode()
        begin=source.index('        actual_parameters = self.read_projection_parameters()')
        end=source.index('        self.actor = ',begin)
        source=source[:begin]+source[end:]
        source=source.replace('from arm_hold_increment_policy_v9 import EpisodeActor, features as v9_features, verify_projection_arm_bounds',
            'from hold_increment_policy_v8 import EpisodeActor, features as v8_features')
        source=source.replace("== 'v9'","== 'v8'").replace("=='v9'","=='v8'").replace('v9_features(', 'v8_features(')
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),'8c3c9459254c12b0b992889bd95aea52bf3a12d4143fd40731641a70ee46072c')

    def instance(self):
        p=params();names=list(policy.EXPECTED_LEFT_NAMES)+['right_'+str(i)for i in range(28)];p['joint_names']=names
        r=runtime.EvaluationRuntime.__new__(runtime.EvaluationRuntime)
        r.physics_identity={};r.env=SimpleNamespace(cfg={'joint_names':names})
        r.left_indices=list(range(28));r.policy_metadata={'joint_names':names,'output_indices':r.left_indices}
        r.baseline_metadata=deepcopy(r.policy_metadata);r.projection_parameters=deepcopy(p)
        r.read_projection_parameters=Mock(return_value=p);r.policy_model=object()
        r.args=SimpleNamespace(name='host',policy='v9');r.write=Mock()
        proof=policy.verify_projection_arm_bounds(p,r.left_indices)
        r.evaluation_identity={'expected_physics_family_sha256':'family','reviewed_original_arm_projection_proofs':[deepcopy(proof)for _ in range(9)]}
        return r

    def invoke(self,r):
        with patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'physics_fingerprint',return_value='family'),patch.object(runtime,'EpisodeActor')as actor:
            r._initialize();return actor

    def test_actual_parameters_read_and_verified_before_actor_reset(self):
        r=self.instance();actor=self.invoke(r)
        r.read_projection_parameters.assert_called_once();actor.return_value.reset.assert_called_once_with('host')
        self.assertEqual(r.evaluation_identity['actual_original_arm_projection_proof'],r.evaluation_identity['reviewed_original_arm_projection_proofs'][0])
        self.assertEqual(r.evaluation_identity['actor_hidden_reset_count'],1)

    def test_changed_actual_cap_or_any_history_proof_rejects_before_actor(self):
        for kind in ('actual_parameters','reviewed_proof','proof_count'):
            with self.subTest(kind=kind):
                r=self.instance()
                if kind=='actual_parameters':r.read_projection_parameters.return_value['command_velocity_limits'][0][0]=.5
                elif kind=='reviewed_proof':r.evaluation_identity['reviewed_original_arm_projection_proofs'][5]['original_projection_cap_rad'][0]=.024
                else:r.evaluation_identity['reviewed_original_arm_projection_proofs'].pop()
                with patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'physics_fingerprint',return_value='family'),patch.object(runtime,'EpisodeActor')as actor:
                    with self.assertRaises(ValueError):r._initialize()
                    actor.assert_not_called()


if __name__=='__main__':unittest.main()
