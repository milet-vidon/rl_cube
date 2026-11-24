"""Independent V10 entry checks, no checkpoint inference or simulator."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from copy import deepcopy
import evaluate_state_arm_hold_increment_v10 as entry
import state_arm_hold_increment_eval_runtime_v10 as runtime
import test_state_arm_hold_increment_eval_v10_prepare as prepare_tests
import test_state_arm_hold_increment_eval_v9_delta as v9_tests

class IndependentBoundary(unittest.TestCase):
    def test_source_snapshot_extra_dependency_rejected(self):
        fixture=prepare_tests.V10PrepareContracts()
        with self.assertRaises(ValueError):fixture.invoke(lambda q,m,a,p:m['source_sha256'].update(unreviewed_dependency_py='0'*64))

    def test_qualified_flag_cannot_be_truthy_integer(self):
        fixture=prepare_tests.V10PrepareContracts()
        with self.assertRaises(ValueError):fixture.invoke(lambda q,m,a,p:m.update(offline_candidate_qualified=1))

    def test_last_of_eleven_actual_projection_proofs_must_match_before_actor(self):
        old=v9_tests.V9RouteContracts().make_initialization();obj=runtime.EvaluationRuntime.__new__(runtime.EvaluationRuntime)
        obj.__dict__.update(old.__dict__);obj.args.policy='v10'
        obj.read_projection_parameters=lambda:deepcopy(obj.projection_parameters)
        proofs=obj.evaluation_identity['reviewed_original_arm_projection_proofs']
        proofs.extend(deepcopy(proofs[:2]));proofs[-1]['original_projection_changed']=True
        with patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'EpisodeActor')as actor:
            with self.assertRaisesRegex(ValueError,'caps differ'):runtime.EvaluationRuntime._initialize(obj)
            actor.assert_not_called()

    def test_snapshot_is_exact_and_cannot_overwrite_existing_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            root=Path(directory);source=root/'source';destination=root/'snapshot'
            source.write_bytes(b'first original');entry.snapshot_bytes(source,destination)
            source.write_bytes(b'changed')
            with self.assertRaises(FileExistsError):entry.snapshot_bytes(source,destination)
            self.assertEqual(destination.read_bytes(),b'first original')

if __name__=='__main__':unittest.main()
