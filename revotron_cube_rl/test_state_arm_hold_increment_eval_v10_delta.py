"""V10 routing inverse and explicit simulation mode; no real simulator."""
import ast,io,json,hashlib,sys,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import evaluate_state_arm_hold_increment_v10 as entry
import state_arm_hold_increment_eval_runtime_v10 as runtime
import test_state_arm_hold_increment_eval_v9_delta as old_tests

ROOT=Path(__file__).parent


class V10RouteContracts(unittest.TestCase):
    def test_runtime_inverse_full_text_preserves_frozen_execution(self):
        old=(ROOT/'state_arm_hold_increment_eval_runtime_v9.py').read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),'27e69ce8c204811b3aa44e86fa84155b3d35fd591778a886ed3dcd015d4006b7')
        new=(ROOT/'state_arm_hold_increment_eval_runtime_v10.py').read_text()
        inverse=new.replace('v10_features','v9_features').replace("'v10'","'v9'").replace('V10','V9').replace('len(reviewed_proofs) != 11','len(reviewed_proofs) != 9')
        self.assertEqual(inverse,old.decode())
        self.assertIs(runtime.EpisodeActor,entry.arm_hold_increment_policy_v9.EpisodeActor)

    def test_main_inverse_retains_execution_exception_and_cleanup(self):
        old=(ROOT/'evaluate_state_arm_hold_increment_v9.py').read_text()
        new=(ROOT/'evaluate_state_arm_hold_increment_v10.py').read_text()
        new=new.replace("'v10'","'v9'").replace("    mode=parser.add_mutually_exclusive_group()\n    mode.add_argument('--preflight-only',action='store_true')\n    mode.add_argument('--evaluate',action='store_true',help='Explicitly authorize the externally scheduled single simulator evaluation')", "    parser.add_argument('--preflight-only',action='store_true')").replace('        if not args.evaluate:','        if args.preflight_only:')
        def main(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='main'),include_attributes=False)
        self.assertEqual(main(old),main(new))

    def test_actual_projection_initialization_requires_all_eleven_proofs_before_actor(self):
        case=old_tests.V9RouteContracts()
        def fixture():
            old=case.make_initialization();obj=runtime.EvaluationRuntime.__new__(runtime.EvaluationRuntime)
            obj.__dict__.update(old.__dict__);obj.args.policy='v10'
            obj.read_projection_parameters=lambda:deepcopy(obj.projection_parameters)
            return obj
        obj=fixture()
        obj.evaluation_identity['reviewed_original_arm_projection_proofs']+=deepcopy(obj.evaluation_identity['reviewed_original_arm_projection_proofs'][:2])
        actor=Mock()
        with patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'EpisodeActor',return_value=actor):
            runtime.EvaluationRuntime._initialize(obj)
        actor.reset.assert_called_once_with(obj.args.name)
        for count in (9,10):
            obj=fixture()
            if count==10:obj.evaluation_identity['reviewed_original_arm_projection_proofs'].append(deepcopy(obj.evaluation_identity['reviewed_original_arm_projection_proofs'][0]))
            with self.subTest(count=count),patch.object(runtime.BaselineRuntime,'_initialize'),patch.object(runtime,'EpisodeActor')as invalid:
                with self.assertRaisesRegex(ValueError,'caps differ'):runtime.EvaluationRuntime._initialize(obj)
                invalid.assert_not_called()

    def invoke_main(self,extra):
        with tempfile.TemporaryDirectory(dir=ROOT)as folder:
            output=Path(folder)/'evaluation';output.mkdir()
            args=['eval','--checkpoint','model','--expected-checkpoint-sha256','hash','--baseline','base',
                  '--qualification','q','--qualification-sha256','qhash','--authorization','auth',
                  '--authorization-sha256','ahash','--dataset','run','--output',str(output),*extra]
            fake=Mock();fake.finalize_evaluation.return_value={'synthetic':True}
            with patch.object(sys,'argv',args),patch.object(entry,'prepare',return_value=({},None,{}, {},{})),\
                 patch.object(entry,'EvaluationRuntime',return_value=fake)as constructor,patch.object(entry,'evaluate',return_value={})as evaluate,\
                 patch.object(entry.signal,'signal'),patch('sys.stdout',new=io.StringIO()):
                entry.main()
            return constructor,evaluate,fake

    def test_default_and_explicit_preflight_do_not_construct_runtime(self):
        for extra in ([],['--preflight-only']):
            constructor,evaluate,fake=self.invoke_main(extra)
            constructor.assert_not_called();evaluate.assert_not_called();fake.close.assert_not_called()

    def test_explicit_evaluate_constructs_and_closes_once(self):
        constructor,evaluate,fake=self.invoke_main(['--evaluate'])
        constructor.assert_called_once();evaluate.assert_called_once();fake.close.assert_called_once()

    def test_eval_and_preflight_flags_are_mutually_exclusive(self):
        with patch('sys.stderr',new=io.StringIO()),self.assertRaises(SystemExit):self.invoke_main(['--evaluate','--preflight-only'])


if __name__=='__main__':unittest.main()
