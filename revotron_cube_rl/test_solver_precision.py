import unittest
from types import SimpleNamespace
from solver_precision import position_iterations, verify_solver

class SolverPrecisionTests(unittest.TestCase):
    def test_only_position_settings_changed_and_restored_on_failure(self):
        original = lambda **kw: kw
        module = SimpleNamespace(physx_config=original)
        baseline = dict(solver_type=1, min_position_iteration_count=32,
                        max_position_iteration_count=64, min_velocity_iteration_count=4)
        with self.assertRaisesRegex(RuntimeError, 'init failed'):
            with position_iterations(64, module):
                got = module.physx_config(**baseline)
                self.assertEqual(got, baseline | {'min_position_iteration_count': 64})
                raise RuntimeError('init failed')
        self.assertIs(module.physx_config, original)

    def test_disabled_is_noop_and_invalid_rejected(self):
        module = SimpleNamespace(physx_config=object())
        original = module.physx_config
        with position_iterations(None, module):
            self.assertIs(module.physx_config, original)
        with self.assertRaises(ValueError):
            with position_iterations(16, module): pass
        self.assertIs(module.physx_config, original)

    def test_verification_rejects_unapplied_settings(self):
        attrs = {'solverType': 'TGS', 'minPositionIterationCount': 32,
                 'maxPositionIterationCount': 64, 'minVelocityIterationCount': 4,
                 'enableExternalForcesEveryIteration': True, 'enableStabilization': False}
        prim = SimpleNamespace(GetAttribute=lambda n: SimpleNamespace(Get=lambda: attrs[n.split(':')[1]]))
        env = SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda _: prim),
                              cfg={'physics_hz': 240, 'control_hz': 30})
        with self.assertRaisesRegex(ValueError, 'not applied'): verify_solver(env, 64)
        attrs['minPositionIterationCount'] = 64
        self.assertTrue(verify_solver(env, 64)['verified'])

if __name__ == '__main__': unittest.main()
