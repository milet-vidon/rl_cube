"""Host control-flow checks against the actual installed stage runtimes."""
import ast
import io
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from full_task_episode import FullTaskEpisode
from test_full_task_episode import FakeEnv, advance

HERE = Path(__file__).resolve().parent


def functions_from_runtime(name, names, namespace):
    source = ast.parse((HERE / (name + '.py')).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names): raise AssertionError('Missing actual candidate functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HERE / name), 'exec'), namespace)
    return namespace


def fake_episode():
    env, journal = FakeEnv(), io.StringIO()
    task = FullTaskEpisode(env, journal, {'task_id': '15deg_test', 'target_degrees': 15,
        'source_sha256': {'driver.py': 'host_fixture'}, 'physics_domain': 'host_fake_only'})
    env.full_task_episode = task
    return env, task, journal


class ActualDraftControlFlowTests(unittest.TestCase):
    def test_actual_main_loop_interruption_inside_teacher_query_prevents_step(self):
        env, task, journal = fake_episode()
        module = ast.parse((HERE / 'single_face_experiment.py').read_text())
        loop = next(node for node in ast.walk(module) if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == 'range(a.max_steps)')
        calls = []
        ns = {'task': task, 'env': env, 'rows': [], 'failure': None, 'stop_requested': False,
              'a': SimpleNamespace(max_steps=10), 'fingers': SimpleNamespace(reset_step=lambda: None),
              'audit': SimpleNamespace(reset_step=lambda: None)}
        def action():
            calls.append(1)
            ns['stop_requested'] = True
            return [[0.] * 56]
        ns['teacher'] = SimpleNamespace(action=action)
        exec(compile(ast.Module(body=[loop], type_ignores=[]), '<actual-main-loop>', 'exec'), ns)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(env.actions), 0)
        self.assertEqual(task.status()['reason'], 'execution_interrupted')
        self.assertTrue(task.status()['truncated'])

    def test_both_stage_wrappers_abort_preflight_rejection_without_fake_step(self):
        for module, is_right in (('centered_opposition_runtime', False), ('opposed_grasp_runtime', True)):
            env, task, journal = fake_episode()
            calls = []
            def body(*args):
                calls.append(1)
                return {'executed': False, 'qualified': False, 'failure': 'preflight_rejected'}
            ns = functions_from_runtime(module, {'execute'}, {'_execute': body})
            args = [env, None, None, None, [], [], None, None]
            if is_right: args.append({})
            result = ns['execute'](*args)
            self.assertEqual(result['failure'], 'preflight_rejected')
            self.assertEqual(task.status()['reason'], 'preflight_rejected')
            self.assertEqual(len(env.actions), 0)
            self.assertNotIn('"record_type": "transition"', journal.getvalue())
            ns['execute'](*args)
            self.assertEqual(len(calls), 1)  # stage refuses to re-enter its body

    def test_actual_center_loop_stops_at_raw_drop_even_when_stage_audit_reports_safe(self):
        self._run_center_case(False)

    def test_interruption_during_center_preflight_adds_no_physics_step(self):
        self._run_center_case(True)

    def _run_center_case(self, interrupt_in_plan):
        env, task, journal = fake_episode()
        advance(task, env)  # represents the preceding left-grasp step
        env.outcomes = [{'drop': True}, {}, {}]
        names = [f'Joint{i}_L' for i in range(28)] + [f'right_{i}' for i in range(28)]
        env.cfg = {'joint_names': names}
        env.targets, env.scales, env.stage = torch.zeros(1, 56), torch.ones(1, 56), None
        rows = [{'step': 0, 'phase': 'tilt_hold', 'q': [0.] * 56,
                 'cube_position': [.5, 0., 1.], 'cube_quaternion_wxyz': [1., 0., 0., 0.]}]
        plan = {'passed': True, 'checks': {}, 'center_target_world_m': [.5, 0., 1.],
                'cube_target_quaternion_wxyz': [1., 0., 0., 0.],
                'phases': [('left_center_transport', [[0.] * 56] * 3)]}
        def append(env, teacher, fingers, audit, rows, run, phase, target):
            row = {'step': len(rows), 'phase': phase, **env.metrics, 'q': [0.] * 56,
                   'cube_position': [.5, 0., 1.], 'layer_measurement': {'maximum_relative_rotation_deg': 0.},
                   'finger_contact_points': []}
            rows.append(row)
            return row
        class Run:
            def __truediv__(self, name): return self
            def write_text(self, text): pass
        torso = ModuleType('torso_collision'); torso.export_fixed_colliders = lambda *a: []
        interrupted = [False]
        def host_plan(*args):
            interrupted[0] = interrupt_in_plan
            return plan
        ns = {'np': np, 'torch': torch, 'R': Rotation, 'json': __import__('json'),
              'host_plan': host_plan, 'append_row': append, 'safety_failure': lambda row: None,
              'contact_topology': lambda *a: dict.fromkeys(('thumb', 'index', 'middle', 'ring', 'little'), True),
              'assess_center_row': lambda *a: {'qualified': True}, 'frame': lambda *a: None}
        functions_from_runtime('centered_opposition_runtime', {'execute', '_execute'}, ns)
        args = SimpleNamespace(center_style='legacy', center_only=False)
        audit = SimpleNamespace(reset_step=lambda: None)
        with patch.dict(sys.modules, {'torso_collision': torso}):
            result = ns['execute'](env, None, audit, audit, rows, [], Run(), args, lambda: interrupted[0])
        count = 1 if interrupt_in_plan else 2
        self.assertEqual(len(env.actions), count)
        self.assertEqual(task.steps, count)
        self.assertEqual(result['failure'], 'execution_interrupted' if interrupt_in_plan else 'cube_drop')
        self.assertFalse(result['qualified'])
        self.assertTrue(task.status()['done'])
        if interrupt_in_plan:
            self.assertTrue(task.status()['truncated'])
        else:
            self.assertTrue(rows[-1]['task_episode']['done'])

    def test_main_opt_in_rejects_unadapted_index_and_partial_task_branches(self):
        source = (HERE / 'single_face_experiment.py').read_text()
        module = ast.parse(source)
        flag_check = next(node for node in module.body if isinstance(node, ast.If)
                          and 'a.full_task_episode' in ast.unparse(node.test))
        class Parser:
            def error(self, text): raise ValueError(text)
        code = compile(ast.Module(body=[flag_check], type_ignores=[]), '<actual-cli-guard>', 'exec')
        for controller, center, degrees, rejected in (
                ('index', False, 15, True), ('opposed', True, 15, True),
                ('opposed', False, 0, True), ('opposed', False, 15, False),
                ('opposed', False, 30, False), ('opposed', False, 90, False)):
            scope = {'a': SimpleNamespace(full_task_episode=True, controller=controller,
                                         center_only=center, push_degrees=degrees), 'p': Parser()}
            if rejected:
                with self.assertRaises(ValueError): exec(code, scope)
            else: exec(code, scope)

    def test_actual_opposed_loop_stops_at_timeout_before_later_path_points(self):
        env, journal = FakeEnv(maximum=2), io.StringIO()
        task = FullTaskEpisode(env, journal, {'task_id': '15deg_test', 'target_degrees': 15,
            'source_sha256': {'driver.py': 'host_fixture'}, 'physics_domain': 'host_fake_only'})
        env.full_task_episode = task
        advance(task, env)
        names = [f'Joint{i}_L' for i in range(28)] + [f'right_{i}' for i in range(28)]
        env.cfg = {'joint_names': names}
        env.targets, env.scales, env.stage = torch.zeros(1, 56), torch.ones(1, 56), None
        rows = [{'step': 0, 'phase': 'center_hold', 'q': [0.] * 56, 'finger_contact_points': [],
                 'cube_position': [.5, 0., 1.], 'cube_quaternion_wxyz': [1., 0., 0., 0.]}]
        plan = {'passed': True, 'checks': {}, 'phases': [('right_route', [[0.] * 56] * 3)]}
        def append(env, teacher, fingers, audit, rows, run, phase, target):
            row = {'step': len(rows), 'phase': phase, **env.metrics, 'q': [0.] * 56,
                   'cube_position': [.5, 0., 1.], 'cube_quaternion_wxyz': [1., 0., 0., 0.],
                   'cubelets': {}, 'finger_contact_points': []}
            rows.append(row)
            return row
        class Run:
            def __truediv__(self, name): return self
            def write_text(self, text): pass
            def open(self, mode): return io.StringIO()
        class Contacts:
            def __init__(self): self.points = []
        topology = {'unexpected_contacts': [], 'fingers': dict.fromkeys(('thumb','index','middle','ring','little'), True)}
        torso = ModuleType('torso_collision'); torso.export_fixed_colliders = lambda *a: []
        ns = {'np': np, 'torch': torch, 'json': __import__('json'), 'time': __import__('time'),
              'host_plan': lambda *a: plan, 'append_row': append, 'safety_failure': lambda row: None,
              'configure_right_fingers': lambda *a: {'right_finger_stiffness': None},
              'OppositionContacts': Contacts, 'right_contacts': lambda *a: topology,
              'measure_layers': lambda *a, **kw: {'maximum_other_cubelet_rotation_deg': 0., 'face_angle_deg': 0.},
              'contact_topology': lambda *a: topology['fingers'],
              'qualified': lambda *a, **kw: {'qualified': True}, 'frame': lambda *a: None}
        functions_from_runtime('opposed_grasp_runtime', {'execute', '_execute'}, ns)
        args = SimpleNamespace(push_degrees=15.)
        audit = SimpleNamespace(reset_step=lambda: None)
        with patch.dict(sys.modules, {'torso_collision': torso}):
            result = ns['execute'](env, None, audit, audit, rows, [], Run(), args, {'passed': True})
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(task.steps, 2)
        self.assertEqual(result['failure'], 'timeout')
        self.assertFalse(result['qualified'])
        self.assertTrue(rows[-1]['task_episode']['truncated'])


if __name__ == '__main__': unittest.main()
