"""CPU-only tests of actual candidate AST; fake contact physics, real task lock."""
import ast, hashlib, io, json, sys, time
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from full_task_episode import FullTaskEpisode
from test_full_task_episode import FakeEnv, advance
from left_index_continuation import SETTINGS
from left_ring_support import RingSupport, JOINT as RING_JOINT, parameters

HERE=Path(__file__).resolve().parent

def candidate_functions(namespace):
    paths=[HERE/'opposed_grasp_runtime.py.index_continuation_candidate',HERE/'audit_opposed_index_candidate_v2.py']
    path=next(p for p in paths if p.exists())
    expected=json.loads((HERE/'index_continuation_runtime_manifest.json').read_text())['opposed_grasp_runtime.py']['candidate_sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise AssertionError('Candidate SHA differs from reviewed manifest')
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('execute','_execute')]
    if len(nodes)!=2:raise AssertionError('Missing actual runtime functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class IndexRuntimeTests(unittest.TestCase):
    def build(self,preflight_passed):
        env,journal=FakeEnv(maximum=3),io.StringIO()
        task=FullTaskEpisode(env,journal,{'task_id':'index_continuation_host','target_degrees':15,
            'source_sha256':{'candidate':'host_ast_only'},'physics_domain':'fake_no_simulator'})
        env.full_task_episode=task
        advance(task,env) # represents completed center history; zero new steps before this stage
        names=[SETTINGS['joint'],RING_JOINT]+[f'left_fixture_{i}' for i in range(26)]+[f'right_fixture_{i}' for i in range(28)]
        env.cfg={'joint_names':names};env.stage=None
        initial=np.zeros(56,dtype=np.float32);initial[0]=.0996;initial[1]=.5
        env.targets=torch.from_numpy(initial.copy()).reshape(1,56);env.scales=torch.ones(1,56)
        rows=[{'step':0,'phase':'center_hold','q':initial.tolist(),'nominal_target':initial.tolist(),
            'index_contact_feedback':{'offset_rad':-.0004,'next_offset_rad':-.00045},
            'finger_contact_points':[],'cube_position':[.5,0.,1.],'cube_quaternion_wxyz':[1.,0.,0.,0.]}]
        points=[initial.copy() for _ in range(4)]
        for i,p in enumerate(points):p[28:]=.2+i*.001
        plan={'passed':True,'checks':11,'phases':[('right_route',[p.tolist() for p in points])]}
        planner_calls=[];nominal_targets=[]
        def host_plan(script,payload,*rest):
            planner_calls.append((script,payload))
            if script=='opposed_grasp_plan.py':return plan
            self.assertEqual(script,'left_support_envelope_preflight_v2.py')
            self.assertEqual(payload['request']['initial'],dict(zip(names,initial.tolist())))
            self.assertEqual(payload['baseline_target'],initial.tolist())
            self.assertEqual(payload['plan'],plan)
            return {'passed':preflight_passed,'checks':22,'physics_validated':False}
        def append(env,teacher,fingers,audit,rows,run,phase,target):
            nominal_targets.append(target.copy())
            row={'step':len(rows),'phase':phase,**env.metrics,'q':target.tolist(),
                 'cube_position':[.5,0.,1.],'cube_quaternion_wxyz':[1.,0.,0.,0.],
                 'cubelets':{},'finger_contact_points':[]}
            rows.append(row);return row
        class Run:
            def __truediv__(self,name):return self
            def write_text(self,text):pass
            def open(self,mode):return io.StringIO()
        class Contacts:
            def __init__(self):self.points=[]
        topology={'unexpected_contacts':[],'fingers':dict.fromkeys(('thumb','index','middle','ring','little'),True)}
        ns={'np':np,'torch':torch,'json':json,'time':time,'host_plan':host_plan,'append_row':append,
            'safety_failure':lambda row:None,'configure_right_fingers':lambda *a:{'right_finger_stiffness':None},
            'OppositionContacts':Contacts,'right_contacts':lambda *a:topology,
            'measure_layers':lambda *a,**k:{'maximum_other_cubelet_rotation_deg':0.,'face_angle_deg':0.},
            'contact_topology':lambda *a:topology['fingers'],'qualified':lambda *a,**k:{'qualified':True},
            'frame':lambda *a:None,'RingSupport':RingSupport,'RING_JOINT':RING_JOINT,
            'ring_support_parameters':parameters,'ring_impulse':lambda row:0.}
        candidate_functions(ns)
        torso=ModuleType('torso_collision');torso.export_fixed_colliders=lambda *a:[]
        args=SimpleNamespace(push_degrees=15.,left_index_continuation=True,left_ring_support=True)
        audit=SimpleNamespace(reset_step=lambda:None)
        call_args=(env,None,audit,audit,rows,[],Run(),args,{'passed':True,'finger_contact_feedback':dict(SETTINGS)})
        return SimpleNamespace(env=env,task=task,rows=rows,ns=ns,torso=torso,args=call_args,
            planner_calls=planner_calls,targets=nominal_targets,initial=initial,points=points,journal=journal)

    def test_joint_preflight_rejection_executes_zero_new_steps_and_locks_task(self):
        f=self.build(False);before=len(f.env.actions)
        with patch.dict(sys.modules,{'torso_collision':f.torso}):
            result=f.ns['execute'](*f.args)
            f.ns['execute'](*f.args)
        self.assertEqual(result['failure'],'left_ring_support_preflight_rejected')
        self.assertEqual(len(f.env.actions),before)
        self.assertEqual(len(f.planner_calls),2)
        self.assertTrue(f.task.status()['done']);self.assertFalse(f.targets)
        with self.assertRaises(RuntimeError):f.task.step([[0.]*56],phase='right_route',step_id=1)
        self.assertEqual(len(f.env.actions),before)

    def test_direct_reference_plus_next_offset_updates_then_timeout_stops(self):
        f=self.build(True)
        with patch.dict(sys.modules,{'torso_collision':f.torso}):result=f.ns['execute'](*f.args)
        self.assertEqual(len(f.targets),2) # four planned points, timeout after second
        reference=float(f.initial[0])-(-.0004)
        expected=[np.float32(reference-.00045),np.float32(reference-.00050)]
        np.testing.assert_array_equal(np.asarray([t[0] for t in f.targets],dtype=np.float32),expected)
        self.assertEqual(f.rows[1]['left_index_continuation']['applied_offset_rad'],-.00045)
        self.assertAlmostEqual(f.rows[1]['left_index_continuation']['next_offset_rad'],-.0005)
        for i,target in enumerate(f.targets):
            np.testing.assert_array_equal(target[2:28],f.initial[2:28])
            np.testing.assert_array_equal(target[28:],f.points[i][28:])
        self.assertEqual(result['failure'],'timeout');self.assertTrue(f.task.status()['truncated'])
        self.assertEqual(f.task.steps,3);before=len(f.env.actions)
        with self.assertRaises(RuntimeError):f.task.step([[0.]*56],phase='right_route',step_id=3)
        with patch.dict(sys.modules,{'torso_collision':f.torso}):f.ns['execute'](*f.args)
        self.assertEqual(len(f.env.actions),before)

if __name__=='__main__':unittest.main()
