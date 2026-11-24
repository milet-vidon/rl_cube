import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import contact_hold_continuous as loop
import contact_hold_loop_data as data
from contact_hold_loop_fit import verify_collection


def row(i,phase='squeeze'):
    return dict(phase=phase,phase_elapsed=i/300,applied_target=[0.]*56,q=[0.]*56,joint_velocity=[0.]*56,
                cube_position=[0.,0.,1.],cube_quaternion_wxyz=[1.,0.,0.,0.],finger_contact_points=[],
                drop=False,self_contact_impulse=0,robot_table_impulse=0,arm_soft_limit_hit=False,
                actual_arm_soft_limit_violation=False,hard_limit_hit=[False]*56,soft_limit_hit=[False]*56,
                layer_measurement={'maximum_relative_rotation_deg':0},student_left_target=[0.]*28,
                student_raw_left_target=[0.]*28,teacher_left_requested_not_executed=[.1]*28)


class LoopTests(unittest.TestCase):
    def test_project_only_safe_preterminal_labeled_joints(self):
        rows=[row(i) for i in range(9)]
        rows[2]['teacher_left_requested_not_executed'][12]=9
        rows[5]['robot_table_impulse']=.1
        bounds=np.tile([-1.,1.],(28,1))
        spec=dict(bounds=bounds,step=[.025]*28,velocity=[10.]*28,dt=1/30)
        a=data.transitions(rows,spec,list(range(28)),bounds)
        self.assertEqual(a['steps'].tolist(),[1,2,3,4]);self.assertFalse(a['mask'][1,12])
        np.testing.assert_allclose(a['y'],.025)
        rows[2]['student_raw_left_target'][4]=.1
        with self.assertRaises(AssertionError): data.transitions(rows,spec,list(range(28)),bounds)

    def test_holdouts_never_ingested(self):
        with self.assertRaises(ValueError): loop.validate(loop.DEFAULT|dict(validation_seeds=[11601]))
        for role in ['validation','test']:
            with self.assertRaises(ValueError): data.collect(dict(role=role,seed=11601),None,[11601])

    def test_worker_rejects_validation_relabeled_in_plan(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); a=p/'samples';a.write_bytes(b'test')
            data.write(p/'student_evaluation.json',dict(evaluation_role='validation'))
            data.write(p/'artifact_manifest.json',dict(split='development',full_episode=True,
                         sha256={'student_evaluation.json':data.sha(p/'student_evaluation.json')}))
            fake=dict(path=str(a),sha256=data.sha(a),source=dict(role='collection',seed=11601,run=str(p),
                      manifest=data.sha(p/'artifact_manifest.json')))
            with self.assertRaisesRegex(ValueError,'declared collection'):
                verify_collection(fake,[11601])

    def test_plateau_vs_two_consecutive_successes(self):
        cfg=loop.DEFAULT|dict(patience=2)
        state=dict(round=0,best_score=[0,2,0],stale=0,streak=0,champion='original')
        for _ in range(2): loop.advance(state,dict(score=[0,1,0],rate=0),'worse',cfg)
        self.assertEqual((state['status'],state['champion']),('plateau','original'))
        for _ in range(2): loop.advance(state,dict(score=[2,9,60],rate=1),'better',cfg)
        self.assertEqual((state['status'],state['champion']),('final_test','better'))

    def test_interrupt_after_fit_and_resume_to_independent_test(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);data.write(home/'config.json',dict(implementation={}))
            cfg=loop.DEFAULT|dict(confirmations=2,validation_seeds=[31],test_seeds=[41])
            state=dict(round=0,streak=0,stale=0,best_score=None,champion=str(data.INITIAL),
                       current=str(data.INITIAL),corpus=[],status='running',stage='baseline')
            events=[]; interruption=[True]
            def panel(path,fit,seeds,role):
                events.append((path.name,role))
                if path.name=='validation' and interruption[0]:
                    interruption[0]=False;raise KeyboardInterrupt
                good=path.name!='baseline'
                return dict(rate=float(good),score=[int(good),2,0],phases={},episodes=[])
            def fit(plan):
                marker=plan.parent/'result.json'
                if not marker.exists(): events.append(('update',None));marker.touch()
            def eval_(slot,fit,seed,role):
                return dict(checkpoint=str(fit),seed=seed,role=role)
            with (patch.object(loop,'run_panel',side_effect=panel),patch.object(loop,'fit_block',side_effect=fit),
                  patch.object(loop,'evaluate',side_effect=eval_),patch.object(loop,'checkpoint',return_value='hash'),
                  patch.object(loop,'ensure_unseen'),patch.object(loop,'collect',side_effect=lambda r,*_:dict(source=r))):
                with self.assertRaises(KeyboardInterrupt): loop.drive(home,state,cfg,3)
                loop.drive(home,data.read(home/'state.json'),cfg,3)
            self.assertEqual(data.read(home/'state.json')['status'],'confirmed_on_test')
            self.assertEqual(events.count(('update',None)),2)
            self.assertEqual(events.count(('final','test')),1)
