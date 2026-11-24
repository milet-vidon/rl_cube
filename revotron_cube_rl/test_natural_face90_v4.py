import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from unittest.mock import patch
import test_natural_face90_controller as baseline_controller_tests
import test_natural_face90_long as baseline_search_tests
import natural_face90_v4_long as search
from test_natural_face90_controller import Plant
from natural_face90_v4_controller import NaturalFace90Controller
from natural_face90_v4_contacts import pair_rows,PairRecorder
from natural_face90_v4_clearance import ForearmClearance,LOWER,UPPER,RADIUS,CLEARANCE
from kinematics import RobotKinematics

class Attribution(unittest.TestCase):
    def test_symmetric_impulses_and_gapped_native_packet(self):
        a='/World/envs/env_0000/Robot/right_index_DIP_link'
        b='/World/envs/env_0000/Robot/left_thumb_DIP_link'
        rows=pair_rows([a,b],[1,1],[2,0],np.array([11,0,22]),np.array([.04,999,.04]),{11:a,22:b})
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(next(iter(rows.values())),.04)
    def test_cross_lane_and_bad_span_fail_closed(self):
        a='/World/envs/env_0000/Robot/Link7_R'
        for counts,starts,lookup in (([1],[0],{11:'/World/envs/env_0001/Robot/Link7_L'}),
                                    ([2],[0],{11:a})):
            with self.assertRaises(ValueError):pair_rows([a],counts,starts,np.array([11]),np.array([.01]),lookup)
    def test_table_contact_not_mislabelled_self_collision(self):
        a='/World/envs/env_0000/Robot/Link7_R'
        self.assertEqual(pair_rows([a],[1],[0],[11],[.04],{11:'/World/envs/env_0000/Table'}),{})
    def test_nonfinite_is_error(self):
        with self.assertRaises(ValueError):
            pair_rows(['/World/envs/env_0000/Robot/A'],[1],[0],[3],[float('nan')],{3:'/World/envs/env_0000/Robot/B'})
    def test_incomplete_sensor_coverage_fails(self):
        with self.assertRaises(ValueError):
            pair_rows(['/World/envs/env_0000/Robot/A'],[],[],[],[],{})
    def test_accumulates_native_substeps_and_clears_without_changing_gate(self):
        a='/World/envs/env_0000/Robot/A';b='/World/envs/env_0000/Robot/B'
        packet=[torch.tensor([.04,.04],dtype=torch.float64),None,None,None,
            torch.tensor([1,1]),torch.tensor([0,1]),torch.tensor([22,11])]
        view=SimpleNamespace(sensor_paths=[a,b],max_contact_data_count=8,get_raw_contact_data=lambda dt:packet)
        scene=SimpleNamespace(n=1,home=torch.zeros(1,56),contacts=torch.zeros(1,13,dtype=torch.float64),
            contact_views=[('robot',view,[0,0])])
        scene.reducer=SimpleNamespace(clear=lambda s:s.contacts.zero_(),
            update=lambda s:s.contacts[:,10].add_(.04),validate=lambda:None)
        recorder=PairRecorder(scene);recorder.lookup={11:a,22:b}
        with patch.dict('sys.modules',{'warp':SimpleNamespace()}):
            recorder.update(scene);recorder.update(scene)
        event=recorder.lane(0,.08)
        self.assertEqual(len(event),1);self.assertAlmostEqual(event[0]['impulse_Ns'],.08)
        self.assertEqual((event[0]['first_substep'],event[0]['last_substep']),(1,2))
        with self.assertRaises(ValueError):recorder.lane(0,.09)
        recorder.clear(scene);self.assertEqual(recorder.lane(0,0.),[])
        self.assertEqual(float(scene.contacts[:,10].sum()),0.)

class Control(unittest.TestCase):
    def setUp(self):
        self.e=Plant(2)
        self.c=NaturalFace90Controller(self.e,torch.zeros(2,6))
        self.active=torch.tensor([True,False]);f=torch.zeros(2,dtype=torch.bool);t=~f
        self.obs=dict(natural_ready=t,right_contact=f,wrong_contact=f,face_deg=torch.zeros(2),
                      structure_ok=t,support=t,contact_goal_reached=f)
        self.target=self.e.home.clone()
        self.c.begin(self.active,self.target,self.obs)
    def test_shape_at_standby_and_freeze_inactive_lane(self):
        self.c.phase[0]=1
        before=self.e.pos.clone()
        req,*_=self.c.action(self.target,self.active,self.obs)
        self.assertFalse(self.c.right_ready[0])
        self.assertLessEqual(float((req[:,self.c.fIDs]-self.target[:,self.c.fIDs]).abs().max()),.011001)
        torch.testing.assert_close(req[1],self.target[1])
        torch.testing.assert_close(self.e.pos,before)
    def test_ready_shape_needs_measured_settling(self):
        self.c.phase[0]=1
        for _ in range(60):self.c.action(self.target,self.active,self.obs)
        self.assertFalse(self.c.right_ready[0])  # frozen actual fingers did not close
        self.e.pos[0,self.c.fIDs]=self.c.fGoal.float()
        for _ in range(9):self.c.action(self.target,self.active,self.obs)
        self.assertFalse(self.c.right_ready[0])
        self.c.action(self.target,self.active,self.obs)
        self.assertTrue(self.c.right_ready[0])
    def test_wrong_contact_never_commands_more_motion(self):
        self.obs['wrong_contact'][0]=True
        req,retreat,failed,_=self.c.action(self.target,self.active,self.obs)
        self.assertTrue(failed[0]);self.assertFalse(retreat[0])
        torch.testing.assert_close(req,self.target)
    def test_blocked_arm_cannot_accumulate_distant_reference(self):
        self.c.phase[0]=2
        for _ in range(100):
            self.target,*_=self.c.action(self.target,self.active,self.obs)
        self.assertLessEqual(float((self.c.right_ref[0]-self.e.pos[0,self.c.rp.ix]).abs().max()),.060001)


class ControllerContracts(baseline_controller_tests.TestNaturalFaceController):
    """Run the same independent kinematic/contact-gating contracts on V4."""
    def setUp(self):
        p=patch.object(baseline_controller_tests,'NaturalFace90Controller',NaturalFace90Controller)
        p.start();self.addCleanup(p.stop)
        super().setUp()
    def test_r27_measured_endpoint_can_present_and_reach_selected_face(self):
        # The new forward detour has its own time budget; keep all original
        # kinematic reach, no-cube-cutting and independent natural-pose assertions.
        import joint_safety
        original=joint_safety.project_target;trace=[]
        guard=ForearmClearance(RobotKinematics(),self.plant.names,'cpu')
        ix=[self.plant.names.index(f'Joint{i}_R') for i in range(1,8)]
        def capture(*a,**kw):
            q,b=original(*a,**kw);trace.append(guard.measure(q[:,ix].double())[0]);return q,b
        with patch.object(baseline_controller_tests,'range',lambda n:range(900 if n==270 else n),create=True),patch.object(joint_safety,'project_target',capture):
            super().test_r27_measured_endpoint_can_present_and_reach_selected_face()
        distances=torch.stack(trace);established=torch.zeros(3,dtype=torch.bool)
        for d in distances:
            self.assertFalse((established&(d<CLEARANCE-2e-6)).any())
            established|=(d>=CLEARANCE)
        self.assertTrue(established.all())


class ClearanceRegression(unittest.TestCase):
    def setUp(self):
        self.data=json.loads(Path(__file__).with_name('natural_face90_v4_regression.json').read_text())
        self.names=self.data['joint_names'];self.guard=ForearmClearance(RobotKinematics(),self.names,'cpu')
        self.ix=[self.names.index(f'Joint{i}_R') for i in range(1,8)]
    def test_all_six_measured_collision_postures_are_excluded(self):
        q=torch.tensor([[e['q'][i] for i in self.ix] for e in self.data['incidents']],dtype=torch.float64)
        clearance,_=self.guard.measure(q)
        self.assertEqual(len(q),6)
        self.assertTrue((clearance<0).all())
    def test_step_toward_recorded_collision_cannot_reduce_clearance(self):
        a=torch.tensor([[e['before'][i] for i in self.ix] for e in self.data['incidents']],dtype=torch.float64)
        b=torch.tensor([[e['q'][i] for i in self.ix] for e in self.data['incidents']],dtype=torch.float64)
        # These are already close states from the last 12 measured frames;
        # the limiter may hold or move away but must not worsen the envelope.
        projected=self.guard.project(a,b)
        self.assertTrue((self.guard.measure(projected)[0]>=self.guard.measure(a)[0]-1e-8).all())
    def test_clearance_gradient_against_finite_differences(self):
        q=torch.tensor([[e['q'][i] for i in self.ix] for e in self.data['incidents']],dtype=torch.float64)
        _,gradient=self.guard.measure(q);numerical=torch.zeros_like(q)
        for j in range(7):
            delta=torch.zeros_like(q);delta[:,j]=1e-6
            numerical[:,j]=(self.guard.measure(q+delta)[0]-self.guard.measure(q-delta)[0])/2e-6
        torch.testing.assert_close(gradient,numerical,rtol=1e-5,atol=1e-7)
    def test_envelopes_enclose_the_actual_collision_mesh_vertices(self):
        root=Path(__file__).resolve().parent.parent/'revotron_revo3/source/revotron_system/meshes/arm/collision'
        def vertices(name):
            data=(root/name).read_bytes();n=int.from_bytes(data[80:84],'little')
            dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
            self.assertEqual(len(data),84+n*50)
            return np.frombuffer(data,dtype=dtype,offset=84,count=n)['vertices'].reshape(-1,3).astype(float)
        waist=vertices('waist_pitch_Link.STL')
        self.assertTrue((waist>=np.array(LOWER)).all() and (waist<=np.array(UPPER)).all())
        arm=vertices('Link5_R.STL');samples=self.guard.points[:,2].numpy()
        dz=np.min(np.abs(arm[:,2,None]-samples),axis=1)
        self.assertLessEqual(float(np.sqrt((arm[:,:2]**2).sum(1)+dz**2).max()),RADIUS)


class SearchContracts(baseline_search_tests.Search):
    def setUp(self):
        p=patch.object(baseline_search_tests,'code',search)
        p.start();self.addCleanup(p.stop)
    def test_collision_evidence_is_required_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);request={'key':'value'}
            search.save(home/'request.json',request)
            (home/'post_measurements.jsonl').write_text('')
            (home/'collision_events.jsonl').write_text('{}\n')
            rep={'request_sha256':search.digest(home/'request.json'),
                'post_evidence_sha256':search.digest(home/'post_measurements.jsonl'),
                'collision_evidence_sha256':search.digest(home/'collision_events.jsonl')}
            search.save(home/'report.json',rep)
            with patch.object(search,'check_report',return_value=[]) as check:
                self.assertEqual(search.verified_rows(home/'report.json',request),[])
                check.assert_called_once()
                (home/'collision_events.jsonl').write_text('{"changed":true}\n')
                with self.assertRaises(ValueError):search.verified_rows(home/'report.json',request)


if __name__=='__main__':unittest.main()
