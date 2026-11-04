import json, tempfile, unittest
from pathlib import Path
from teacher_bootstrap import dataset, sha

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        names=[f'Joint{i}_L' for i in range(28)]+[f'Joint{i}_R' for i in range(28)]
        self.result=dict(full_episode=True,dynamic_grasp_passed=True,center_transport_passed=True,
                         failure=None,cube_pose_writes_during_control=0,
                         robot_table_impulse_sum_Ns=0,self_contact_impulse_sum_Ns=0)
        self.rows=[dict(q=[float(i)]*56,cube_position=[0.,0.,1.],cube_quaternion_wxyz=[1.,0.,0.,0.],
                        step=i,nominal_target=[float(i+10)]*56,applied_target=[float(i+20)]*56) for i in range(3)]
        (self.root/'config.json').write_text(json.dumps({'joint_names':names}))
        self.write()
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        (self.root/'trajectory.json').write_text(json.dumps(self.rows))
        (self.root/'result.json').write_text(json.dumps(self.result))
        manifest={n:sha(self.root/n) for n in ('trajectory.json','result.json','config.json')}
        (self.root/'artifact_manifest.json').write_text(json.dumps({'sha256':manifest}))
    def test_pre_action_alignment(self):
        x,y,_,indices,_=dataset(self.root)
        self.assertEqual(x.shape,(2,64));self.assertEqual(y.shape,(2,28))
        self.assertEqual(x[0,0],0.);self.assertEqual(y[0,0],11.)
        self.assertEqual(indices,list(range(28)))
    def test_reject_changed_evidence(self):
        (self.root/'trajectory.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):dataset(self.root)
    def test_reject_failed_or_reset_episode(self):
        for key,value in [('center_transport_passed',False),('full_episode',False),
                          ('cube_pose_writes_during_control',1),('self_contact_impulse_sum_Ns',.1)]:
            old=self.result[key];self.result[key]=value;self.write()
            with self.assertRaisesRegex(ValueError,'not an accepted'):dataset(self.root)
            self.result[key]=old
    def test_reject_nan(self):
        self.rows[1]['nominal_target'][0]=float('nan');self.write()
        with self.assertRaisesRegex(ValueError,'Non-finite'):dataset(self.root)
    def test_applied_labels_preserve_pre_action_alignment(self):
        x,y,_,_,_=dataset(self.root,'applied_target')
        self.assertEqual(x[0,0],0.);self.assertEqual(y[0,0],21.)

if __name__=='__main__':unittest.main()
