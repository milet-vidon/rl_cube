"""Check admission of GPU correction batches before they can affect the real optimizer."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contact_hold_gpu_campaign import identity, verify_batch
from contact_hold_loop_data import write
from contact_hold_dataset import sha


class GpuCollectionBoundary(unittest.TestCase):
    def setUp(self):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        self.root=Path(root.name);self.out=self.root/'batch';self.out.mkdir()
        parent=self.root/'parent';parent.mkdir();(parent/'best.pt').write_bytes(b'test parent')
        digest=sha(parent/'best.pt')
        write(parent/'result.json',dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256=digest))
        self.p=dict(parent=str(parent),parent_sha=digest,seeds=list(range(41000,41004)),code=identity())
        self.r=dict(schema='cuda_wrist_query_v1',scope='reset_to_wrist_hold_or_termination',checkpoint=digest,
                    num_envs=4,seeds=self.p['seeds'],student_start_phase=[0,4,0,5],physics_device='cuda:0',
                    role='collection',reset_count_per_lane=1,eligible_success_demonstration=False,full_episode=False,
                    cross_lane_contacts=0,physics_hz=240,control_hz=30,action_dim=56,
                    solver=dict(enableGPUDynamics=True,broadphaseType='GPU',solverType='TGS',
                                minPositionIterationCount=64,minVelocityIterationCount=4),samples=4)
        self.a=dict(x=np.zeros((4,198),np.float32),y=np.zeros((4,28),np.float32),mask=np.ones((4,28),bool),
                    val=np.array([False,False,True,False]),lane=np.arange(4),step=np.arange(10,14),phase=np.array([0,4,8,5]))
        self.a['x'][np.arange(4),198-13+self.a['phase']]=1.

    def receipt(self):
        plan=self.root/'collect.json';data=self.out/'batch.npz'
        write(plan,self.p);np.savez(data,**self.a)
        write(self.out/'report.json',dict(self.r,data_sha256=sha(data),plan_sha256=sha(plan)))
        return dict(path=str(self.out),sha=sha(self.out/'report.json'),plan_sha=sha(plan))

    def test_accepts_projected_student_transitions_through_wrist(self):
        a,r=verify_batch(self.receipt())
        self.assertEqual(r['samples'],4);self.assertEqual(a['phase'][2],8)

    def test_assisted_approach_and_assisted_squeeze_are_not_student_data(self):
        for lane,earlier in ((1,3),(3,4)):
            self.a['phase'][lane]=earlier;self.a['x'][lane,-13:-1]=0.
            self.a['x'][lane,198-13+earlier]=1.
            with self.subTest(lane=lane),self.assertRaisesRegex(ValueError,'wrongly aligned'):
                verify_batch(self.receipt())
            self.a['phase'][lane]=(4 if lane==1 else 5)
            self.a['x'][lane,-13:-1]=0.;self.a['x'][lane,198-13+self.a['phase'][lane]]=1.

    def test_unaligned_observations_are_ineligible(self):
        self.a['phase'][2]=4
        with self.assertRaisesRegex(ValueError,'wrongly aligned'):verify_batch(self.receipt())

    def test_cannot_fit_on_validation_or_contaminated_physics(self):
        self.p['seeds'][0]=21611
        with self.assertRaisesRegex(ValueError,'Ineligible'):verify_batch(self.receipt())
        self.p['seeds'][0]=41000;self.r['cross_lane_contacts']=1
        with self.assertRaisesRegex(ValueError,'Ineligible'):verify_batch(self.receipt())

    def test_edited_rollout_rejected_even_with_unchanged_shape(self):
        e=self.receipt();self.a['y'][0,0]=1.;np.savez(self.out/'batch.npz',**self.a)
        with self.assertRaisesRegex(ValueError,'Ineligible'):verify_batch(e)


if __name__=='__main__':unittest.main()
