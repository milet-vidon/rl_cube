import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

from contact_hold_wide_replay import Bank
from contact_hold_wide_planner import bind_path


class WideSamplingTests(unittest.TestCase):
    def test_stream_replay_bounded_disjoint_and_repeatable(self):
        def collect():
            b=Bank(8,123)
            for j in range(80):
                rows=np.arange(12)+12*j
                v=rows%4>1;f=rows%2==1
                b.add(dict(id=rows,phase=np.where(f,4,0),val=v,x=np.column_stack((rows,rows%4))))
            return b,b.finish()
        b,a=collect();_,again=collect()
        self.assertEqual(b.seen,[240]*4)
        self.assertEqual(len(a['id']),32)
        self.assertEqual(len(set(a['id'])),32)
        np.testing.assert_array_equal(a['val'],a['id']%4>1)
        np.testing.assert_array_equal(a['phase']==4,a['id']%2==1)
        np.testing.assert_array_equal(a['id'],again['id'])
        np.testing.assert_array_equal(np.bincount(a['x'][:,1]),[8]*4)
        self.assertGreater(max(a['id']),500)

    def test_empty_or_all_assisted_wave_cannot_train(self):
        with self.assertRaisesRegex(ValueError,'No eligible'):
            Bank(8,0).finish()

    def test_precomputed_ik_cannot_be_served_to_another_lane_pose(self):
        class Planner:diagnostics=[]
        xyz=np.array([.4,.2,1.]);path=[('ready',np.zeros((3,56)))]
        solve=bind_path(xyz,.25,path,['reset IK'])
        self.assertIs(solve(Planner(),xyz.copy(),.25),path)
        with self.assertRaisesRegex(RuntimeError,'Wrong reset pose'):solve(Planner(),xyz+.001,.25)
        with self.assertRaisesRegex(RuntimeError,'Wrong reset pose'):solve(Planner(),xyz,.24)

    def test_aborted_physics_wave_is_never_fit_and_retries_at_a_measured_size(self):
        import contact_hold_wide_campaign as w
        from contact_hold_loop_data import read, write
        class StopAfterBackoff(Exception):pass
        attempts=[]
        def run_sampler(cmd,log,fds):
            attempts.append(len(read(Path(cmd[cmd.index('--plan')+1]))['seeds']))
            if len(attempts)==2:raise StopAfterBackoff()
            log.write_text('out of memory')
            return dict(ok=False,reason='device_or_host_memory_headroom')
        with TemporaryDirectory() as tmp:
            home=Path(tmp)
            write(home/'settings.json',dict(cfg=dict(rounds=1,envs=1536,tiers=[768,1280,1536]),code=w.identity()))
            write(home/'state.json',dict(round=0,status='running',current=str(w.BASE),champion=str(w.BASE),vector_corpus=[]))
            with patch.object(w,'checkpoint',return_value=w.DIGEST),patch.object(w.loop,'CHILD_LOCKS',()),patch.object(w.loop,'gpu_slot',return_value=nullcontext(())),patch.object(w,'monitored',side_effect=run_sampler),self.assertRaises(StopAfterBackoff):
                w.campaign(home,{},True)
            s=read(home/'state.json')
            self.assertEqual(attempts,[1536,1280]);self.assertEqual(s['effective_envs'],1280)
            self.assertEqual(s['round'],0);self.assertEqual(s['vector_corpus'],[])
            self.assertFalse(read(Path(s['backoffs'][0]['receipt']))['ok'])


if __name__=='__main__':unittest.main()
