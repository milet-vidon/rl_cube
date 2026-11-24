import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import training as tr


class TrainingTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "runs/contact_hold90_long_v2"
        self.rd = self.home / "round_001"
        self.wave = self.rd / "collection/wave_0000_0004"
        self.wave.mkdir(parents=True)
        self.log = self.wave / "sample_deadbeef.log"
        self.state = dict(status="running",stage="collection",active_round=1,completed_round=0,current_envs=4,
                          updated="2026-11-12T18:00:00+08:00",log=str(self.log))
        self.write(self.home/'settings.json',dict(envs=4,rounds=100,horizon=3500,val_envs=4))
        self.write(self.home/'state.json',self.state)
        (self.home/'campaign.lock').touch()

    @staticmethod
    def write(p,obj):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj,ensure_ascii=False))

    def snapshot(self):
        return tr.training_snapshot(self.root,self.home.name)

    def test_sampling_mixed_hold_does_not_leak_into_validation(self):
        self.log.write_text("GPU90_READY 4 role collection IK accepted 4\nGPU90_STEP 0 active 4 hold 0\nGPU90_STEP 120 active 2 hold 1\n")
        sample=self.wave/'sample_x'
        self.write(sample/'report.json', dict(cube_edge_m=.09,role='collection',num_envs=4,samples=250,
            metrics=dict(hold_count=1,ending_hold_count=1,returned_to_table_count=0)))
        self.write(self.rd/'collection/accepted_0000.json',dict(dest=str(sample)))
        # This failed attempt has more holds and cannot count toward the accepted collection.
        self.write((self.wave/'sample_old/report.json'),dict(cube_edge_m=.09,role='collection',num_envs=100,samples=50000,
                                                             metrics=dict(hold_count=90)))
        s=self.snapshot()
        self.assertEqual(s["progress"]["step"],120);self.assertEqual(s["timeline"][-1]["hold"],1)
        self.assertEqual(s["history"][0]["collection"]["envs"],4)
        self.assertEqual(s["history"][0]["collection"]["samples"],250)
        self.assertIsNone(s["validation"])
        self.write(self.rd/'validation_summary.json',dict(envs=4,held=0,hold_rate=0.,still_held_at_end=0))
        self.assertEqual(self.snapshot()["validation"]["hold_rate"],0.)

    def test_monitored_batch_and_round_limit_follow_frozen_round_only(self):
        self.write(self.home/'settings.json',dict(engine='contact_hold90_tunable_v1',envs=4,rounds=100,batch=1024))
        self.write(self.rd/'control.json',dict(round=1,params=dict(batch=256,rounds=75)))
        self.write(self.home/'control_pending.json',dict(params=dict(batch=4096,rounds=80)))
        snap=self.snapshot()
        self.assertEqual((snap['settings']['batch'],snap['rounds']),(256,75))

    def test_interrupted_retry_does_not_reuse_previous_attempt(self):
        (self.home/'console.log').write_text("GPU90_STEP 3000 active 6 hold 42\n")
        with patch.object(tr,"_lock_status",return_value=(False,None)):
            s=self.snapshot();self.assertEqual(s['status'],'stale');self.assertEqual(s["timeline"],[])
            self.assertNotIn("step",s["progress"])
        with patch.object(tr,"_lock_status",return_value=(None,None)):
            self.assertEqual(self.snapshot()['status'],'unknown')
        self.write(self.home/'state.json',dict(self.state,status='interrupted'))
        self.assertEqual(self.snapshot()['status'],'interrupted')

    def test_lock_held_by_writer_detected_without_touching_file(self):
        p=self.home/'campaign.lock'
        with p.open('r') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            held,pid=tr._lock_status(p)
        self.assertTrue(held);self.assertEqual(pid,os.getpid())

    def test_live_and_completed_fit_have_checkpoint_and_distinct_curve(self):
        fd=self.rd/'fit';fd.mkdir()
        (self.rd/'fit.log').write_text("FIT90 250 / 500 heldout_mse .01 best_step 250\nFIT90 500 / 500 heldout_mse .02 best_step 250\n")
        self.write(fd/'result.json',dict(completed=True,updates=500,selected_step=250,checkpoint_sha256="a"*64,
            best_metrics={"phase_balanced_normalized_mse":.01},last_metrics={"phase_balanced_normalized_mse":.02}))
        (fd/'best.pt').write_bytes(b'best')
        (fd/'last.pt').write_bytes(b'last')
        self.write(self.home/'state.json',dict(self.state,stage="fit",log=str(self.rd/'fit.log')))
        s=self.snapshot();f=s["history"][0]["fit"]
        self.assertEqual((s['progress']['update'],len(f['curve']),f['mse'],f['last_mse']),(500,2,.01,.02))
        self.assertTrue(s["history"][0]['fit_resume_saved'])
        self.assertEqual(s['checkpoints'][0]['path'],'runs/contact_hold90_long_v2/round_001/fit/best.pt')

    def test_excludes_path_escape_large_logs_and_invalid_campaigns(self):
        rogue=self.root/'private.log';rogue.write_text('secret outside the run')
        self.log.symlink_to(rogue)
        self.assertNotIn('secret',str(self.snapshot()))
        for name in ('../../private','contact_hold90_../other','/etc'):
            with self.assertRaises(ValueError):tr.training_snapshot(self.root,name)
        self.log.unlink();self.log.write_text('secret-before-tail\n'+'x'*(tr._LOG_BYTES+500)+'\nGPU90_STEP 10 active 3 hold 0\n')
        self.assertNotIn('secret',str(self.snapshot()))
        (self.home.parent/'contact_hold90_reset_probe_after').mkdir()
        (self.home.parent/'contact_hold90_long_symlink').symlink_to(self.root)
        smoke=self.home.parent/'contact_hold90_smoke_1';self.write(smoke/'settings.json',dict(envs=4,rounds=1))
        self.assertEqual([(x['id'],x['kind']) for x in tr.runs_catalog(self.root)],
                         [(self.home.name,'training'),(smoke.name,'diagnostic')])


if __name__=="__main__":unittest.main()
