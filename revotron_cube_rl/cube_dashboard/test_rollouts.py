"""Concurrency/recovery tests use a short CPU-only helper process, never Isaac or CUDA."""
import fcntl
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cube_dashboard import rollouts as ro


class RolloutTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.pr=Path(temp.name)/"project";(self.pr/"runs").mkdir(parents=True)
        ro._global(self.pr).mkdir(parents=True)
        for n in ro._NAMES:(ro._global(self.pr)/n).touch()
        self.model={"id":"known","label":"test","sha256":"a"*64}
        self.addCleanup(patch.stopall)
        patch.object(ro.inference,"catalog",return_value={"models":[self.model]}).start()
        self.gpu=patch.object(ro,"_check_gpu").start()
        real=ro._locks
        self.table=patch.object(ro,"_locks",side_effect=lambda:(real()[0],True)).start()

    @staticmethod
    def flock(p):
        f=p.open("a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

    def test_existing_campaign_even_if_state_is_stale_blocks_without_polling_gpu(self):
        d=self.pr/"runs"/"legacy_campaign";d.mkdir()
        (d/"state.json").write_text('{"status":"completed"}')
        with self.flock(d/"campaign.lock"),patch.object(ro,"_spawn") as spawn:
            self.assertFalse(ro.rollout_status(self.pr)["available"])
            with self.assertRaises(ro.RolloutBusy):ro.start_rollout(self.pr,"known")
            self.assertFalse(spawn.called);self.assertFalse(self.gpu.called)
            self.assertFalse(ro._store(self.pr).exists())

    def test_unknown_visibility_and_model_cannot_schedule(self):
        self.table.side_effect=lambda:({},False)
        self.assertFalse(ro.rollout_status(self.pr)["available"])
        with self.assertRaises(KeyError):ro.start_rollout(self.pr,"../../model.pt")
        with self.assertRaises(ro.RolloutUnavailable):ro.start_rollout(self.pr,"known")
        self.assertFalse(self.gpu.called)

    def test_lock_race_is_refused_and_reserved_fds_released(self):
        d=self.pr/"runs"/"newly_started";d.mkdir();f=d/"campaign.lock";f.touch()
        st=f.stat()
        import os
        inode=(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
        self.table.side_effect=[({},True),({inode:999},True)]
        with patch.object(ro,"_spawn") as spawn:
            with self.assertRaises(ro.RolloutBusy):ro.start_rollout(self.pr,"known")
            self.assertFalse(spawn.called)
        for n in ro._NAMES:
            with self.flock(ro._global(self.pr)/n):pass

    def test_gpu_and_launch_failures_do_not_leak_global_locks(self):
        self.gpu.side_effect=ro.RolloutBusy("已有 GPU 任务")
        with self.assertRaises(ro.RolloutBusy):ro.start_rollout(self.pr,"known")
        self.gpu.side_effect=None
        with patch.object(ro,"_spawn",side_effect=OSError("launch error")):
            with self.assertRaises(ro.RolloutUnavailable):ro.start_rollout(self.pr,"known")
        self.assertEqual(ro.rollout_status(self.pr)["jobs"][0]["status"],"failed")
        for n in ro._NAMES:
            with self.flock(ro._global(self.pr)/n):pass

    def test_child_owns_real_locks_and_persisted_frames_match_model(self):
        code="""from pathlib import Path
from hashlib import sha256
import json,sys,time
d=Path(sys.argv[1]); req=json.loads((d/'request.json').read_text())
for _ in range(500):
 if (d/'finish').exists():break
 time.sleep(.01)
raw=b'{"step":0}\\n';(d/'frames.jsonl').write_bytes(raw)
(d/'worker_result.json').write_text(json.dumps(dict(seed=20000000,panel_seeds=list(range(20000000,20000004)),
 checkpoint_sha256=req['sha256'],step_count=1,trajectory_sha256=sha256(raw).hexdigest())))
"""
        def fake_spawn(root,mid,d,fds,log):
            return subprocess.Popen([sys.executable,"-c",code,str(d)],pass_fds=tuple(fds),
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        with patch.object(ro,"_spawn",side_effect=fake_spawn):
            j=ro.start_rollout(self.pr,"known");d=ro._store(self.pr)/j["id"]
            try:
                self.assertEqual(ro.rollout_status(self.pr)["active_job"],j["id"])
                with self.assertRaises(ro.RolloutBusy):ro.start_rollout(self.pr,"known")
                with self.assertRaises(ro.RolloutBusy):ro.rollout_frames(self.pr,j["id"])
                for n in ro._NAMES:
                    with (ro._global(self.pr)/n).open("rb") as lk:
                        with self.assertRaises(BlockingIOError):fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                (d/"finish").touch()
            for _ in range(250):
                if (d/"exit.json").exists():break
                time.sleep(.01)
            s=ro.rollout_status(self.pr);self.assertTrue(s["available"]);self.assertEqual(s["jobs"][0]["status"],"completed")
            self.assertEqual(ro.rollout_frames(self.pr,j["id"])["frames"],[{"step":0}])
            # All state is on disk; even without manager's exit marker the authenticated result survives restart.
            (d/"exit.json").unlink()
            self.assertEqual(ro.rollout_status(self.pr)["jobs"][0]["status"],"completed")
            (d/"frames.jsonl").write_text('{"step":0,"height_m":9}\n')
            with self.assertRaises(ro.RolloutUnavailable):ro.rollout_frames(self.pr,j["id"])
            req=json.loads((d/"request.json").read_text());req["sha256"]="f"*64
            (d/"request.json").write_text(json.dumps(req))
            self.assertEqual(ro.rollout_status(self.pr)["jobs"][0]["status"],"failed")

    def test_refuses_path_traversal(self):
        for n in ("../x","/etc","GPU","7"*33):
            with self.assertRaises(KeyError):ro.rollout_frames(self.pr,n)
        folder=ro._store(self.pr);folder.symlink_to(self.pr)
        with self.assertRaises(ro.RolloutUnavailable):ro.rollout_status(self.pr)


if __name__=="__main__":unittest.main()
