"""Offline tests for the frozen R70 source and physical-run selection."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import fcntl
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import natural_face90_hover_ab as ab
import natural_face90_v5_long as original_locks


class HoverRunner(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root/'project/runs'
        self.runs.mkdir(parents=True)
        self.home = self.runs/'hover'
        self.control = self.root/'A'
        self.control.mkdir()
        self.req = dict(role='validate', learn=False, iteration=70, checkpoint_sha=ab.R70_SHA,
                        checkpoint='R70.pt', seeds=list(ab.SEEDS))
        ab.campaign.save(self.control/'report.json', {})
        ab.campaign.save(self.control/'request.json', self.req)
        mocks = (patch.object(ab, 'RUNS', self.runs),
                 patch.object(ab, 'CONTROL', self.control),
                 patch.object(ab,'source_wave',return_value=(self.req,self.control/'report.json')))
        for ctx in mocks:
            ctx.start()
            self.addCleanup(ctx.stop)

    def invoke(self, *args):
        with patch('sys.argv',['hover','--run-dir',str(self.home),*args]), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            ab.main()

    def test_checkpoint_and_lane_order_are_pinned(self):
        self.assertEqual(ab.baseline()[2]['seeds'], ab.SEEDS)
        for change in (dict(seeds=list(reversed(ab.SEEDS))), dict(checkpoint_sha='1'*64), dict(iteration=71)):
            with patch.dict(self.req, change), self.assertRaises(ValueError):
                ab.baseline()

    def test_check_and_held_lock_do_not_create_run_or_touch_gpu(self):
        lock_dir = self.root/'work/teacher_optimization_20261103'
        lock_dir.mkdir(parents=True)
        with (lock_dir/original_locks.LOCK_NAMES[0]).open('a') as held, \
             patch.object(original_locks, 'ROOT', self.runs.parent), \
             patch.object(ab.subprocess,'check_output') as nvidia, \
             patch.object(ab.subprocess,'run') as worker, \
             patch('natural_face90_hover_ab_metrics.summarize', return_value={'group':{}}):
            fcntl.flock(held, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.invoke('--check')
            with self.assertRaises(SystemExit) as ctx:
                self.invoke()
            self.assertEqual(ctx.exception.code, 2)
            worker.assert_not_called()
            nvidia.assert_not_called()
            self.assertFalse(self.home.exists())

    def test_completed_treatment_reopens_without_new_rollout(self):
        req, control, plan = ab.baseline()
        self.home.mkdir()
        ab.campaign.save(self.home/'plan.json', plan)
        with patch.object(ab,'gpu',return_value=nullcontext((1,2))) as gpu, \
             patch.object(ab,'finish') as finish:
            with self.assertRaises(SystemExit):
                self.invoke()
            (self.home/'treatment').mkdir()
            ab.campaign.save(self.home/'treatment/report.json', {})
            self.invoke()
            finish.assert_called_once_with(self.home, req, control, plan)
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
