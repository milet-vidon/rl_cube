"""Exercise the archived C receipt and a D rerun without obtaining an Isaac lease."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import natural_face90_hover_d as d


class HoverDRunner(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.source, self.home = root / 'c', root / 'd'
        c_dir = self.source / 'treatment'
        c_dir.mkdir(parents=True)
        self.report = c_dir / 'report.json'
        d.campaign.save(self.report, {})
        self.req = dict(iteration=70, checkpoint='R70.pt', seeds=[101, 102])
        self.c_plan = dict(request_sha256='request-digest')
        self.metrics = dict(group=dict(C=dict(unsafe=2, hover_frames=5)), per_seed=[
            dict(C=dict(tracking=dict(samples=n))) for n in (5, 0)])
        self.saved = dict(plan=self.c_plan, updated=False, independent=False,
                          treatment_report=str(self.report), treatment_report_sha256=d.campaign.digest(self.report),
                          anchor_frames_by_lane=[5, 0], comparison=self.metrics)
        d.campaign.save(self.source / 'plan.json', self.c_plan)
        d.campaign.save(self.source / 'hover_bc_summary.json', self.saved)
        self.learning = dict(approach_recenter_ablation=dict(operator=d.OPERATOR, inner_m=d.INNER_M,
                                                              outer_m=d.OUTER_M, frames_by_lane=[2, 0]))
        for mock in (patch.object(d.ab, 'RUNS', root), patch.object(d, 'BASE', self.source),
                     patch.object(d.c, 'inputs', return_value=(self.req, root / 'b.json', self.c_plan, {})),
                     patch.object(d.c, 'compare', return_value=self.metrics),
                     patch.object(d.c, 'bc_labels', side_effect=lambda result: result),
                     patch.object(d.c, 'frozen_report', return_value=(self.learning, [5, 0])),
                     patch.object(d, 'read_episodes', return_value=[dict(anchor_frames=5, unsafe=True),
                                                                    dict(anchor_frames=0, unsafe=True)]),
                     patch.object(d.c, 'check_anchor'), patch.object(d, 'FILES', ('natural_face90_hover_d.py',))):
            self.stack.enter_context(mock)

    def invoke(self, *flags):
        with patch('sys.argv', ['d', '--run-dir', str(self.home), *flags]), \
                redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            d.main()

    def test_check_is_read_only_and_recomputes_c_baseline(self):
        with patch.object(d.ab, 'gpu') as gpu:
            self.invoke('--check')
            self.assertFalse(self.home.exists())
            d.c.check_anchor.assert_called_once()
            d.read_episodes.assert_called_once_with(self.report, self.req['seeds'], treatment=False)
            self.saved['comparison'] = dict(group={}, per_seed=[])
            d.campaign.save(self.source / 'hover_bc_summary.json', self.saved)
            with self.assertRaisesRegex(ValueError, 'physical evidence'):
                self.invoke('--check')
            gpu.assert_not_called()

    def test_refuses_foreign_or_changed_baseline(self):
        d.campaign.save(self.report, dict(replaced=True))
        with patch.object(d.ab, 'gpu') as gpu:
            with self.assertRaisesRegex(ValueError, 'frozen C'):
                self.invoke('--check')
            gpu.assert_not_called()

    def test_busy_and_partial_runs_cannot_launch_a_second_worker(self):
        with patch.object(d.ab, 'gpu', side_effect=RuntimeError('GPU 0 正被使用')), \
                patch.object(d.subprocess, 'run') as worker, self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.home.exists())
        worker.assert_not_called()
        self.home.mkdir()
        d.campaign.save(self.home / 'plan.json', d.inputs()[2])
        with self.assertRaises(SystemExit), patch.object(d.ab, 'gpu') as gpu:
            self.invoke()
        gpu.assert_not_called()

    def test_completed_evaluation_only_publishes_a_matching_receipt(self):
        self.home.mkdir()
        d.campaign.save(self.home / 'plan.json', d.inputs()[2])
        treatment = self.home / 'treatment'
        treatment.mkdir()
        d.campaign.save(treatment / 'report.json', {})
        summary = self.home / 'hover_cd_summary.json'
        comparison = dict(group=dict(D=dict(unsafe=1)), per_seed=[
            dict(D=dict(hold_frames=2, anchor_frames=5)), dict(D=dict(hold_frames=0, anchor_frames=0))])
        with patch.object(d, 'compare', return_value=comparison), patch.object(d.ab, 'gpu') as gpu:
            self.invoke()
            first = summary.read_bytes()
            self.invoke()
            self.assertEqual(first, summary.read_bytes())
            self.assertIs(d.campaign.read(summary)['updated'], False)
            self.learning['approach_recenter_ablation']['frames_by_lane'][0] = 3
            with self.assertRaisesRegex(ValueError, 'control steps'):
                self.invoke()
            self.assertEqual(first, summary.read_bytes())
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
