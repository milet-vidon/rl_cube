"""CPU-only provenance, trace coverage and re-entry checks for the frozen B/C validation."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import natural_face90_hover_c as c


class HoverCRunner(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.req = dict(seeds=[101, 202], checkpoint='round70.pt')
        self.home = self.root / 'hover_c'
        self.source = self.root / 'hover_ab'
        folder = self.source / 'treatment'
        folder.mkdir(parents=True)
        self.b = folder / 'report.json'
        c.campaign.save(self.b, {})
        c.campaign.save(folder / 'request.json', self.req)
        ab_plan = dict(control_request_sha256=c.campaign.digest(folder / 'request.json'))
        self.rows = [dict(B=dict(tracking=dict(samples=n))) for n in [2, 0]]
        metrics = dict(group=dict(B=dict(unsafe=8)), per_seed=self.rows)
        self.saved = dict(plan=ab_plan, updated=False, independent=False,
                          treatment_report=str(self.b), treatment_report_sha256=c.campaign.digest(self.b),
                          hover_intervention_frames_by_lane=[2, 0], comparison=metrics)
        c.campaign.save(self.source / 'plan.json', ab_plan)
        c.campaign.save(self.source / 'hover_ab_summary.json', self.saved)
        self.learning = dict(hover_ablation=dict(operator=c.ab.OPERATOR, frames_by_lane=[2, 0]))
        for target in (patch.object(c.ab, 'RUNS', self.root), patch.object(c, 'BASE', self.source),
                       patch.object(c.ab, 'baseline', return_value=(self.req, self.b, ab_plan)),
                       patch.object(c, 'FILES', ('natural_face90_hover_c.py',)),
                       patch.object(c, 'compare', return_value=metrics),
                       patch.object(c.campaign.previous, 'verify_turn_report',
                                    return_value=([{}, {}], self.learning))):
            self.stack.enter_context(target)

    def invoke(self, *flags):
        with patch('sys.argv', ['hover', '--run-dir', str(self.home), *flags]), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            c.main()

    def test_check_is_read_only_and_requires_the_verified_b_receipt(self):
        with patch.object(c.ab, 'gpu') as gpu:
            self.invoke('--check')
            self.assertFalse(self.home.exists())
            self.saved['hover_intervention_frames_by_lane'][0] += 1
            c.campaign.save(self.source / 'hover_ab_summary.json', self.saved)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                self.invoke('--check')
            gpu.assert_not_called()

    def test_completed_and_partial_treatments_never_launch_a_second_worker(self):
        req, baseline, plan, _ = c.inputs()
        self.home.mkdir()
        c.campaign.save(self.home / 'plan.json', plan)
        with patch.object(c.ab, 'gpu') as gpu, patch.object(c, 'finish') as finish:
            with self.assertRaises(SystemExit):
                self.invoke()
            folder = self.home / 'treatment'
            folder.mkdir()
            c.campaign.save(folder / 'report.json', {})
            self.invoke()
            finish.assert_called_once_with(self.home, req, baseline, plan)
            gpu.assert_not_called()

    def test_busy_lease_leaves_no_partial_directory(self):
        with patch.object(c.ab, 'gpu', side_effect=RuntimeError('GPU 0 正被使用')) as gpu:
            with self.assertRaises(SystemExit) as error:
                self.invoke()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.home.exists())
            gpu.assert_called_once()

    def test_anchor_is_measured_only_on_hover_and_counted_per_lane(self):
        self.learning['hover_anchor_ablation'] = dict(operator=c.OPERATOR, frames_by_lane=[1, 0])
        trace = self.b.parent / 'turn_contact_evidence.jsonl'
        row = dict(seed=101, lane=0, step=1, pre_hover_measured_anchor=1, pre_hover_residual_off=1,
                   pre_ik_anchor_error_max_rad=0., pre_previous_reference_error_max_rad=.12,
                   pre_previous_command_error_max_rad=.041, sent_command_error_max_rad=.027)
        outside = dict(row, seed=202, lane=1, pre_hover_measured_anchor=0, pre_hover_residual_off=0,
                       pre_ik_anchor_error_max_rad=.12)
        def verify(frame):
            trace.write_text(json.dumps(row) + '\n' + json.dumps(frame) + '\n')
            c.check_anchor(self.b, self.learning, [1, 0], self.req['seeds'])
        verify(outside)
        for fields in (dict(pre_ik_anchor_error_max_rad=0), dict(pre_hover_measured_anchor=1),
                       dict(sent_command_error_max_rad=float('nan')), dict(seed=101),
                       dict(pre_hover_residual_off=1, pre_hover_measured_anchor=1, pre_ik_anchor_error_max_rad=0)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                verify(dict(outside, **fields))

    def test_window_and_deltas_use_real_b_c_arm_names(self):
        before = dict(A=dict(unsafe=8), B=dict(unsafe=5),
                      per_seed=[dict(common_hover_delta_B_minus_A={'rms_mm': -2},
                                     precontact_2s=dict(status='no_early_right_digit_anchor_in_A'),
                                     end_reason='keep_native_A')], A_median_pad_peak_to_peak_mm=52.)
        actual = c.bc_labels(before)
        self.assertEqual((actual['B']['unsafe'], actual['C']['unsafe']), (8, 5))
        self.assertEqual(actual['per_seed'][0]['common_hover_delta_C_minus_B']['rms_mm'], -2)
        self.assertEqual(actual['per_seed'][0]['precontact_2s']['status'], 'no_early_right_digit_anchor_in_B')
        self.assertEqual(actual['per_seed'][0]['end_reason'], 'keep_native_A')
        self.assertEqual(actual['B_median_pad_peak_to_peak_mm'], 52.)


if __name__ == '__main__':
    unittest.main()
