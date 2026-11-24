import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import train_contact_hold_corrected as fit


class CorrectionSupervisionTest(unittest.TestCase):
    def test_partial_labels_only_differentiate_named_columns(self):
        names_in_model_order = ['f0', 'arm_b', 'f1', 'arm_a']
        cols = torch.tensor([names_in_model_order.index(n) for n in ('arm_a','arm_b')])
        pred = torch.tensor([[10., 2., 30., 4.]], requires_grad=True)
        fit.residuals(pred, torch.tensor([[3., 1.5]]), cols).square().sum().backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[0., 1., 0., 2.]]))

    def test_unreviewed_summary_refused_before_reading_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fit, 'CANDIDATES', Path(tmp)):
            (Path(tmp)/'summary.json').write_text('{"training_authorized":true}')
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                fit.reviewed_data()
