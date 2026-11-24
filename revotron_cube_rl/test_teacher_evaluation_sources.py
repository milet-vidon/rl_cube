import unittest,json
from unittest.mock import patch
from pathlib import Path
from teacher_evaluation_sources import resolve_schedule

class EvaluationSourceTests(unittest.TestCase):
    def test_unmatched_seed_rejected_without_loading_another_workspace(self):
        with patch('teacher_evaluation_sources.dataset') as read:
            with self.assertRaisesRegex(ValueError,'No accepted'):resolve_schedule('/local/runs/base',{'source_seed':9501,'joint_names':[]},9901)
            read.assert_not_called()
    def test_matching_extra_resolved_in_current_clone_and_hash_checked(self):
        metadata={'source_seed':9501,'joint_names':['J'],'additional_sources':[{'seed':9901,'source':'/old/workspace/runs/new','manifest_sha256':'known'}]}
        with patch('teacher_evaluation_sources.sha',return_value='known'),patch('teacher_evaluation_sources.dataset',return_value=(None,None,['J'],None,{'seed':9901})) as read:
            path,result,matched=resolve_schedule('/local/runs/base',metadata,9901)
            self.assertEqual(path,Path('/local/runs/new'));self.assertTrue(matched);read.assert_called_once_with(path)
    def test_changed_extra_manifest_rejected(self):
        metadata={'source_seed':9501,'joint_names':[],'additional_sources':[{'seed':9901,'source':'/old/new','manifest_sha256':'known'}]}
        with patch('teacher_evaluation_sources.sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'evidence mismatch'):resolve_schedule('/local/runs/base',metadata,9901)
    def test_explicit_wrong_seed_is_not_silently_accepted(self):
        with patch('teacher_evaluation_sources.dataset',return_value=(None,None,[],None,{'seed':9501})):
            with self.assertRaisesRegex(ValueError,'seed does not match'):resolve_schedule('/local/base',{'joint_names':[]},9901,explicit='/local/base')
    def test_legacy_mismatch_requires_explicit_option_and_is_reported(self):
        with patch('teacher_evaluation_sources.dataset',return_value=(None,None,[],None,{'seed':9501})):
            _,_,matched=resolve_schedule('/local/base',{'source_seed':9501,'joint_names':[]},9901,allow_mismatch=True)
            self.assertFalse(matched)

if __name__=='__main__':unittest.main()
