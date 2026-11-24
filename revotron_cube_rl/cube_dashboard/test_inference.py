"""No GPU or simulator is opened by these tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import inference as api


class Validation(unittest.TestCase):
    def test_tampered_source_and_link_outside_runs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs';root.mkdir()
            original=root/'episode.json';original.write_bytes(b'abc')
            old=hashlib.sha256(b'abc').hexdigest()
            external=Path(tmp)/'external';external.write_text('abc')
            link=root/'inside';link.symlink_to(external)
            with patch.object(api, 'RUNS', root):
                self.assertEqual(api._verified(original,old,10), b'abc')
                original.write_bytes(b'bad')
                for p in [original, link, external]:
                    with self.subTest(path=p), self.assertRaises(api.InferenceUnavailable):
                        api._verified(p,old,10)

    def test_model_list_excludes_in_progress_fit_and_different_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs';root.mkdir()
            for ver in ('1','2'):
                p=root/('contact_hold90_long_v'+ver);p.mkdir()
                (p/'state.json').write_text(json.dumps({'completed_round':1}))
                (p/'settings.json').write_text(json.dumps({'source_weights_sha256':api.R15_SHA if ver=='2' else '0'*64}))
                for k in (1,2):
                    fit=p/f'round_{k:03d}/fit';fit.mkdir(parents=True)
                    (fit/'result.json').write_text(json.dumps({'completed':True,'checkpoint_reload_verified':True,'checkpoint_sha256':'b'*64}))
                    (fit/'best.pt').write_bytes(b'example')
            with patch.object(api, 'RUNS', root):
                self.assertEqual(list(api._model_cards()),['contact_hold90_long_v2-r1'])
                for model in ('../bad','rl_cube','contact_hold90_long_v2-r2'):
                    with self.subTest(model=model), self.assertRaises(api.InferenceNotFound):
                        api.predict(model, 'r15-canonical-21611', 0)
                with self.assertRaises(api.InferenceNotFound):api.trajectory('../../etc/passwd')

    def test_new_training_card_requires_exact_already_completed_parent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(api,'RUNS',Path(tmp)):
            def record(name, cfg):
                d=api.RUNS/name; fit=d/'round_001/fit';fit.mkdir(parents=True)
                for path,payload in [(d/'settings.json',cfg),(d/'state.json',{'completed_round':1}),
                    (fit/'result.json',dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256='b'*64))]:
                    path.write_text(json.dumps(payload))
                (fit/'best.pt').write_bytes(b'example')
                return fit
            source=record('contact_hold90_long_v2',dict(source_weights_sha256=api.R15_SHA))
            settings=dict(engine=api.TUNABLE_ENGINE,warm_start_sha256=api.R15_SHA,source_fit=str(source),source_weights_sha256='b'*64)
            record('contact_hold90_ui_ok',settings)
            record('contact_hold90_ui_bad',dict(settings,source_weights_sha256='a'*64))
            record('contact_hold90_ui_unfinished',dict(settings,source_fit=str(source).replace('001','002')))
            self.assertEqual(set(api._model_cards()),{'contact_hold90_long_v2-r1','contact_hold90_ui_ok-r1'})

    def test_fractional_or_boolean_frame_refused(self):
        for bad in (0.3,True,-1,'20',None):
            with self.subTest(step=bad), self.assertRaises(ValueError):api.predict('r15','r15-canonical-21611',bad)


@unittest.skipUnless((api.RUNS/api.ARCHIVE/'retarget_report.json').is_file(), 'requires the recorded local R15 episodes')
class LocalEvidence(unittest.TestCase):
    def test_frames_use_world_coordinates_and_physical_report_not_checkpoint_claims(self):
        ep=api.trajectory('r15-canonical-21611')
        self.assertEqual(len(ep['frames']),590)
        summary=ep['summary']
        self.assertAlmostEqual(summary['peak_height_m'],max(f['height_m'] for f in ep['frames']),places=7)
        self.assertTrue(summary['final_table_contact']);self.assertFalse(summary['whole_turn'])
        self.assertAlmostEqual(summary['best_hold_s'],.5)
        row=ep['frames'][0]
        self.assertAlmostEqual(row['cube'][2]-.045-ep['table_z_m'],row['height_m'],places=7)
        self.assertGreater(row['left_palm'][2],ep['table_z_m'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU PyTorch')
    def test_pre_action_observation_reproduces_recorded_r15_outputs(self):
        episodes=api.catalog()['episodes']
        for episode in episodes:
            for k in (0, episode['step_count']//2, episode['step_count']-1):
                got=api.predict('r15',episode['id'],k)
                with self.subTest(episode=episode['id'],step=k):
                    self.assertEqual(got['input_frame'],'reset' if k==0 else k-1)
                    self.assertLess(got['max_abs_error_rad'],2e-4)
                    self.assertEqual(len(got['predicted']),28)
                    self.assertEqual(got['device'],'cpu');self.assertFalse(got['physics_stepped'])
            with self.assertRaises(ValueError):api.predict('r15',episode['id'],episode['step_count'])


if __name__ == '__main__':
    unittest.main()
