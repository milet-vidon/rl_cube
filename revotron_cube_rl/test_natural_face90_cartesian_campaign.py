from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_face90_cartesian as app


class CartesianCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = app.self_check()['receipt']

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.geometry = {'test': 1}
        self.cfg = dict(rng=123, source_sha256=app.fingerprint(), geometry_sha256=app.wrist.geometry_hash(self.geometry),
                        scene_sha256='a'*64, pad_orientation_geometry=self.geometry, placo=self.solver)

    def test_neutral_train_resume_requests(self):
        probe = app.request(self.cfg, 'validate', app.PAIRED_SEEDS, 0, None)
        app.validate_request(probe)
        ck = self.home/'checkpoint.pt'
        ck.write_bytes(b'new-policy')
        fresh = app.request(self.cfg, 'train', list(range(app.TRAIN_BASE, app.TRAIN_BASE+4)), 1, None)
        app.validate_request(fresh)
        req = app.request(self.cfg, 'train', list(range(app.TRAIN_BASE+32, app.TRAIN_BASE+36)), 2, str(ck))
        app.validate_request(req)
        for k, v in [('seeds', app.PAIRED_SEEDS), ('placo', {}), ('transfer', {}), ('geometry_sha256', 'bad'),
                     ('learn', False), ('checkpoint_sha', 'legacy')]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                app.validate_request(dict(req, **{k:v}))
        with self.assertRaises(ValueError):
            app.validate_request(dict(fresh, checkpoint=str(ck), checkpoint_sha=req['checkpoint_sha']))
        with self.assertRaises(ValueError):
            app.validate_request(dict(probe, seeds=app.PAIRED_SEEDS[::-1]))
        for seeds in [list(range(app.VALIDATION_BASE+32, app.VALIDATION_BASE+36)), list(range(69_000_000,69_000_004))]:
            app.validate_request(app.request(self.cfg, 'validate', seeds, 2, str(ck)))

    def test_train_requires_paired_physical_evidence_before_creating_directory(self):
        probe = self.home/'r27_recipe_cartesian_probe'
        train = self.home/'r27_recipe_cartesian_ppo15'
        args = ['--probe-dir', str(probe), '--run-dir', str(train)]
        source = dict(policy_seed=123, pad_orientation_geometry=self.geometry)
        plan = dict(input_summary_sha256='input', orientation_plan=self.cfg)
        with (patch.object(app.points, 'RUNS', self.home), patch.object(app, 'self_check', return_value={'receipt':self.solver}),
              patch.object(app.preshape, 'prepare', return_value=(plan,source,{},{})),
              patch.object(app, 'wave', side_effect=AssertionError('unexpected physics')),
              patch.object(app, 'campaign', side_effect=AssertionError('unexpected training')),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            app.main(['check', *args])
            with self.assertRaises(SystemExit):
                app.main(['train', *args])
        self.assertFalse(probe.exists())
        self.assertFalse(train.exists())

    def test_gate_uses_same_fixed_seeds_and_full_group_unsafe(self):
        seeds = app.anchor.metrics.FIXED_SEEDS
        rows = [dict(seed=s, lane=i, safe_qualified_index_first=i<2, safe_drive_2deg=1<=i<=2)
                for i,s in enumerate(seeds)]
        ready = dict(ready_step_by_lane=[0]*11, handoff_error_rad_by_lane=[0]*11,
                     orientation=dict(feasible_by_lane=[True]*11))
        candidate = dict(per_seed=rows, unsafe_seeds=[])
        original = dict(unsafe_seeds=[])
        def result(): return app.preshape.paired_result(candidate, original, original, ready)
        self.assertFalse(result()['paired_contact_gate_met'])
        rows[0]['safe_drive_2deg'] = True
        self.assertTrue(result()['paired_contact_gate_met'])
        candidate['unsafe_seeds'] = [next(s for s in app.PAIRED_SEEDS if s not in seeds)]
        self.assertFalse(result()['paired_contact_gate_met'])

    def test_resume_joins_accepted_train_weights_even_if_training_ended(self):
        ck = self.home/'new.pt'
        ck.write_bytes(b'checkpoint')
        folder = self.home/'iteration_0001'
        slot = folder/'train'
        attempt = slot/'attempt_ab123456'
        attempt.mkdir(parents=True)
        cfg = dict(self.cfg, lanes=4, iterations=1, origin=dict(iteration=0, checkpoint=None, sha=None))
        req = app.request(cfg,'train',list(range(app.TRAIN_BASE,app.TRAIN_BASE+4)),1,None)
        app.v7.save(slot/'request.json', req)
        report = attempt/'report.json'
        app.v7.save(report, {})
        app.v7.save(slot/'accepted.json', dict(attempt=attempt.name,request_sha=app.v7.digest(slot/'request.json'),
                                               report_sha=app.v7.digest(report)))
        learning = dict(output_checkpoint=str(ck), output_checkpoint_sha=app.v7.digest(ck))
        app.v7.save(folder/'result.json', dict(learning=learning))
        state = dict(iteration=1, checkpoint=str(ck), checkpoint_sha=app.v7.digest(ck), status='iteration_limit')
        app.v7.save(self.home/'state.json', state)
        with patch.object(app,'verify_report',return_value=([],learning)) as verify:
            app.campaign(self.home,cfg,None)
            verify.assert_called_once_with(report,req)
            app.v7.save(self.home/'state.json', dict(state, checkpoint_sha='bad'))
            with self.assertRaisesRegex(ValueError, 'differs from'):
                app.campaign(self.home,cfg,None)


if __name__ == '__main__':
    unittest.main()
