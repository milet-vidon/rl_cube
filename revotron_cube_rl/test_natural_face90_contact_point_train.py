"""CPU-only checks of the versioned PPO campaign; all worker waves here are synthetic."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import natural_face90_contact_point_train as train


def outcome(target, driven):
    return dict(target_deg=target, success=True, unsafe=False, natural_ready=True,
                turn_started=True, progress=1., right_contact=1., posture_quality=1., hold_fraction=1.,
                distal_contact_frames=6, qualified_index_frames=6, drive_frames=5,
                peak_index_driven_deg=driven, peak_drive_response_deg=2.,
                peak_isolated_drive_progress_deg=driven, end_reason='post_turn_verified')


class ContactPointTrainingTests(unittest.TestCase):
    def test_material_point_and_promotion_only_on_first_update(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            path.write_bytes(b'unit test')
            origin = dict(iteration=80, checkpoint=str(path), sha=train.v7.digest(path))
            cfg = dict(schema=train.points.SCHEMA, origin=origin, source_sha256={},
                       rng=327, contact_point_offset_deg=-7, target_deg=90)
            first = train.request(cfg, 'train', [49_000_000, 49_000_001], 81, str(path))
            self.assertEqual(first['promote_from_15'], origin)
            self.assertNotIn('transfer', first)
            for req in (first, train.request(cfg, 'train', [49_000_032, 49_000_033], 82, str(path)),
                        train.request(cfg, 'validate', [68_500_000, 68_500_001], 81, str(path))):
                train.points.validate_request(req)
                self.assertEqual(req['target_deg'], 90)
                self.assertEqual(req['contact_point_offset_deg'], -7)
                if req is not first:
                    self.assertNotIn('promote_from_15', req)

    def test_independent_gate_requires_full_quarter_turn_when_promoted(self):
        self.assertEqual(train.score([outcome(15, 2)])['qualification_successes'], 1)
        partial = train.score([outcome(90, 89.9)])
        self.assertEqual(partial['qualification_successes'], 0)
        self.assertEqual(partial['index_driven_successes'], 1)
        self.assertEqual(train.score([outcome(90, 90)])['qualification_successes'], 1)
        self.assertEqual(train.score([dict(outcome(90, 90), unsafe=True)])['qualification_successes'], 0)

    def test_15_degree_promotion_requires_registered_and_qualified_heldout_panel(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = root / 'runs/r27_recipe_sweep'
            home = root / 'runs/r27_recipe15'
            registry = root / 'runs/natural_face90_test_registry'
            folder = home / 'iteration_0075'
            for path in (sweep, registry, folder):
                path.mkdir(parents=True)
            save = train.v7.save
            save(sweep / 'contact_point_summary.json', {'selected_offset_deg': 7})
            source = dict(transfer={'iteration': 70})
            cfg = dict(schema=train.points.SCHEMA, target_deg=15, origin=source['transfer'],
                       sweep_dir=str(sweep), sweep_summary_sha256=train.v7.digest(sweep / 'contact_point_summary.json'),
                       source_sha256={}, contact_point_offset_deg=7, validation_seeds=16)
            save(home / 'config.json', cfg)
            save(home / 'state.json', dict(status='qualified_15deg_state_specialist', iteration=75,
                                            checkpoint='policy', checkpoint_sha='hash'))
            seeds = list(range(69_000_000, 69_000_032))
            save(registry / 'allocations.json', {str(home): seeds[0]})
            valid = dict(unsafe=0, qualification_successes=14)
            tested = dict(unsafe=0, qualification_successes=28)
            save(folder / 'result.json', dict(validation_after_update=valid, independent_test=tested))
            held = dict(checkpoint='policy', sha='hash', seeds=seeds, results=tested)
            save(home / 'final_test.json', held)
            save(home / 'qualified_policy.json', dict(checkpoint='policy', sha='hash', results=tested,
                                                       target_deg=15, contact_point_offset_deg=7, source_sha256={}))
            with patch.object(train.v7, 'ROOT', root), patch.object(train.points, 'fingerprint', return_value={}), \
                    patch.object(train, 'check_state'), patch.object(train, 'accepted_panel', side_effect=[valid, tested, valid]):
                origin = train.qualified_origin(home, sweep, 7, source)
                self.assertEqual(origin, dict(iteration=75, checkpoint='policy', sha='hash'))
                save(home / 'final_test.json', dict(held, seeds=list(range(69_000_032, 69_000_064))))
                with self.assertRaisesRegex(ValueError, 'registry'):
                    train.qualified_origin(home, sweep, 7, source)

    def test_mocked_campaign_seed_blocks_and_promotion(self):
        for target in (15, 90):
            with self.subTest(target=target), TemporaryDirectory() as tmp:
                home = Path(tmp)
                ck = home / 'source.pt'
                ck.write_bytes(b'entry fixture')
                iteration = 70 if target == 15 else 100
                origin = dict(checkpoint=str(ck), sha=train.v7.digest(ck), iteration=iteration)
                cfg = dict(schema=train.points.SCHEMA, origin=origin, source_sha256={},
                           target_deg=target, contact_point_offset_deg=7, rng=327, lanes=16,
                           iterations=iteration + 1, validation_seeds=16, validate_every=5)
                requests = []

                def wave(slot, req, fd):
                    requests.append(req)
                    rows = [outcome(target, target) for _ in req['seeds']]
                    if not req['learn']:
                        return rows, {}
                    slot.mkdir(parents=True)
                    output = slot / 'updated.pt'
                    output.write_bytes(b'updated fixture')
                    return rows, dict(output_checkpoint=str(output), output_checkpoint_sha=train.v7.digest(output))

                patches = dict(wave=wave, fingerprint=lambda: {},
                               final_seeds=lambda *args: list(range(69_000_000, 69_000_032)))
                with patch.dict(train.campaign.__globals__, patches), patch.object(train.points, 'wave', wave):
                    train.campaign(home, cfg, 0)
                self.assertEqual([r['role'] for r in requests], ['train', 'validate', 'validate', 'validate'])
                self.assertEqual(requests[0]['seeds'][0], train.TRAIN_SEED_BASE[target] + iteration * 32)
                self.assertEqual(requests[1]['seeds'][0], train.VALIDATION_SEED_BASE[target] + iteration * 32)
                self.assertEqual(len({s for req in requests for s in req['seeds']}), 64)
                state = train.v7.read(home / 'state.json')
                self.assertEqual(state['status'], f'qualified_{target}deg_state_specialist')
                policy = train.v7.read(home / 'qualified_policy.json')
                self.assertEqual(policy['target_deg'], target)
                self.assertEqual(policy['contact_point_offset_deg'], 7)


if __name__ == '__main__':
    unittest.main()
