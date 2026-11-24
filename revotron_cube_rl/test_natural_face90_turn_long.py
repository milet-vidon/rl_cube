"""Offline receipt, source selection, and restart tests. Never launches a simulator."""
import contextlib
import fcntl
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import natural_face90_recipe_long as guided
import natural_face90_v6_long as turn


class ContactTurnLauncher(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.parent = Path(tmp.name)/'r27_recipe_guided_fixture'
        self.parent.mkdir()
        self.pins = {name:f'fixture-{name}' for name in turn.FILES}
        self.old_pins = {name:self.pins[name] for name in guided.FILES}
        self.old_cfg = dict(source_sha256=self.old_pins, schema=guided.GUIDED_SCHEMA,
                            parent_sha=turn.PARENT, rng=327)
        turn.save(self.parent/'config.json', self.old_cfg)

    def make_guided_round(self, k=50):
        folder = self.parent/f'iteration_{k:04d}'
        slot = folder/'train'
        attempt = slot/'attempt_12345678'
        attempt.mkdir(parents=True)
        req = guided.request(self.old_cfg, 'train', [45_000_111], k, None)
        for plan in (slot/'request.json',attempt/'request.json'):
            turn.save(plan, req)
        files = dict(post_evidence_sha256='post_measurements.jsonl', controller_evidence_sha256='controller_states.jsonl',
                     collision_evidence_sha256='collision_events.jsonl')
        for name in (*files.values(),'policy_transitions.jsonl'):
            (attempt/name).write_text('')
        ck = attempt/'checkpoint.pt'
        ck.write_bytes(f'fixture {k}'.encode())
        learning = dict(kind='state_PPO_specialist', deterministic=False, input_checkpoint_sha=None,
                        actor_updated=True, actor_max_delta=.001, learned_transitions=32, real_transitions=32,
                        output_checkpoint=str(ck), output_checkpoint_sha=turn.digest(ck),
                        transition_evidence_sha=turn.digest(attempt/'policy_transitions.jsonl'))
        row = dict(seed=req['seeds'][0], parameters=turn.ZERO, target_deg=15, end_reason='post_horizon',
                   prefix_steps=100, steps=300, unsafe=False, success=False, turn_started=False, natural_ready=True,
                   posture_quality=.5, right_contact=.2, hold_fraction=0., progress=0.)
        report = dict(schema='natural_face90_rollout_v1', role='train', parent_sha=turn.PARENT,
                      rows=[row], reset_count=1, physics_stepped=True, cross_lane_contacts=False,
                      face='nZ', software=req['software'], request_sha256=turn.digest(attempt/'request.json'),
                      expert_learning=learning, **{key:turn.digest(attempt/value) for key,value in files.items()})
        turn.save(attempt/'report.json', report)
        turn.save(slot/'accepted.json', dict(request_sha=turn.digest(slot/'request.json'),
                  report_sha=turn.digest(attempt/'report.json'), attempt=attempt.name))
        turn.save(folder/'result.json', dict(iteration=k, learning=learning))
        turn.save(self.parent/'state.json',dict(iteration=k, checkpoint=str(ck)))
        return folder, ck

    def test_latest_and_explicit_round_pinning(self):
        original, ck = self.make_guided_round()
        newer, _ = self.make_guided_round(51)
        (self.parent/'iteration_0052').mkdir()
        self.assertEqual(turn.transfer_origin(self.parent,self.pins)['folder'],str(newer))
        origin = turn.transfer_origin(original,self.pins)
        self.assertEqual(origin['checkpoint'],str(ck))
        self.assertEqual(origin['schema'],guided.GUIDED_SCHEMA)
        with self.assertRaises(ValueError):
            turn.transfer_origin(self.parent/'iteration_0052',self.pins)

    def test_accepted_origin_hash_and_schema_must_match(self):
        folder, ck = self.make_guided_round()
        changed = dict(self.pins)
        changed['natural_face90_recipe_guide.py'] = 'modified'
        with self.assertRaises(ValueError):
            turn.transfer_origin(folder,changed)
        (folder/'train'/'attempt_12345678'/'collision_events.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence'):
            turn.transfer_origin(folder,self.pins)
        (folder/'train'/'attempt_12345678'/'collision_events.jsonl').write_text('')
        ck.write_bytes(b'wrong weights')
        with self.assertRaisesRegex(ValueError,'PPO parameter'):
            turn.transfer_origin(folder,self.pins)
        turn.save(self.parent/'config.json',dict(self.old_cfg,schema=turn.SCHEMA))
        with self.assertRaises(ValueError):
            turn.transfer_origin(folder,self.pins)

    def test_active_source_fails_without_starting_a_competing_run(self):
        folder, _ = self.make_guided_round()
        with (self.parent/'campaign.lock').open('ab') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, '源训练仍在运行'):
                turn.check_source_idle(folder)
            self.assertEqual(turn.transfer_origin(self.parent,self.pins)['iteration'],50)
        turn.check_source_idle(self.parent)

    def test_transfer_request_schema_roles_and_source_hash_must_match(self):
        folder, ck = self.make_guided_round()
        origin = turn.transfer_origin(folder,self.pins)
        cfg = dict(schema=turn.SCHEMA, rng=327, source_sha256=self.pins, origin=origin)
        req = turn.request(cfg, 'train', [46_000_001,46_000_002], 51, str(ck))
        turn.validate_request(req)
        with self.assertRaises(ValueError):
            turn.validate_request(dict(req, transfer=dict(origin,schema=turn.SCHEMA)))
        with self.assertRaises(ValueError):
            turn.validate_request(dict(req, role='validate',learn=False))
        with self.assertRaises(ValueError):
            turn.validate_request(dict(req, checkpoint_sha='1'*64))
        with self.assertRaises(ValueError):
            turn.validate_request(dict(req, recipe_schema=guided.GUIDED_SCHEMA))
        self.assertNotIn('transfer',turn.request(cfg, 'train',req['seeds'],52,str(ck)))

    def test_child_worker_path_inherited_gpu_fds_and_no_implicit_retry(self):
        folder, ck = self.make_guided_round()
        origin = turn.transfer_origin(folder,self.pins)
        cfg = dict(schema=turn.SCHEMA, rng=327, source_sha256=self.pins, origin=origin)
        req = turn.request(cfg,'train',[46_000_000,46_000_001],51,str(ck))
        slot = self.parent.parent/'child'/'train'
        with patch.object(turn, 'available_gpu', return_value=contextlib.nullcontext([11,12])), \
             patch.object(turn.subprocess,'run',return_value=SimpleNamespace(returncode=4)) as run:
            with self.assertRaisesRegex(RuntimeError,'exited 4'):
                turn.wave(slot,req,77)
        self.assertEqual(run.call_count,1)
        self.assertEqual(Path(run.call_args.args[0][3]).name,'natural_face90_v6_worker.py')
        self.assertEqual(run.call_args.kwargs['pass_fds'],(77,11,12))
        self.assertFalse((slot/'accepted.json').exists())
        self.assertEqual(turn.read(slot/'request.json'),req)

    def test_new_turn_evidence_is_verified_on_initial_acceptance_and_receipt_reuse(self):
        folder, ck = self.make_guided_round()
        origin = turn.transfer_origin(folder,self.pins)
        cfg = dict(schema=turn.SCHEMA, rng=327, source_sha256=self.pins, origin=origin)
        req = turn.request(cfg,'train',[46_000_000,46_000_001],51,str(ck))
        slot = self.parent.parent/'child'/'train'
        learning = {}
        def write_fixture(cmd, **kwargs):
            attempt = Path(cmd[-1])
            attempt.mkdir()
            turn.save(attempt/'report.json', {})
            (attempt/'turn_contact_evidence.jsonl').write_text('fixture evidence')
            return SimpleNamespace(returncode=0)
        with patch.object(turn,'available_gpu',side_effect=lambda:contextlib.nullcontext([11,12])), \
             patch.object(turn.subprocess,'run',side_effect=write_fixture) as run, \
             patch.object(turn,'verify_result',return_value=([],learning)):
            with self.assertRaisesRegex(ValueError,'turn-contact evidence'):
                turn.wave(slot,req,77)
            self.assertFalse((slot/'accepted.json').exists())
            evidence = next(slot.glob('attempt_*/turn_contact_evidence.jsonl'))
            learning['turn_evidence_sha256'] = turn.digest(evidence)
            turn.wave(slot,req,77)
            self.assertEqual(run.call_count,2)
            attempt = slot/turn.read(slot/'accepted.json')['attempt']
            (attempt/'turn_contact_evidence.jsonl').write_text('subsequent change')
            with self.assertRaisesRegex(ValueError,'turn-contact evidence'):
                turn.wave(slot,req,77)
            self.assertEqual(run.call_count,2)

    def test_first_validation_and_committed_sha_after_interrupt(self):
        folder, ck = self.make_guided_round()
        origin = turn.transfer_origin(folder,self.pins)
        home = self.parent.parent/'child'; home.mkdir()
        cfg = dict(schema=turn.SCHEMA, rng=327, source_sha256=self.pins, origin=origin,
                   iterations=53, lanes=32, validation_seeds=16, validate_every=10)
        new_ck = home/'new.pt'; new_ck.write_bytes(b'fixture')
        score = dict(unsafe=1, successes=0, progress=0., natural=0.)
        def mock_wave(slot,req,fd):
            turn.validate_request(req)
            if req['iteration'] == 52:
                self.assertNotIn('transfer',req)
                raise InterruptedError
            self.assertEqual(req['checkpoint'],str(ck)); self.assertEqual(req['transfer'],origin)
            self.assertEqual(req['seeds'][0],46_000_000+50*32)
            slot.mkdir(parents=True)
            return [],dict(output_checkpoint=str(new_ck),output_checkpoint_sha=turn.digest(new_ck))
        with patch.object(turn,'wave',side_effect=mock_wave), patch.object(turn,'fingerprint',return_value=self.pins), \
             patch.object(turn,'stats',return_value=score), patch.object(turn,'panel',return_value=score) as valid:
            with self.assertRaises(InterruptedError):
                turn.campaign(home,cfg,0)
        self.assertEqual(valid.call_count,1)
        self.assertEqual(valid.call_args.args[2:4],(51,str(new_ck)))
        self.assertEqual(valid.call_args.args[4][0],66_000_000+50*32)
        state=turn.read(home/'state.json')
        self.assertEqual(state['iteration'],51)
        self.assertEqual(state['checkpoint_sha'],turn.digest(new_ck))
        new_ck.write_bytes(b'tampered')
        with patch.object(turn,'fingerprint',return_value=self.pins), patch.object(turn,'wave') as wave:
            with self.assertRaisesRegex(ValueError,'parent checkpoint changed'):
                turn.campaign(home,cfg,0)
            wave.assert_not_called()


if __name__ == '__main__':
    unittest.main()
