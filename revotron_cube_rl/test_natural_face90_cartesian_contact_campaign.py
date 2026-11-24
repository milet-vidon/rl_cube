"""Offline contact-curriculum campaign tests with synthetic paired physical readbacks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_face90_cartesian as cart
import natural_face90_cartesian_contact as campaign
from test_natural_face90_hover_d_metrics import outcome, sample


PAIRED = list(cart.PAIRED_SEEDS)
FIXED = list(cart.anchor.metrics.FIXED_SEEDS)


def write_wave(slot, req, *, first=(), driven=(), unsafe=()):
    """Create an accepted rollout with independently scorable contact/turn intervals."""
    attempt = slot / 'attempt_1234abcd'
    attempt.mkdir(parents=True)
    all_frames, rows = [], []
    for lane, seed in enumerate(req['seeds']):
        # An initial other-digit touch and a later isolated drive are distinct
        # events. This lets tests split first-touch and +2° across different lanes.
        first_touch = sample(2, index=.01) if seed in first else sample(2, other=.01)
        frames = [sample(1), first_touch, sample(3, seen=True, index=.01),
                  sample(4, seen=True, index=.01, drive=True, peak=2.1 if seed in driven else .01)]
        for frame in frames:
            frame.update(seed=seed, lane=lane)
        row = outcome(frames, seed=seed, lane=lane, unsafe=seed in unsafe,
                      post_samples=len(frames), final_face_deg=frames[-1]['face_deg'],
                      natural_ready=True, progress=0., right_contact=1., posture_quality=1.,
                      hold_fraction=1., peak_index_driven_deg=frames[-1]['face_deg'])
        all_frames.extend(frames)
        rows.append(row)
    count = len(req['seeds'])
    learning = dict(learned_lanes=count, learned_transitions=len(all_frames),
                    pad_preshape=dict(ready_step_by_lane=[1]*count, handoff_error_rad_by_lane=[0.]*count,
                                      orientation=dict(feasible_by_lane=[True]*count)))
    learning.update(pad_orientation=learning['pad_preshape']['orientation'], cartesian=dict(action_dim=3), placo=req['placo'])
    if req['learn']:
        ck = attempt/'checkpoint.pt'
        ck.write_bytes(f"synthetic update {req['iteration']}".encode())
        learning.update(output_checkpoint=str(ck), output_checkpoint_sha=cart.v7.digest(ck))
    trace = attempt/'turn_contact_evidence.jsonl'
    trace.write_text(''.join(json.dumps(r)+'\n' for r in all_frames))
    learning['turn_evidence_sha256'] = cart.v7.digest(trace)
    report = attempt/'report.json'
    cart.v7.save(report, dict(seeds=req['seeds'], rows=rows, expert_learning=learning))
    cart.v7.save(slot/'request.json', req)
    cart.v7.save(slot/'accepted.json', dict(attempt=attempt.name, request_sha=cart.v7.digest(slot/'request.json'),
                                           report_sha=cart.v7.digest(report)))
    return rows, learning


def read_verified_fixture(report, req):
    saved = cart.v7.read(report)
    learning = saved['expert_learning']
    if cart.v7.digest(report.parent/'turn_contact_evidence.jsonl') != learning['turn_evidence_sha256']:
        raise ValueError('Changed contact evidence in synthetic physical fixture')
    return saved['rows'], learning


class ContactCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = cart.self_check()['receipt']

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probe = self.root/'r27_recipe_probe'
        self.contact = self.root/'r27_recipe_contact'
        self.face = self.root/'r27_recipe_face'
        self.probe.mkdir()
        self.geometry = {'test': 1}
        self.common = dict(rng=123, geometry_sha256=cart.wrist.geometry_hash(self.geometry),
                           scene_sha256='a'*64, pad_orientation_geometry=self.geometry,
                           placo=self.solver, source_sha256=campaign.fingerprint(),
                           probe_dir=str(self.probe), probe_summary_sha256='b'*64)
        self.neutral = dict(baseline=dict(unsafe_seeds=[]), candidate=dict(unsafe_seeds=[]),
                            paired_contact_gate_met=False)
        self.origin = dict(iteration=0, checkpoint=None, sha=None)
        self.cfg = dict(self.common, schema=campaign.CONTACT, curriculum_stage='contact2',
                        lanes=4, iterations=1, updates=1, validate_every=1, origin=self.origin)
        self.qualifying = set(FIXED[:2])
        self.args = ['--probe-dir', str(self.probe), '--contact-dir', str(self.contact),
                     '--run-dir', str(self.face), '--updates','1','--lanes','4','--validate-every','1']

    def test_failed_neutral_probe_is_verified_and_authorizes_pretraining(self):
        archive = self.root/'anchor.json'
        plan = dict(schema=cart.PROBE_SCHEMA, seeds=PAIRED, rng=self.common['rng'],
                    source_sha256=cart.fingerprint(), input_summary=str(archive), input_summary_sha256='input',
                    scene_sha256=self.common['scene_sha256'], geometry_sha256=self.common['geometry_sha256'],
                    placo=self.solver)
        cart.v7.save(self.probe/'plan.json', plan)
        initial_cfg = dict(self.common, source_sha256=plan['source_sha256'])
        neutral_req = cart.request(initial_cfg,'validate',PAIRED,0,None)
        write_wave(self.probe/'probe',neutral_req,first=FIXED,driven=())
        reference = dict(unsafe_seeds=[])
        prepared = (dict(orientation_plan=self.common, input_summary_sha256='input'),
                    dict(policy_seed=123, pad_orientation_geometry=self.geometry),reference,reference)
        with (patch.object(cart,'verify_report',side_effect=read_verified_fixture),
              patch.object(cart.anchor.metrics,'_geometry'),
              patch.object(cart.preshape,'prepare', return_value=prepared),
              patch.object(cart,'self_check',return_value={'receipt':self.solver})):
            result = cart.probe_result(self.probe/'probe',neutral_req,plan,reference,reference)
            self.assertFalse(result['paired_contact_gate_met'])
            cart.v7.save(self.probe/cart.SUMMARY, result)
            common, checked = campaign.verified_baseline(self.probe)
            self.assertEqual(checked, result)
            self.assertEqual(common['probe_summary_sha256'], cart.v7.digest(self.probe/cart.SUMMARY))
            cart.v7.save(self.probe/cart.SUMMARY,dict(result,paired_contact_gate_met=True))
            with self.assertRaisesRegex(ValueError, 'Neutral summary'):
                campaign.verified_baseline(self.probe)

    def test_distinct_stage_and_evaluation_seed_roles(self):
        train = campaign.train_req(self.cfg,1,None)
        campaign.validate_request(train)
        self.assertIsNone(train['checkpoint'])
        ck = self.root/'contact.pt'
        ck.write_bytes(b'contact')
        paired = campaign.request(self.cfg, 'validate', PAIRED, 1, str(ck))
        campaign.validate_request(paired)
        face_cfg = dict(self.cfg, schema=campaign.FACE, curriculum_stage='face15',
                        origin=dict(iteration=1,checkpoint=str(ck),sha=cart.v7.digest(ck)))
        face = campaign.train_req(face_cfg,2,str(ck))
        campaign.validate_request(face)
        self.assertEqual(face['contact_origin'],face_cfg['origin'])
        dev = campaign.seed_block(campaign.VALID_BASE,2,8)
        held = list(range(69_000_000,69_000_008))
        pools = [train['seeds'], face['seeds'], PAIRED, dev, held]
        self.assertEqual(sum(map(len,pools)), len(set().union(*pools)))
        for seeds in (dev, held):
            campaign.validate_request(campaign.request(face_cfg,'validate',seeds,2,str(ck)))
        for bad in (dict(train,seeds=PAIRED), dict(train,paired_validation=True),
                    dict(paired,seeds=dev), dict(paired,checkpoint=None,checkpoint_sha=None),
                    dict(face,seeds=train['seeds']), dict(face,contact_origin=dict(face['contact_origin'],sha='z'*64))):
            with self.subTest(bad=bad['role']), self.assertRaises(ValueError):
                campaign.validate_request(bad)

    def test_paired_gate_requires_joint_evidence_per_seed_and_full_group_safety(self):
        ck = self.root/'contact.pt'
        ck.write_bytes(b'contact')
        req = campaign.request(self.cfg,'validate',PAIRED,1,str(ck))
        first = FIXED[:2]
        another = next(s for s in PAIRED if s not in FIXED)
        for i,(driven,unsafe,met) in enumerate(((FIXED[1:3],[],False),(first,[],True),(first,[another],False))):
            with self.subTest(driven=driven,unsafe=unsafe):
                slot = self.root/f'wave_{i}'
                _, learn = write_wave(slot,req,first=first,driven=driven,unsafe=unsafe)
                with patch.object(cart.anchor.metrics,'_geometry'):
                    gate = campaign.paired_gate(cart.v7.accepted_report(slot,req),req,self.neutral,learn)
                self.assertEqual(gate['paired_contact_gate_met'],met)
                self.assertEqual(gate['qualifying_fixed_seeds'],sorted(set(first)&set(driven)))
                self.assertEqual(gate['fixed_11']['episodes'],11)
                self.assertEqual(gate['all_16']['unsafe'],len(unsafe))

    def run_long(self, *, passing, requests, stages):
        def mock_wave(slot,req,fd):
            campaign.validate_request(req)
            requests.append(req)
            return write_wave(slot, req, first=self.qualifying, driven=self.qualifying if passing else ())

        def run_stage(command,check):
            self.assertTrue(check)
            args = command[command.index(str(Path(campaign.__file__).resolve()))+1:]
            stages.append(args[0])
            campaign.main(args)

        with (patch.object(campaign.points,'RUNS',self.root),
              patch.object(campaign,'verified_baseline',return_value=(self.common,self.neutral)),
              patch.object(campaign,'wave',side_effect=mock_wave),
              patch.object(campaign,'verify_report',side_effect=read_verified_fixture),
              patch.object(cart.anchor.metrics,'_geometry'),
              patch.dict(campaign.face_campaign.__globals__, {'wave':mock_wave}),
              patch.dict(campaign._panel.__globals__, {'wave':mock_wave}),
              patch.object(campaign.subprocess,'run',side_effect=run_stage),redirect_stdout(io.StringIO())):
            campaign.main(['long',*self.args])

    def test_long_reuses_certified_trained_checkpoint_and_completed_stages(self):
        waves, stages = [], []
        self.run_long(passing=True,requests=waves,stages=stages)
        self.assertEqual(stages,['train','train15'])
        train, paired, face, *valid = waves
        self.assertIsNone(train['checkpoint'])
        ck = cart.v7.read(self.contact/'state.json')
        self.assertEqual(ck['status'],'contact_ready')
        self.assertEqual(paired['seeds'],PAIRED)
        self.assertEqual((paired['checkpoint'],paired['checkpoint_sha']),(ck['checkpoint'],ck['checkpoint_sha']))
        self.assertEqual((face['checkpoint'],face['checkpoint_sha']),(ck['checkpoint'],ck['checkpoint_sha']))
        self.assertEqual(face['contact_origin']['iteration'],ck['iteration'])
        self.assertTrue(all(v['role']=='validate' for v in valid))
        self.assertEqual(cart.v7.read(self.face/'state.json')['status'],'iteration_limit')
        waves.clear()
        self.run_long(passing=True,requests=waves,stages=[])
        self.assertFalse(waves)

        # Substituting a different actor just for the evaluation invalidates promotion.
        slot=self.contact/'iteration_0001/validate/wave_0000'
        req=cart.v7.read(slot/'request.json')
        cart.v7.save(slot/'request.json',dict(req,checkpoint='replacement'))
        with (patch.object(campaign,'verify_report',side_effect=read_verified_fixture),
              self.assertRaisesRegex(ValueError,'accepted wave')):
            campaign.verified_origin(self.contact,self.common,self.neutral)

    def test_false_raw_gate_or_forged_contact_ready_cannot_start_train15(self):
        waves, stages = [], []
        self.run_long(passing=False,requests=waves,stages=stages)
        self.assertEqual(stages,['train'])
        self.assertEqual([w['role'] for w in waves],['train','validate'])
        self.assertFalse(self.face.exists())
        state=cart.v7.read(self.contact/'state.json')
        self.assertEqual(state['status'],'iteration_limit')
        # A certificate and state that appear to pass must still be compared
        # with the accepted post-update checkpoint and its original physical frames.
        forged = dict(state,status='contact_ready')
        cart.v7.save(self.contact/'state.json',forged)
        folder=self.contact/'iteration_0001'
        result=cart.v7.read(folder/'result.json')
        fake_gate = dict(result['paired_gate'],paired_contact_gate_met=True)
        cart.v7.save(folder/'result.json',dict(result,paired_gate=fake_gate))
        req=cart.v7.read(folder/'validate/wave_0000/request.json')
        report=cart.v7.accepted_report(folder/'validate/wave_0000',req)
        cfg=cart.v7.read(self.contact/'config.json')
        cart.v7.save(self.contact/'contact_ready.json',campaign.certificate(cfg,forged,report,fake_gate))
        with (patch.object(campaign,'verify_report',side_effect=read_verified_fixture),
              patch.object(cart.anchor.metrics,'_geometry'),
              self.assertRaisesRegex(ValueError,'failed verification')):
            campaign.verified_origin(self.contact,self.common,self.neutral)


if __name__=='__main__':
    unittest.main()
