import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import quality, server, training
from cube_dashboard.test_http import request


class PairedValidationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root/'runs'/'contact_hold90_long_v2'
        self.home.mkdir(parents=True)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def round(self, number, spec, **extra):
        # (seed, ended at endpoint, held at end, ever turned). Independent flags:
        # reaching the endpoint need not imply that the cube was held or turned.
        rd = self.home/f'round_{number:03d}'; width = len(spec)
        dest = rd/'validate'/f'wave_0000_{width:04d}'/'sample_test'
        lanes = [dict(seed=seed, ever_hold=True, ending_hold=held, ever_whole_turn=turn,
                      end_reason='teacher_wrist_endpoint' if endpoint else 'tilt_hold_timeout',
                      teacher_executed_left_steps=0,
                      physical_rotation={'best_controlled_hold': {'seconds': float(seed)}})
                 for seed, endpoint, held, turn in spec]
        plan_sha = 'd' * 64
        checkpoint = str(number) * 64
        self.write(rd/'fit/result.json',dict(completed=True,checkpoint_reload_verified=True,checkpoint_sha256=checkpoint))
        report = dict(parent_sha=checkpoint,checkpoint=checkpoint,role='validate',cube_edge_m=.09,plan_sha256=plan_sha,num_envs=width,
                      per_lane=lanes,seeds=[r[0] for r in spec],student_start_phase=[0]*width,
                      control_hz=30,physics_hz=240,requested_steps=3500,action_dim=56,
                      solver={'type':'TGS'},physical_hold_definition='same fixture gate',
                      physics_device='cuda:0', code_sha256={'sampler':'d'*64},
                      **dict.fromkeys(quality._HASH_FIELDS, 'f'*64))
        report.update(extra)
        report_sha = self.write(dest/'report.json', report)
        self.write(rd/'validate/accepted_0000.json', dict(dest=str(dest),width=width,
                                                           report_sha=report_sha,plan_sha=plan_sha))
        self.write(rd/'validation_summary.json',dict(envs=width,held=width,
                    still_held_at_end=sum(r[2] for r in spec),whole_turn=sum(r[3] for r in spec),
                    mean_controlled_hold_s=sum(r[0] for r in spec)/width))
        return dest/'report.json'

    def compare(self):
        return training.comparison_snapshot(self.root,self.home.name,'1','2')

    def test_match_by_seed_with_partial_overlap_and_inverted_order(self):
        self.round(1,[(11,True,True,True),(12,True,True,False),(13,True,True,True)])
        self.round(2,[(14,True,True,True),(12,True,True,True),(11,True,False,True)])
        r=self.compare();self.assertEqual((r['status'],r['matched'],r['base_only'],r['target_only']),
                                         ('ready',2,1,1))
        self.assertEqual([x['seed'] for x in r['rows']],[11,12])
        self.assertFalse(r['rows'][0]['target']['joint_pass'])
        self.assertEqual(r['rows'][1]['base']['hold_s'],12)
        self.assertEqual(r['metrics']['joint_pass'],dict(lost=1,gained=1,both_pass=0,both_fail=0,
                                                          base_pass=1,target_pass=1))
        self.assertEqual(r['metrics']['end_held']['lost'],1)
        self.assertEqual(r['metrics']['whole_turn']['gained'],1)

    def test_no_overlap_and_damaged_receipt_do_not_create_losses(self):
        self.round(1,[(11,True,True,True)])
        report=self.round(2,[(12,False,False,False)])
        self.assertEqual((self.compare()['status'],self.compare()['metrics']),('no_overlap',{}))
        report.write_bytes(report.read_bytes()+b' ')
        r=self.compare()
        self.assertEqual((r['status'],r['target_status'],r['matched'],r['rows']),
                         ('unavailable','invalid_evidence',None,[]))

    def test_different_or_unknown_validation_conditions_hide_transition_counts(self):
        spec=[(12,True,True,True)]
        self.round(1,spec)
        for kw, status, field in [({'requested_steps':840},'protocol_mismatch','requested_steps'),
                                  ({'student_start_phase':[1]},'protocol_mismatch','student_start_phase'),
                                  ({'scene_sha256':None},'protocol_unknown','scene_sha256')]:
            with self.subTest(kw=kw):
                self.round(2,spec,**kw); r=self.compare()
                self.assertEqual((r['status'],r['matched'],r['metrics'],r['protocol_fields']),
                                 (status,1,{},[field]))

    def test_unrelated_model_receipt_blocks_comparison_but_not_old_trends(self):
        spec=[(11,True,True,True)]
        self.round(1,spec)
        self.round(2,spec,checkpoint='a'*64,parent_sha='a'*64)
        r=self.compare()
        self.assertEqual((r['status'],r['target_status'],r['metrics']),('unavailable','unverified_model',{}))
        row=quality.quality_history(self.home,[(2,self.home/'round_002')])[0]
        self.assertEqual(row['detail_status'],'verified')

    def test_http_requires_two_distinct_rounds_in_named_run(self):
        with patch.object(server,'PROJECT',self.root):
            for q in ('base=1&target=2','run=../&base=1&target=2',
                      'run='+self.home.name+'&base=1&target=1',
                      'run='+self.home.name+'&base=1&target=../../run'):
                h=request('/api/compare?'+q); self.assertEqual(h.send_json.call_args.args[0],400)
            self.round(1,[(11,True,True,True)])
            h=request('/api/compare?base=1&target=2&run='+self.home.name)
            code,body=h.send_json.call_args.args
            self.assertEqual((code,body['status'],body['target_status']), (200,'unavailable','missing_summary'))
