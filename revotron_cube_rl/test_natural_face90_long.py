import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import natural_face90_long as code


def row(seed, params, **changes):
    data = dict(seed=seed, parameters=params, prefix_steps=5, steps=12, end_reason='post_horizon', target_deg=15,
                natural_ready=False, turn_started=False, success=False, unsafe=False, progress=0.,
                posture_quality=0., right_contact=0., hold_fraction=0.)
    data.update(changes)
    return data


class Search(unittest.TestCase):
    def test_false_success_wrong_params_and_unsafe(self):
        req = dict(seeds=[1,2], parameters=[code.ZERO,code.ZERO], role='train',target_deg=15)
        rep = dict(schema='natural_face90_rollout_v1',parent_sha=code.PARENT,role='train',reset_count=1,physics_stepped=True,
                   face='nZ',cross_lane_contacts=0,rows=[row(i,code.ZERO) for i in req['seeds']])
        for change in (dict(success=True),dict(parameters=[1]*6),dict(success=True,turn_started=True,natural_ready=True,unsafe=True)):
            data=copy.deepcopy(rep)
            data['rows'][1].update(change)
            with self.assertRaises(ValueError):code.check_report(data,req)
        self.assertEqual(len(code.check_report(rep,req)),2)

    def test_unsafe_cannot_beat_safe_and_posture_cannot_fake_turn(self):
        base = row(1,code.ZERO,natural_ready=True,posture_quality=1.)
        bad = dict(base,unsafe=True,turn_started=True,progress=1.)
        turned = dict(base,posture_quality=.5,turn_started=True,progress=.3)
        self.assertGreater(code.rank(code.stats([base])),code.rank(code.stats([bad])))
        self.assertGreater(code.rank(code.stats([turned])),code.rank(code.stats([base])))

    def test_paired_validation_and_independent_qualification(self):
        panels=[]
        def simulate(folder,role,candidates,seeds,cfg,fd):
            panels.append((folder.name,list(seeds),len(candidates)))
            return [[row(seed,p,natural_ready=True,turn_started=True,success=True,progress=1.) for seed in seeds] for p in candidates]
        config=dict(population=8,seeds_per_candidate=4,validation_seeds=8,rng=327,rounds=4,patience=3,source_sha256={})
        with tempfile.TemporaryDirectory() as tmp, patch.object(code,'evaluate',side_effect=simulate), patch.object(code,'final_seeds',return_value=list(range(69000000,69000032))):
            home=Path(tmp)
            (home/'round_001').mkdir()
            code.campaign(home,config,3)
            self.assertTrue((home/'qualified_policy.json').exists())
            self.assertEqual(code.read(home/'state.json')['mean'],code.ZERO)  # No drift on tied rollouts.
            self.assertEqual(code.read(home/'state.json')['std'],[.65]*6)
            code.campaign(home,config,3)  # Terminal resume must not re-test.
        self.assertEqual([v[0] for v in panels],['train','validate','final_test'])
        self.assertEqual(len(set(s for _,seeds,_ in panels for s in seeds)),4+8+32)

    def test_independent_test_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)
            a=code.final_seeds(home/'a',home/'registry')
            b=code.final_seeds(home/'b',home/'registry')
            self.assertFalse(set(a)&set(b))
            self.assertEqual(a,code.final_seeds(home/'a',home/'registry'))

    def test_saved_candidate_remains_evaluated_vector_after_elite_update(self):
        state=dict(incumbent=code.ZERO.copy(),mean=code.ZERO.copy(),std=[.65]*6)
        pop=code.population(state,dict(rng=5,population=8),1)
        original=copy.deepcopy(pop)
        results=[code.stats([row(i,p,natural_ready=True,posture_quality=i/8)]) for i,p in enumerate(pop)]
        code.update_distribution(state,pop,results)
        self.assertNotEqual(state['mean'],code.ZERO)
        self.assertEqual(pop,original)

    def test_plateau_exits_before_round_limit_when_only_noise_is_seen(self):
        def simulate(folder,role,candidates,seeds,cfg,fd):
            folder.mkdir(parents=True,exist_ok=True)
            return [[row(s,p,natural_ready=True,turn_started=True,progress=.02) for s in seeds] for p in candidates]
        cfg=dict(population=4,seeds_per_candidate=2,validation_seeds=2,rng=327,rounds=10,patience=3,source_sha256={})
        with tempfile.TemporaryDirectory() as tmp, patch.object(code,'evaluate',side_effect=simulate):
            home=Path(tmp)
            code.campaign(home,cfg,5)
            state=code.read(home/'state.json')
            self.assertEqual((state['status'],state['round']),('no_safe_layer_progress',3))
            self.assertFalse((home/'qualified_policy.json').exists())
