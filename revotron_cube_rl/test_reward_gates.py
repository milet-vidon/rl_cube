"""Pure reward-contract tests; does not import or launch Isaac."""
import json
from pathlib import Path
import unittest

from rewards import pickup_reward

CFG=json.loads((Path(__file__).parent/'config.json').read_text())


def case(**kwargs):
    required=round(CFG['success']['hold_seconds']*CFG['control_hz'])
    params=dict(cfg=CFG,distances=[.05,.05],height=.1,impulse=[.01,.01],table_impulse=0.,
                linear_speed=0.,angular_speed=0.,previous_hold=required-1,action_rate=0.,
                joint_speed=0.,drop=False,previous_potential=0.,timeout=False)
    params.update(kwargs)
    return pickup_reward(**params)


class PickupRewardTests(unittest.TestCase):
    def test_success_gates(self):
        self.assertTrue(case()[3])
        s=CFG['success']
        invalid_cases=[dict(impulse=[0.,0.]),dict(impulse=[.01,0.]),
                       dict(impulse=[s['contact_impulse_min'],.01]),
                       dict(height=s['lift_m']-.001),dict(table_impulse=.001),
                       dict(linear_speed=s['linear_speed_max']),
                       dict(angular_speed=s['angular_speed_max']),dict(drop=True)]
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                _,_,hold,success,_=case(**invalid)
                self.assertFalse(success)
                self.assertEqual(hold,0)
        self.assertEqual(case(impulse=[0.,0.])[1]['lift'],0.)
        self.assertEqual(case(drop=True)[1]['drop'],1.)

    def test_continuous_hold_and_interruption(self):
        required=round(CFG['success']['hold_seconds']*CFG['control_hz'])
        hold=0
        for tick in range(required):
            _,_,hold,success,_=case(previous_hold=hold)
            self.assertEqual(success,tick==required-1)
        _,_,hold,success,_=case(previous_hold=hold,impulse=[.01,0.])
        self.assertEqual(hold,0)
        self.assertFalse(success)
        self.assertFalse(case(previous_hold=hold)[3])

    def test_static_contact_cannot_farm_positive_reward(self):
        potential=case(previous_hold=0,height=.04)[1]['potential']
        for _ in range(100):
            reward,terms,hold,success,both=case(previous_hold=0,height=.04,previous_potential=potential)
            self.assertTrue(both)
            self.assertFalse(success)
            self.assertEqual(hold,0)
            self.assertLess(reward,0.)
            self.assertAlmostEqual(reward,(CFG.get('discount',.998)-1.)*potential-.002)
            potential=terms['potential']

    def test_terminal_potential_is_zero(self):
        for reason in [dict(),dict(drop=True),dict(previous_hold=0,timeout=True)]:
            with self.subTest(reason=reason):
                _,terms,_,_,_=case(previous_potential=.7,**reason)
                self.assertEqual(terms['terminal'],1.)
                self.assertEqual(terms['potential'],0.)
                self.assertGreater(terms['raw_potential'],0.)
                self.assertAlmostEqual(terms['potential_difference'],-.7)
                self.assertAlmostEqual(sum(terms['raw_component'].values()),terms['raw_potential'])

    def test_discounted_shaping_telescopes(self):
        gamma=.93
        cfg=dict(CFG,discount=gamma)
        for terminal in (False,True):
            with self.subTest(terminal=terminal):
                initial=.17
                potential=initial
                discounted=0.
                heights=[.0,.02,.05,.07]
                for tick,height in enumerate(heights):
                    _,terms,_,_,_=case(cfg=cfg,previous_hold=0,height=height,
                                      previous_potential=potential,
                                      timeout=terminal and tick==len(heights)-1)
                    discounted+=gamma**tick*terms['potential_difference']
                    potential=terms['potential']
                self.assertAlmostEqual(discounted,-initial+gamma**len(heights)*potential,places=12)
                if terminal:
                    self.assertEqual(potential,0.)

    def test_exact_formula_and_normalized_speed(self):
        cfg=dict(CFG,discount=.91,rewards={'legacy_weights_must_not_be_used':1e9})
        reward,terms,_,success,_=case(cfg=cfg,previous_hold=0,height=.04,
                                     previous_potential=.3,action_rate=.25,joint_speed=.4)
        expected=.91*terms['potential']-.3-.002+5.*success-.005*.25-.001*.4
        self.assertAlmostEqual(reward,expected)
        self.assertEqual(terms['joint_speed'],.4)
        self.assertAlmostEqual(terms['joint_speed_penalty'],-.0004)
        self.assertEqual(case(joint_speed=2.)[1]['joint_speed'],1.)
        self.assertEqual(case(joint_speed=-1.)[1]['joint_speed'],0.)
        self.assertAlmostEqual(case(previous_potential=.3,drop=True)[0],-.3-.002-2.)
        self.assertAlmostEqual(case(previous_potential=.3)[0],-.3-.002+5.)

    def test_default_discount(self):
        cfg={key:value for key,value in CFG.items() if key!='discount'}
        _,terms,_,_,_=case(cfg=cfg,previous_hold=0,height=.04,previous_potential=.2)
        self.assertAlmostEqual(terms['potential_difference'],.998*terms['potential']-.2)

    def test_config_reward_audit_matches_implementation(self):
        self.assertEqual(CFG['rewards'],{
            'potential_weights':{'reach':.2,'bilateral_contact':.1,'lift':.5,'hold_progress':.2},
            'step_penalty':-.002,'success_once':5.,'drop':-2.,'action_rate':-.005,
            'bounded_normalized_joint_speed':-.001,'terminal_potential':0.,
            'per_step_contact_bonus':0.,'per_step_hold_bonus':0.})


if __name__=='__main__':
    unittest.main(verbosity=2)
