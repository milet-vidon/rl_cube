import copy
import math
import unittest

from contact_hold_physical_metrics import summarize_hold, supported, steady

CFG = {'control_hz':30, 'robot_z':1.15, 'success':{'contact_impulse_min':1e-5, 'lift_m':.08,
       'linear_speed_max':.08, 'angular_speed_max':.6, 'hold_seconds':3.}}


def row(i):
    frame = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    return {'step':i,'height_m':.15,'table_impulse':0.,'cube_table_contact':False,
            'linear_speed':.05,'angular_speed':.5, 'cube_position':[.1,.02,1.16],
            'measured_hand_frames':{'left_palm_link':frame,'right_palm_link':copy.deepcopy(frame)},
            'contact_impulse':[.006,0.], 'drop':False,'self_contact_impulse':0.,
            'robot_table_impulse':0.,'arm_soft_limit_hit':False,'actual_arm_soft_limit_violation':False,
            'hard_limit_hit':[False]*56,'pose_reset_count':1,
            'layer_measurement':{'assembled_for_grasp':True,'maximum_relative_rotation_deg':.5}}


class PhysicalHoldTest(unittest.TestCase):
    def test_true_palm_support_alone_meets_existing_pickup_speed_limit(self):
        # With no finger contacts and entirely before tilt_hold, an actually
        # stable left palm support must count, unlike the original wrist score.
        rows=[row(i) for i in range(90)]
        result=summarize_hold(rows, CFG)
        self.assertTrue(result['physical_hold_observed'])
        self.assertTrue(result['physical_hold_at_end'])
        self.assertEqual(result['physical_steady']['frames'], 90)
        self.assertEqual(result['wrist_speed_steady']['frames'],0)
        self.assertFalse(summarize_hold(rows[1:], CFG)['physical_hold_observed'])

    def test_safety_support_floor_and_timer_reset(self):
        r=row(0)
        for change in ({'table_impulse':1e-5},{'cube_table_contact':True},{'contact_impulse':[0.,1.]},
                       {'height_m':.0799}, {'drop':True}, {'self_contact_impulse':.001},
                       {'hard_limit_hit':[True]}, {'layer_measurement':{'assembled_for_grasp':False,
                        'maximum_relative_rotation_deg':0.}}):
            with self.subTest(change=change):
                self.assertFalse(supported(r|change,CFG))
        rows=[row(i) for i in range(100)]
        rows[55].update(cube_table_contact=True)
        report=summarize_hold(rows,CFG)
        self.assertFalse(report['physical_hold_observed'])
        self.assertEqual(report['physical_steady']['step_range'],[0,54])
        self.assertEqual(report['physical_steady']['ending_streak_frames'],44)
        self.assertEqual(report['first_table_return_after_supported_lift'],55)
        self.assertFalse(steady(r|{'linear_speed':.08},CFG))
        self.assertFalse(steady(r|{'angular_speed':.6},CFG))

    def test_palm_co_moving_cube_does_not_look_like_slip(self):
        rows=[row(i) for i in range(2)]
        # Rotate left palm 90 degrees about Y and translate in *root* coords.
        # Cube offset (0.1,0.02,0.01) does not move in palm coordinates.
        rows[1]['measured_hand_frames']['left_palm_link']=[[0,0,1,.1],[0,1,0,.2],[-1,0,0,.3],[0,0,0,1]]
        rows[1]['cube_position']=[.11,.22,CFG['robot_z']+.2]
        d=summarize_hold(rows,CFG)['palm_drift_during_longest_supported'][0]
        self.assertLess(d['max_displacement_from_first_m'],1e-12)
        rows[1]['cube_position'][1]+=.02
        d=summarize_hold(rows,CFG)['palm_drift_during_longest_supported'][0]
        self.assertAlmostEqual(d['first_to_last_m'], .02)
        rows[1]['step']=3
        with self.assertRaisesRegex(ValueError,'gap'):summarize_hold(rows,CFG)

if __name__=='__main__':unittest.main()
