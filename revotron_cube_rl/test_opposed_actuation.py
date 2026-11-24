import json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from opposed_actuation import configure_right_fingers
class RightFingerGainTests(unittest.TestCase):
    def env(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        return SimpleNamespace(cfg=cfg,ids=list(reversed(range(56))),robot=SimpleNamespace(write_joint_stiffness_to_sim_index=Mock()))
    def test_only_right_fingers_change(self):
        env=self.env();report=configure_right_fingers(env,20);call=env.robot.write_joint_stiffness_to_sim_index.call_args.kwargs
        expected=[env.ids[i] for i,n in enumerate(env.cfg['joint_names']) if n.startswith('right_')]
        self.assertEqual(call['joint_ids'],expected);self.assertEqual(len(expected),21)
        self.assertTrue((call['stiffness']==20).all());self.assertFalse(report['arm_gains_changed']);self.assertFalse(report['effort_limits_changed'])
    def test_default_is_no_write(self):
        env=self.env();report=configure_right_fingers(env)
        env.robot.write_joint_stiffness_to_sim_index.assert_not_called();self.assertFalse(report['override_applied'])
    def test_unsupported_gains_rejected(self):
        for value in (-1,0,40,float('nan')):
            with self.assertRaises(ValueError):configure_right_fingers(self.env(),value)
if __name__=='__main__':unittest.main()
