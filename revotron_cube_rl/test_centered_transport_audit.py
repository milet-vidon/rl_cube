import copy,json,unittest
from pathlib import Path
from centered_transport_audit import assess_center_row

class CenterAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture=json.loads(Path(__file__).with_name('center_audit_fixture.json').read_text())
        self.row=copy.deepcopy(self.fixture['row'])
    def audit(self):
        f=self.fixture
        return assess_center_row(self.row,f['joint_names'],f['goal_position'],f['goal_quaternion'])
    def test_measured_center_hold(self):self.assertTrue(self.audit()['qualified'])
    def test_old_side_position_rejected(self):
        self.row['cube_position'][1]=.287
        self.assertIn('center_position',self.audit()['failed_conditions'])
    def test_reversed_wrist_rejected(self):
        self.row['q'][self.fixture['joint_names'].index('Joint6_L')]=-.2
        self.assertIn('wrist_direction',self.audit()['failed_conditions'])
    def test_torso_contact_rejected(self):
        self.row['self_contact_impulse']=.149
        self.assertIn('safe',self.audit()['failed_conditions'])
    def test_deformed_layers_rejected(self):
        self.row['layer_measurement']['assembled_for_grasp']=False
        self.assertIn('cube_assembled',self.audit()['failed_conditions'])
    def test_cached_elbow_metrics_cannot_override_positions(self):
        self.row['measured_elbow']['elbow_root_m'][1]=0
        self.assertIn('elbow_outward',self.audit()['failed_conditions'])
if __name__=='__main__':unittest.main()
