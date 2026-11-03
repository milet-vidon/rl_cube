import unittest
from turn_pacer import TurnPacer

class TurnPacerTests(unittest.TestCase):
    def test_layer_lag_repeats_last_target_then_resumes(self):
        p=TurnPacer([0,1,2,3,4],max_lead_deg=2.)
        self.assertEqual([p.next_index(0,True)[0] for _ in range(4)],[0,1,2,2])
        self.assertEqual(p.next_index(1.1,True),(3,None))
        self.assertEqual(p.next_index(2.1,True),(4,None))
        self.assertEqual(p.next_index(4,True),(None,None));self.assertTrue(p.completed)
    def test_contact_loss_blocks_advance_even_when_angle_tracks(self):
        p=TurnPacer([0,1,2]);p.next_index(0,True)
        self.assertEqual(p.next_index(2,False),(0,None))
        self.assertEqual(p.next_index(2,True),(1,None))
    def test_permanent_stall_fails_without_unbounded_wait(self):
        p=TurnPacer([0,3],stall_ticks=2)
        p.next_index(0,True);p.next_index(0,True);p.next_index(0,True)
        self.assertEqual(p.next_index(0,True),(None,'turn_contact_or_progress_stalled'))
        self.assertEqual(p.report()['held_ticks'],2)
        self.assertEqual(p.report()['control_ticks'],3)
    def test_progress_cannot_extend_total_budget_forever(self):
        p=TurnPacer([0,1,2,3,4],stall_ticks=20,extra_ticks=1)
        for _ in range(6):p.next_index(0,True)
        self.assertEqual(p.next_index(0,True),(None,'turn_pacing_step_budget'))
    def test_invalid_feedback_and_reverse_paths_fail(self):
        self.assertEqual(TurnPacer([0,1]).next_index(float('nan'),True),(None,'nonfinite_layer_feedback'))
        for angles in ([0,2,1],[],[float('inf')]):
            with self.assertRaises(ValueError):TurnPacer(angles)

if __name__=='__main__':unittest.main()
