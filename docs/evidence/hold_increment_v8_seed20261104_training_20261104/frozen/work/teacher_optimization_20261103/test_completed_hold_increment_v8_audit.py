"""Check detailed audit inventory against the independent synthetic histories."""
import unittest
import audit_completed_hold_increment_v8 as audit
from test_train_hold_increment_v8_peer import fixtures


class CompletedAuditInventory(unittest.TestCase):
    def test_detailed_errors_reproduce_full_causal_assessment(self):
        _,model,episodes=fixtures()
        metrics=audit.t.assess(episodes,model,'cpu',streaming=True)
        mistakes,counts=audit.errors_by_actual_history(episodes,model)
        self.assertEqual(len(mistakes),metrics['classification_error_count'])
        self.assertEqual(sum(r['class_errors']for r in counts),len(mistakes))
        self.assertEqual(len(mistakes),14)
        self.assertEqual(sum(r['step']==499 for r in mistakes),8)
        self.assertEqual(sum(r['step']==545 for r in mistakes),6)
        self.assertTrue(all(r['predicted_class']==audit.t.policy.BASE for r in mistakes))
        self.assertTrue(all(len(r['logits'])==3 and r['absolute_projected_error_rad']>0 for r in mistakes))


if __name__=='__main__':unittest.main()
