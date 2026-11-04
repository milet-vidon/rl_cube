"""Independent narrow V9 completed-audit adaptation checks; no optimizer."""
from pathlib import Path
import hashlib,unittest
import audit_completed_arm_hold_increment_v9 as audit
from test_train_arm_hold_increment_v9 import fixtures


class V9CompletedAuditPeer(unittest.TestCase):
    def test_inverse_version_and_560_condition_restore_exact_v8_audit(self):
        source=Path(audit.__file__).read_text()
        pairs=[('import train_arm_hold_increment_v9 as t','import train_hold_increment_v8 as t'),
            ("==final['step560_regression_violation_count']",''),
            ('completed_v9_nine_history_cpu_product_replay_v1','completed_v8_eight_history_cpu_product_replay_v1'),
            ('Nine previously reviewed','Eight previously reviewed'),
            ('V9_COMPLETED_PRODUCT_AUDIT','V8_COMPLETED_PRODUCT_AUDIT')]
        for old,new in pairs:
            self.assertEqual(source.count(old),1);source=source.replace(old,new)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),'2ce8b0a3d1c2b67b0fffda0db5c254c548daa34766ccc50b009c3aa552fc1e96')

    def test_full_nine_history_detailed_error_inventory_includes_index560(self):
        model,episodes=fixtures();metrics=audit.t.assess(episodes,model,'cpu',streaming=True)
        mistakes,counts=audit.errors_by_actual_history(episodes,model)
        self.assertEqual(len(mistakes),metrics['classification_error_count'])
        self.assertEqual(sum(r['class_errors']for r in counts),len(mistakes));self.assertEqual(len(counts),9)
        self.assertEqual(len(mistakes),17);self.assertEqual(sum(r['step']==560 for r in mistakes),3)
        self.assertEqual(sum(r['step']==545 for r in mistakes),6);self.assertEqual(sum(r['step']==499 for r in mistakes),8)
        self.assertEqual(metrics['step560_regression_violation_count'],3)


if __name__=='__main__':unittest.main()
