"""Round-trip recorded measurements to a browser publisher, entirely on CPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from natural_face90_demo_record import FRAMES,META,SUMMARY,Recorder
from natural_face90_demo_replay import replay
from test_natural_face90_demo_record import FakeEnv,Q0


class EndReplay(Exception):
    pass


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        e = FakeEnv()
        r = Recorder(self.path,e)
        self.addCleanup(r.close)
        e.frame([[]]*8)
        r.record(0,'pickup',{},Q0,Q0)
        r.begin_handoff(1,Q0)
        e.frame([[]]*8)
        self.row = r.record(1,'manual',{},Q0,Q0)
        r.finish('operator_saved')

    def test_loops_manual_segment_without_reconstructing_geometry_or_advancing_physics(self):
        ui = Mock()
        ui.publish.side_effect=[None,EndReplay()]
        with patch('natural_face90_demo_replay.time.sleep') as wait, self.assertRaises(EndReplay):
            replay(ui,self.path/FRAMES)
        self.assertEqual(wait.call_count,1)
        for call in ui.publish.call_args_list:
            frame = call.args[0]
            self.assertEqual((frame['phase'],frame['recorded_phase'],frame['step']),('replay','manual',1))
            self.assertEqual(frame['bodies'],self.row['bodies'])
            self.assertEqual(frame['cube'],self.row['cube'])
            self.assertEqual(frame['frame_count'],1)

    def test_tamper_and_legacy_fail_before_any_frame_is_published(self):
        ui = Mock()
        path = self.path/FRAMES
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"angle_deg":0.0',b'"angle_deg":1.0'))
        with self.assertRaisesRegex(ValueError,'differ'):
            replay(ui,self.path)
        path.write_bytes(raw)
        meta = json.loads((self.path/META).read_bytes())
        meta['schema']='r27_human_two_finger_demo_v1'
        (self.path/META).write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'older files'):
            replay(ui,self.path)
        ui.publish.assert_not_called()


if __name__=='__main__':
    unittest.main()
