"""Access to the browser's optional simulation control is strictly local."""
import io
import json
from types import SimpleNamespace
from email.message import Message
import unittest
from unittest.mock import Mock,patch

from cube_dashboard import server

def request(path,*,host='127.0.0.1:8765',origin=None,body=None,typ='application/json'):
    handler=server.Handler.__new__(server.Handler)
    handler.path=path;handler.command='GET' if body is None else 'POST'
    handler.server=SimpleNamespace(server_port=8765)
    handler.headers=Message();handler.headers['Host']=host
    if origin is not None:handler.headers['Origin']=origin
    payload=json.dumps(body).encode() if body is not None else b''
    handler.headers['Content-Length']=str(len(payload));handler.headers['Content-Type']=typ
    handler.rfile=io.BytesIO(payload);handler.send_json=Mock();handler.send_data=Mock()
    handler.handle_request()
    return handler

class BrowserBoundary(unittest.TestCase):
    def test_no_foreign_origin_or_host_can_start_worker(self):
        with patch('cube_dashboard.rollouts.start_rollout') as start:
            for host,origin in [('127.0.0.1:8765','http://somewhere.invalid'),('somewhere.invalid:8765',None)]:
                r=request('/api/rollouts',host=host,origin=origin,body={'model':'r15'})
                self.assertEqual(r.send_json.call_args.args[0],403)
            r=request('/api/rollouts',body={'model':'r15'},typ='text/plain');self.assertEqual(r.send_json.call_args.args[0],415)
            r=request('/api/rollouts',body={'model':'r15','seeds':[1]});self.assertEqual(r.send_json.call_args.args[0],400)
            start.assert_not_called()
    def test_busy_does_not_acknowledge_job(self):
        from cube_dashboard.rollouts import RolloutBusy
        with patch('cube_dashboard.rollouts.start_rollout',side_effect=RolloutBusy('训练占用 GPU')):
            r=request('/api/rollouts',body={'model':'r15'});self.assertEqual(r.send_json.call_args.args,(409,{'error':'训练占用 GPU'}))
    def test_controls_save_start_and_resume_have_distinct_status_and_same_origin(self):
        from cube_dashboard.controls import ControlConflict
        with patch('cube_dashboard.controls.control_action',return_value={'ok':True}) as action:
            for name in ('start','resume','apply','draft'):
                uri='/api/training/control/'+name
                denied=request(uri,origin='http://example.invalid',body={'run':None,'values':{}})
                self.assertEqual(denied.send_json.call_args.args[0],403)
                called=request(uri,body={'run':None,'values':{}})
                action.assert_called_with(server.PROJECT,name,{'run':None,'values':{}})
                self.assertEqual(called.send_json.call_args.args[0],202 if name in ('start','resume') else 200)
            action.side_effect=ControlConflict('GPU 忙')
            busy=request('/api/training/control/start',body={'run':None,'values':{}})
            self.assertEqual(busy.send_json.call_args.args,(409,{'error':'GPU 忙'}))

    def test_no_arbitrary_local_files_served(self):
        r=request('/static/../server.py');self.assertEqual(r.send_json.call_args.args[0],404)
        r.send_data.assert_not_called()
