"""Remote telemetry is additive, bounded and never a training control surface."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import remote, remote_reader as reader
from cube_dashboard.test_http import request


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'runs';self.root.mkdir()
        self.home=self.root/'r27_recipe_probe_fixture';self.home.mkdir()
        self.write('config.json',dict(schema='natural_face90_recipe_state_ppo_v1',smoke=True,lanes=8))
        self.batch=self.home/'smoke/attempt_fixture';self.batch.mkdir(parents=True)
        self.write('smoke/attempt_fixture/request.json',dict(role='smoke',iteration=1))
        self.gpu=patch.object(reader,'gpu',return_value={'available':False});self.gpu.start();self.addCleanup(self.gpu.stop)

    def write(self,path,value):
        p=self.home/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(value))

    def hashes(self):
        return {str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}

    def test_observations_do_not_become_success_and_no_files_changed(self):
        self.write('smoke/attempt_fixture/progress.json',dict(step=1535,entered_post=4))
        (self.batch/'post_measurements.jsonl').write_text(json.dumps(dict(lane=0,face_deg=15,natural_ready=True))+'\n')
        # A checkpoint is neither deserialized nor imported by the monitor.
        (self.batch/'checkpoint.pt').write_bytes(b'not a pickle')
        before=self.hashes()
        s=reader.snapshot(self.root,self.home.name,{self.home.name:[dict(pid=123,state='S',attempt=str(self.batch))]})
        self.assertEqual(s['status'],'running');self.assertEqual(s['kind'],'diagnostic')
        self.assertIsNone(s['summary']);self.assertFalse(s['qualified'])
        self.assertEqual(s['live_episodes'][0]['face_deg'],15)
        self.assertEqual(before,self.hashes())

    def test_failed_init_is_not_completed_or_running(self):
        self.write('smoke/attempt_fixture/failed.json',dict(error='missing frame'))
        s=reader.snapshot(self.root,self.home.name,{})
        self.assertEqual(s['status'],'failed');self.assertEqual(s['error'],'missing frame')

    def test_completed_diagnostic_and_validation_are_separate(self):
        row=dict(lane=0,seed=1,success=False,unsafe=True,natural_ready=True,prefix_endpoint_held=True,target_contact_steps=2)
        self.write('smoke/attempt_fixture/report.json',dict(rows=[row],expert_learning={'actor_updated':True}))
        self.write('smoke.json',dict(rows=[row],diagnostic_only=True))
        self.write('iteration_0010/result.json',dict(iteration=10,training_rollouts_before_update={'successes':7},validation_after_update={'successes':2},independent_test={'successes':1}))
        s=reader.snapshot(self.root,self.home.name,{})
        self.assertEqual(s['status'],'completed');self.assertEqual(s['summary']['successes'],0)
        self.assertEqual(s['summary']['unsafe'],1);self.assertFalse(s['qualified'])
        self.assertEqual(s['history'][0]['train']['successes'],7)
        self.assertEqual(s['history'][0]['validation']['successes'],2)
        self.assertEqual(s['history'][0]['independent_test']['successes'],1)

    def test_paths_and_commands_are_restricted(self):
        for command in ('uname','snapshot ../config','snapshot r27_recipe_probe_fixture extra','catalog; uname'):
            with self.assertRaises(ValueError):reader.handle(command,self.root)
        outside=Path(self.tmp.name)/'outside';outside.mkdir()
        (self.root/'r27_recipe_escape').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):reader.run_home(self.root,'r27_recipe_escape')
        (outside/'secret').write_text('unrelated')
        (self.home/'external.json').symlink_to(outside/'secret')
        self.assertEqual(reader.load(self.home/'external.json',self.home),{})

    def test_local_experiment_is_not_in_remote_catalog(self):
        local=self.root/'contact_hold90_long_v2';local.mkdir();(local/'state.json').write_text('{"best_round":27}')
        before=self.hashes()
        self.assertEqual([r['id'] for r in reader.catalog(self.root,{})],[self.home.name])
        self.assertEqual(before,self.hashes())


class BridgeTests(unittest.TestCase):
    def setUp(self):
        remote.CACHE.clear();self.addCleanup(remote.CACHE.clear)
        self.cfg=dict(id='nexus5090',label='Nexus-5090-1',host='10.3.22.31',user='nexus',identity_file='/private/key',known_hosts_file='/private/known_hosts')

    def test_disconnect_preserves_last_good_data_and_marks_stale(self):
        payload=dict(schema='cube_remote_monitor_v1',host='remote',sampled_at='2026-11-18T00:00:00Z',data={'id':'r27_recipe_probe','summary':{'successes':3}})
        with patch.object(remote,'config',return_value=self.cfg),patch.object(remote.time,'monotonic',return_value=100),patch.object(remote,'fetch_remote',return_value=payload):
            first=remote.remote_snapshot('r27_recipe_probe')
        with patch.object(remote,'config',return_value=self.cfg),patch.object(remote.time,'monotonic',return_value=200),patch.object(remote,'fetch_remote',side_effect=subprocess.TimeoutExpired('ssh',9)):
            stale=remote.remote_snapshot('r27_recipe_probe')
        self.assertTrue(first['connection']['ok']);self.assertFalse(stale['connection']['ok'])
        self.assertTrue(stale['connection']['stale']);self.assertEqual(stale['summary']['successes'],3)
        self.assertEqual(stale['connection']['last_success_at'],first['connection']['last_success_at'])

    def test_ssh_has_no_shell_and_uses_dedicated_identity(self):
        payload=dict(schema='cube_remote_monitor_v1',data={})
        with patch.object(remote.subprocess,'run') as run:
            run.return_value.stdout=json.dumps(payload);remote.fetch_remote(self.cfg,'catalog')
        args=run.call_args.args[0]
        self.assertIn('IdentitiesOnly=yes',args);self.assertIn('StrictHostKeyChecking=yes',args)
        self.assertEqual(args[-1],'catalog');self.assertNotIn('shell',run.call_args.kwargs)
        with self.assertRaises(ValueError):remote.remote_snapshot('../contact_hold90_long_v2')

    def test_routes_cannot_control_remote_or_leak_connection_files(self):
        with patch.object(remote,'remote_snapshot',return_value={'runs':[]}) as snapshot:
            self.assertEqual(request('/api/remote/runs').send_json.call_args.args[0],200)
            snapshot.assert_called_once_with(None)
            for uri in ('/api/remote/runs','/api/remote/training','/api/remote/start'):
                self.assertEqual(request(uri,body={}).send_json.call_args.args[0],404)
            self.assertEqual(request('/api/remote/runs',origin='http://outside.invalid').send_json.call_args.args[0],403)
            self.assertEqual(request('/api/remote/training').send_json.call_args.args[0],400)
            self.assertEqual(request('/static/remote_reader.py').send_json.call_args.args[0],404)


if __name__=='__main__':unittest.main()
