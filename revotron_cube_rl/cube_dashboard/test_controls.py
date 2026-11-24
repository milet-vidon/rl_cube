"""Temporary run trees and CPU helper processes only: no PhysX or GPU trainers are launched."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import contact_hold90_controls as engine
from cube_dashboard import controls as con, rollouts as ro


class ControlTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)/"project";self.runs=self.root/"runs";self.runs.mkdir(parents=True)
        ro._global(self.root).mkdir(parents=True)
        for n in ro._NAMES:(ro._global(self.root)/n).touch()
        self.addCleanup(patch.stopall)
        old=ro._locks
        patch.object(ro,"_locks",side_effect=lambda:(old()[0],True)).start()
        patch.object(con.training,"_lock_status",return_value=(False,None)).start()
        self.gpu=patch.object(ro,"_check_gpu").start()
        self.parents={"r15":({"id":"r15","label":"基线","sha256":"f"*64},self.runs/"r15"/"best.pt"),
                      "old-r1":({"id":"old-r1","label":"第 1 轮","sha256":"e"*64},self.runs/"old"/"round_001"/"fit"/"best.pt")}
        patch.object(con,"_models",return_value=self.parents).start()
        self.verify=patch.object(con.inference,"_verified").start()
        self.p=dict(engine.DEFAULTS)
        self.name="contact_hold90_ui_test"
        self.body={"run":self.name,"values":self.p}

    def run_home(self,name=None,*,legacy=False):
        d=self.runs/(name or self.name);d.mkdir()
        cfg=dict(self.p)
        if not legacy:cfg["engine"]=engine.ENGINE
        (d/"settings.json").write_text(json.dumps(cfg))
        (d/"state.json").write_text(json.dumps(dict(status="interrupted",completed_round=0)))
        if not legacy:engine.initialize(d,self.p)
        (d/"campaign.lock").touch()
        return d

    def test_read_only_schema_empty_defaults_and_draft_survives_reload(self):
        s=con.control_status(self.root)
        self.assertTrue(s["available"]);self.assertIsNone(s["draft"]);self.assertEqual({f["key"] for f in s["schema"]},set(self.p))
        self.assertFalse(con._store(self.root).exists())
        vals=self.p|{"rounds":150,"batch":65536,"hold_score":2}
        res=con.control_action(self.root,"draft",dict(run=None,name=self.name,values=vals,parent="old-r1"))
        self.assertEqual(res["draft"]["values"],vals)
        s=con.control_status(self.root);self.assertEqual(s["draft"]["values"],vals);self.assertEqual(s["draft"]["parent"],"old-r1")
        self.assertFalse((self.runs/self.name).exists())
        self.assertFalse(self.verify.called);self.assertTrue(self.gpu.called)

    def test_legacy_dataset_cannot_be_changed_while_gpu_busy_draft_is_independent(self):
        d=self.run_home(legacy=True);before={p.name:p.read_bytes() for p in d.iterdir()}
        with (d/"campaign.lock").open("rb") as f,patch.object(con,"_spawn") as start:
            fcntl.flock(f,fcntl.LOCK_EX)
            state=con.control_status(self.root,self.name)
            self.assertFalse(state["run"]["online"]);self.assertFalse(state["available"])
            with self.assertRaises(con.ControlConflict):con.control_action(self.root,"apply",self.body)
            with self.assertRaises(con.ControlConflict):con.control_action(self.root,"start",dict(run=self.name+"_next",values=self.p))
            with self.assertRaises(con.ControlConflict):con.control_action(self.root,"resume",self.body)
            con.control_action(self.root,"draft",self.body)
            self.assertFalse(start.called);self.assertFalse(self.gpu.called)
        self.assertEqual(before,{p.name:p.read_bytes() for p in d.iterdir()})

    def test_applied_config_persists_across_api_restart_but_does_not_change_started_round(self):
        d=self.run_home();engine.freeze_round(d,1);s=con.control_status(self.root,self.name)
        changes=self.p|{"ending_score":3.0,"rounds":120,"batch":65536}
        r=con.control_action(self.root,"apply",{"run":self.name,"values":changes,"expected_revision":s["effective_revision"]})
        self.assertEqual(r["earliest_round"],2)
        s2=con.control_status(self.root,self.name);self.assertEqual(s2["effective"],self.p);self.assertEqual(s2["pending"]["values"],changes)
        with self.assertRaises(con.ControlConflict):
            con.control_action(self.root,"apply",{"run":self.name,"values":self.p,"expected_revision":s["effective_revision"]})
        self.assertEqual(engine.freeze_round(d,1)["params"],self.p)
        self.assertEqual(engine.freeze_round(d,2)["params"],changes)
        self.assertIsNone(con.control_status(self.root,self.name)["pending"])

    def test_invalid_input_and_links_never_write(self):
        for body in [dict(run="../x",values=self.p),dict(run=None,values=self.p|{"lr":float("nan")}),
                     dict(run=None,values=self.p|{"rounds":True}),dict(run=None,values=self.p|{"envs":5}),
                     dict(run=None,values={"batch":23}),dict(run=None,values=self.p,name="../x"),
                     dict(run=None,values=self.p,parent="../weights"),self.body|{"path":"/tmp"}]:
            with self.assertRaises(ValueError):con.control_action(self.root,"draft",body)
        for name in ("contact_hold90_long_v99","contact_hold90_ui_"):
            with self.assertRaises(ValueError):con.control_action(self.root,"start",self.body|{"run":name})
            with self.assertRaises(ValueError):con.control_action(self.root,"draft",dict(run=None,values=self.p,name=name))
        self.assertFalse(con._store(self.root).exists())
        con._store(self.root).symlink_to(self.root)
        with self.assertRaises(con.ControlUnavailable):con.control_action(self.root,"draft",self.body|{"run":None})

    def test_external_gpu_user_without_project_locks_disables_start_but_not_drafts(self):
        self.gpu.side_effect=ro.RolloutBusy("外部 CUDA 进程正在运行")
        s=con.control_status(self.root);self.assertFalse(s["available"]);self.assertIn("CUDA",s["reason"])
        con.control_action(self.root,"draft",dict(run=None,values=self.p))
        with self.assertRaises(con.ControlConflict):con.control_action(self.root,"start",self.body)

    def test_existing_name_and_unverified_parent_do_not_launch(self):
        self.run_home(legacy=True)
        with patch.object(con,"_spawn") as start:
            with self.assertRaises(con.ControlConflict):con.control_action(self.root,"start",self.body)
            self.verify.side_effect=con.inference.InferenceUnavailable("wrong hash")
            with self.assertRaises(con.ControlUnavailable):con.control_action(self.root,"start",self.body|{"run":self.name+"_new"})
            self.assertFalse(start.called)
        self.assertFalse((self.runs/(self.name+"_new")).exists())

    def test_launch_reserves_interlock_across_child_lifetime_and_recovers_on_reload(self):
        argv=[];release=self.root/"finish"
        def launch(root,cmd,fd,log):
            argv.extend(cmd)
            script="from pathlib import Path\nimport time,sys\nfor i in range(500):\n if Path(sys.argv[1]).exists():break\n time.sleep(.01)\n"
            return subprocess.Popen([sys.executable,"-c",script,str(release)],pass_fds=(fd,),stdout=log,stderr=log)
        with patch.object(con,"_spawn",side_effect=launch):
            res=con.control_action(self.root,"start",self.body|{"parent":"old-r1"})
            self.assertEqual(self.verify.call_args.args[1],self.parents["old-r1"][0]["sha256"])
            self.assertEqual(argv[argv.index("--parent-sha")+1],"e"*64)
            self.assertEqual(argv[0],"/bin/bash");self.assertEqual(argv[argv.index("--parent")+1],str(self.parents["old-r1"][1].parent))
            self.assertEqual(json.loads(Path(argv[argv.index("--control-config")+1]).read_text()),self.p)
            try:
                self.assertFalse(con.control_status(self.root)["available"])
                self.assertEqual(con.control_status(self.root)["jobs"][0]["status"],"starting")
                with self.assertRaises(con.ControlConflict):con.control_action(self.root,"start",self.body|{"run":self.name+"_second"})
                with self.assertRaises(ro.RolloutBusy):ro._availability(self.root,ro._locks())
            finally:release.touch()
            d=con._store(self.root)/"jobs"/res["id"]
            for _ in range(400):
                if (d/"exit.json").exists():break
                time.sleep(.01)
            self.assertTrue(con.control_status(self.root)["available"])
            self.assertEqual(con.control_status(self.root)["jobs"][0]["status"],"interrupted")

    def test_resuming_passes_original_directory_and_stages_changes_before_start(self):
        home=self.run_home();engine.freeze_round(home,1)
        def fast(root,cmd,fd,log):
            self.assertEqual(cmd[-1],"--resume");self.assertEqual(cmd[cmd.index("--run-dir")+1],str(home))
            self.assertEqual(engine.control_snapshot(home)["pending"]["params"]["rounds"],175)
            return subprocess.Popen([sys.executable,"-c","pass"],pass_fds=(fd,))
        with patch.object(con,"_spawn",side_effect=fast):
            res=con.control_action(self.root,"resume",self.body|{"values":self.p|{"rounds":175}})
            d=con._store(self.root)/"jobs"/res["id"]
            for _ in range(400):
                if (d/"exit.json").exists():break
                time.sleep(.01)
        self.assertFalse(self.verify.called)


if __name__=="__main__":unittest.main()
