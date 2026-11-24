"""CPU algorithm/bridge contracts only; these tests do not demonstrate a skill."""
import io
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import natural_face90_recipe_policy as p
from natural_face90_recipe_worker import PostPhase, ScriptedPostPhase
from natural_face90_recipe_long import GUIDED_SCHEMA, validate_request, request, worker_failure

torch.set_num_threads(1)


class Algorithm(unittest.TestCase):
    def test_masked_returns_do_not_cross_prefix_or_dead_lanes(self):
        r=torch.tensor([[999.,1.],[1.,2.],[2.,999.],[3.,999.]])
        v=torch.zeros_like(r)
        valid=torch.tensor([[False,True],[True,True],[True,False],[True,False]])
        done=torch.tensor([[False,False],[False,True],[False,False],[True,False]])
        adv,returns=p.gae(r,v,valid,done,gamma=1.,lam=1.)
        torch.testing.assert_close(adv,torch.tensor([[0.,3.],[6.,2.],[5.,0.],[3.,0.]]))
        torch.testing.assert_close(adv,returns)

    def test_incomplete_episode_is_not_silently_terminal(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            p.gae(torch.ones(3,2),torch.zeros(3,2),torch.ones(3,2,dtype=torch.bool),torch.zeros(3,2,dtype=torch.bool))

    def test_terminal_potential_has_no_surviving_distance_bonus(self):
        potential=torch.tensor([.3,.8,1.2,.6])
        rewards=[]
        for t in range(3):
            rewards.append(p.step_reward(potential[t:t+1],potential[t+1:t+2],torch.tensor([t==2]),
                torch.tensor([False]),torch.tensor([False]),torch.zeros(1,28))[0]+.001)
        total=sum((p.GAMMA**i)*r for i,r in enumerate(rewards))
        self.assertAlmostEqual(float(total),-.3,places=6)

    def test_unsafe_transition_never_earns_positive_credit(self):
        result=p.step_reward(torch.tensor([1.,1.]),torch.tensor([1000.,1000.]),
            torch.tensor([True,True]),torch.tensor([True,False]),torch.tensor([True,True]),torch.zeros(2,28))
        self.assertLess(float(result[0]),-20.)
        self.assertGreater(float(result[1]),0.)

    def test_privileged_critic_does_not_change_actor(self):
        model=p.StateExpert();obs=torch.randn(2,p.ACTOR_DIM);critic=torch.randn(2,p.CRITIC_DIM)
        a=model.sample(obs,critic,True);b=model.sample(obs,critic+2,True)
        torch.testing.assert_close(a[0],b[0]);self.assertFalse(torch.equal(a[2],b[2]))

    def test_complete_real_shape_batch_updates_actor_even_under_rollout_no_grad(self):
        torch.manual_seed(13);model=p.StateExpert();alg=p.PPO(model)
        # Synthetic inputs test PPO numerics only, never accepted as PhysX evidence.
        obs=torch.randn(16,4,p.ACTOR_DIM);critic=torch.randn(16,4,p.CRITIC_DIM)
        with torch.no_grad():
            latent,logp,value=model.sample(obs,critic)
            done=torch.zeros(16,4,dtype=torch.bool);done[-1]=True
            data=dict(obs=obs,critic=critic,latent=latent,logp=logp,value=value,
                reward=latent[:,:,0].tanh(),valid=torch.ones_like(done),done=done)
            result=alg.update(data)
        self.assertGreater(result['actor_max_delta'],0.)
        self.assertEqual(result['real_transitions'],64)
        self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))

    def test_observation_dimensions_and_nonfinite_fail_closed(self):
        n=2;z=lambda *dims:torch.zeros(n,*dims)
        args=[z(56),z(56),z(56),z(7),z(5,3),z(3),z(),z(),z().bool(),z(),z(26,7),z(5),z(6)]
        obs,critic=p.pack_observation(*args)
        self.assertEqual(obs.shape,(2,p.ACTOR_DIM));self.assertEqual(critic.shape,(2,p.CRITIC_DIM))
        args[0][0,0]=float('nan')
        with self.assertRaises(ValueError):p.pack_observation(*args)


class Bridge(unittest.TestCase):
    def make(self):
        x=PostPhase.__new__(PostPhase);n=2
        x.e=SimpleNamespace(robot=SimpleNamespace(get_dof_positions=lambda:torch.zeros(n,56)),
                            ids=list(range(56)),step=torch.full((1,56),.01),vel=torch.ones(1,56))
        x.device=torch.device('cpu');x.training=True;x.max_steps=2200;x.n=n
        x.learning=torch.zeros(n,dtype=torch.bool);x.learn_steps=torch.zeros(n,dtype=torch.long)
        x.handoff_target=torch.zeros(n,56)
        x.pad_geometry=(None,None,None)
        def arm_command(target,q,handoff,fresh,learned,latent,*geometry):
            cmd=handoff.clone()
            cmd[:,28:35]+=.008*latent.tanh()
            return cmd
        x.guide=SimpleNamespace(command=arm_command)
        x.controller=SimpleNamespace(detour_done=torch.ones(n,dtype=torch.bool),phase=torch.tensor([2,2]),
            rp=SimpleNamespace(ix=list(range(28,35))),
            clearance=SimpleNamespace(project=lambda a,b:b))
        x.obs=dict(natural_ready=torch.tensor([True,True]),contact_goal_reached=torch.tensor([False,False]),
                   turn_started=torch.tensor([False,False]))
        x.action_phase=torch.zeros(n,dtype=torch.long);x.retract=torch.zeros(n,dtype=torch.bool)
        x.best_distance=torch.full((n,),float('inf'));x.cumulative_reward=torch.zeros(n)
        x.req=dict(seeds=[1,2]);x.reward_log=io.StringIO();x.data=[]
        x.model=SimpleNamespace(sample=lambda o,c,deterministic:(torch.full((n,7),5.),torch.ones(n),torch.zeros(n)))
        x.state=lambda target,remaining:(torch.zeros(n,198),torch.zeros(n,391),torch.ones(n),torch.full((n,),.01))
        return x

    def test_handoff_keeps_left_and_inactive_targets_and_caps_right(self):
        x=self.make();target=torch.full((2,56),.2)
        with patch.object(ScriptedPostPhase,'propose',side_effect=lambda t,a,s:t.clone()):
            result=x.propose(target,torch.tensor([True,False]),torch.tensor([400,200]))
        self.assertTrue(x.learning[0]);self.assertFalse(x.learning[1])
        torch.testing.assert_close(result[:, :28],target[:,:28])
        torch.testing.assert_close(result[1],target[1])
        self.assertLessEqual(float((result-target).abs().max()),.010001)
        self.assertEqual(x.pending['valid'].tolist(),[True,False])
        # Keep the sampled latent even though the actual actuator command was projected.
        self.assertEqual(float(x.pending['latent'][0,0]),5.)
        x.after_step(torch.tensor([True,False]),torch.tensor([False,False]),
                     torch.tensor([True,False]),['face_structure_violation',''],torch.tensor([401,200]))
        self.assertTrue(x.data[0]['done'][0]);self.assertLess(float(x.data[0]['reward'][0]),-20.)
        self.assertEqual(float(x.data[0]['reward'][1]),0.)

    def test_natural_ready_without_safe_detour_does_not_handoff(self):
        x=self.make();x.controller.detour_done[:]=False
        with patch.object(ScriptedPostPhase,'propose',side_effect=lambda t,a,s:t.clone()):
            x.propose(torch.zeros(2,56),torch.ones(2,dtype=torch.bool),torch.zeros(2))
        self.assertFalse(x.learning.any());self.assertIsNone(x.pending)

    def test_validation_records_no_optimizer_data(self):
        x=self.make();x.training=False
        with patch.object(ScriptedPostPhase,'propose',side_effect=lambda t,a,s:t.clone()):
            x.propose(torch.zeros(2,56),torch.ones(2,dtype=torch.bool),torch.zeros(2))
        x.after_step(torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),['post_horizon']*2,torch.ones(2))
        self.assertEqual(x.data,[])

    def test_learned_trace_does_not_report_stale_script_goal(self):
        x=self.make();x.learning[:]=True;x.action_phase[:]=6;x.phase_names=('x',)*6+('ppo_contact',)
        x.traces=io.StringIO();x.e.names=[str(i) for i in range(56)]
        x.e.robot.get_dof_velocities=lambda:torch.zeros(2,56)
        x.e.core=lambda:torch.zeros(2,7)
        x.body_names=['world'];x.body_poses=lambda:torch.zeros(2,1,7)
        x.record_state(torch.tensor([True,False]),450,torch.zeros(2,56),'terminal','post_horizon')
        row=json.loads(x.traces.getvalue())
        self.assertTrue(row['learned_policy_active'])
        self.assertIn('learned_contact_marker_error_m',row)
        self.assertNotIn('right_goal_error_m',row)

    def test_requests_cannot_train_under_validation_role(self):
        cfg=dict(schema=GUIDED_SCHEMA,rng=327,source_sha256={})
        req=request(cfg,'train',[45_000_000,45_000_001],1,None)
        validate_request(req)
        req['role']='validate'
        with self.assertRaises(ValueError):validate_request(req)
        req['learn']=False
        with self.assertRaises(ValueError):validate_request(req)


class ContactFrame(unittest.TestCase):
    def scene(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=Usd.Stage.CreateInMemory();root='/World/Robot'
        body=UsdGeom.Xform.Define(stage,root+'/right_index_DIP_link')
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        body.AddTranslateOp().Set((1.,2.,3.));body.AddRotateXYZOp().Set((10.,20.,30.))
        path=str(body.GetPath())+'/right_index_DIP_rubber_link'
        frame=UsdGeom.Xform.Define(stage,path)
        frame.AddTranslateOp().Set((.01,.02,.03));frame.AddRotateXYZOp().Set((0.,0.,35.))
        for branch in ('visuals','collisions'):
            decoy=UsdGeom.Xform.Define(stage,str(body.GetPath())+'/'+branch+'/right_index_DIP_rubber_link')
            decoy.AddTranslateOp().Set((9.,9.,9.))
        return stage,root,path

    def test_real_asset_reproduces_old_error_and_resolves_the_actual_frame(self):
        import ast
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        from natural_face90_recipe_frames import contact_frame, check_source_scene
        root=Path(__file__).resolve().parent
        # Execute the unchanged lookup function alone: assets.py also imports
        # the Kit-only PhysxSchema module, which this read-only USD test does not
        # need. Do not launch Kit or stub out the geometry to reproduce the bug.
        source_ast=ast.parse((root/'assets.py').read_text())
        fn=next(n for n in source_ast.body if isinstance(n,ast.FunctionDef) and n.name=='fixed_frame')
        scope=dict(Usd=Usd,UsdGeom=UsdGeom,UsdPhysics=UsdPhysics,Gf=Gf)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(root/'assets.py'),'exec'),scope)
        fixed_frame=scope['fixed_frame']
        source=root/'runs/contact_hold_size_r15_90mm_retarget_approach_20261112_084150_canonical_run/from_reset/seed_21611'
        scene=source/'dynamic_training_scene.usda'
        stage=Usd.Stage.Open(str(scene))
        with self.assertRaisesRegex(AssertionError,'right_index_DIP_rubber_link.*3'):
            fixed_frame(stage,'/World/Robot','right_index_DIP_rubber_link')
        before=stage.GetRootLayer().ExportToString()
        result=check_source_scene(source)
        self.assertEqual(result['body'],'right_index_DIP_link')
        torch.testing.assert_close(torch.tensor(result['offset']),torch.zeros(3),atol=1e-10,rtol=0.)
        self.assertEqual(stage.GetRootLayer().ExportToString(),before)
        # Exercise the exact runtime namespace without constructing a simulator.
        clone=Usd.Stage.CreateInMemory()
        proto=UsdGeom.Xform.Define(clone,'/World/envs/env_0000')
        proto.GetPrim().GetReferences().AddReference(str(scene),'/World')
        proto.AddTranslateOp().Set((4.,-4.,0.))
        body,offset,quat=contact_frame(clone,'/World/envs/env_0000/Robot')
        self.assertEqual(body,result['body'])
        torch.testing.assert_close(torch.tensor(offset),torch.tensor(result['offset']),atol=1e-10,rtol=0.)
        torch.testing.assert_close(torch.tensor(quat),torch.tensor(result['quaternion_wxyz']),atol=1e-10,rtol=0.)

    def test_geometry_decoys_cannot_override_nonzero_link_transform(self):
        from natural_face90_recipe_frames import contact_frame
        from pxr import Gf
        stage,root,path=self.scene()
        before=stage.GetRootLayer().ExportToString()
        body,offset,quat=contact_frame(stage,root)
        self.assertEqual(body,'right_index_DIP_link')
        torch.testing.assert_close(torch.tensor(offset),torch.tensor([.01,.02,.03]),atol=1e-7,rtol=0.)
        q=Gf.Rotation(Gf.Vec3d(0,0,1),35).GetQuat()
        torch.testing.assert_close(torch.tensor(quat),torch.tensor([q.GetReal(),*q.GetImaginary()]))
        self.assertEqual(stage.GetRootLayer().ExportToString(),before)

    def test_missing_canonical_frame_does_not_fall_back_to_geometry(self):
        from natural_face90_recipe_frames import contact_frame
        stage,root,path=self.scene();stage.RemovePrim(path)
        with self.assertRaisesRegex(ValueError,'Missing/invalid calibrated link frame'):
            contact_frame(stage,root)

    def test_separate_rigid_body_or_scaled_frame_is_rejected(self):
        from pxr import UsdPhysics,UsdGeom
        from natural_face90_recipe_frames import contact_frame
        stage,root,path=self.scene();UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
        with self.assertRaises(ValueError):contact_frame(stage,root)
        stage,root,path=self.scene();UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddScaleOp().Set((.001,.001,.001))
        with self.assertRaisesRegex(ValueError,'unit scale'):contact_frame(stage,root)

    def test_worker_error_reaches_launcher_message(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)
            (dest/'failed.json').write_text(json.dumps(dict(error="AssertionError(('right_index_DIP_rubber_link', 3))")))
            error=worker_failure(1,dest,'/tmp/smoke.log')
            self.assertIn('right_index_DIP_rubber_link',str(error))
            self.assertIn('/tmp/smoke.log',str(error))
            (dest/'failed.json').write_text('truncated{')
            self.assertIn('exited 1',str(worker_failure(1,dest,'/tmp/smoke.log')))


if __name__=='__main__':unittest.main()
