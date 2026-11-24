"""Exercise the dashboard's read-only contract with the contact-point runner formats."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cube_dashboard import remote_reader as reader


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ContactPointMonitor(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sweep = self.root / 'r27_recipe_contact_point_sweep70'
        self.plan = dict(schema=reader.SWEEP, offsets_deg=[7,-7], seeds=list(range(16)))
        save(self.sweep / 'plan.json', self.plan)
        mock_gpu = patch.object(reader, 'gpu', return_value={'available': False})
        mock_gpu.start()
        self.addCleanup(mock_gpu.stop)

    def publish(self, winner=None):
        angles = {str(a): dict(all_16={'episodes':16,'unsafe':2}, fixed_C_no_contact_horizon={'episodes':11},
                               qualifying_fixed_seeds=[], per_seed=[{'seed':0,'end_reason':'post_horizon'}])
                  for a in (0,7,-7)}
        save(self.sweep / reader.SWEEP_RESULT,
             dict(plan=self.plan, updated=False, independent=False, angles=angles,
                  fixed_C_no_contact_horizon_seeds=list(range(11)), selected_offset_deg=winner,
                  decision='no_candidate' if winner is None else 'selected'))

    def snapshot(self):
        return reader.snapshot(self.root,self.sweep.name,{})

    def test_no_winner_is_a_finished_screen_and_downstream_runs_are_unstarted(self):
        self.publish()
        entry, = reader.catalog(self.root,{})
        self.assertEqual((entry['kind'],entry['status'],entry['recorded_status']),('sweep','completed','no_candidate'))
        result = self.snapshot()['sweep']
        self.assertEqual(result['angles']['7']['all_16']['unsafe'],2)
        self.assertEqual(len(result['fixed_C_no_contact_horizon_seeds']),11)
        self.assertIsNone(result['selected_offset_deg'])
        self.assertEqual([(s['target_deg'],s['status'],s['run_id']) for s in result['stages']],
                         [(15,'blocked',None),(90,'blocked',None)])
        self.assertEqual(list(self.root.iterdir()),[self.sweep])

    def test_live_sweep_uses_real_attempt_and_receipt(self):
        attempt = 'attempt_1234abcd'
        save(self.sweep/'plus7'/'accepted.json',{'attempt':attempt})
        save(self.sweep/'plus7'/attempt/'report.json',{'rows':[]})
        second = self.sweep/'minus7'/attempt
        save(second/'progress.json',dict(step=140))
        save(second.parent/'request.json',dict(role='validate',iteration=70,target_deg=15,contact_point_offset_deg=-7))
        s = reader.snapshot(self.root,self.sweep.name,{self.sweep.name:[dict(pid=10,state='S',attempt=str(second))]})
        self.assertEqual((s['status'],s['completed'],s['planned']),('running',1,2))
        self.assertEqual((s['settings']['contact_point_offset_deg'],s['progress']['step']),(-7,140))
        self.assertIsNone(s['sweep']['decision'])
        self.assertIsNone(s['summary'])

    def test_ppo_links_and_target_reflect_the_actual_campaign(self):
        self.publish(7)
        self.assertEqual(self.snapshot()['sweep']['stages'][0]['status'],'not_started')
        home = self.root/'r27_recipe_contact_point_custom15'
        cfg = dict(schema=reader.POINT_PPO,sweep_dir=str(self.sweep),target_deg=15,contact_point_offset_deg=7,iterations=100)
        save(home/'config.json',cfg)
        save(home/'state.json',dict(iteration=88,status='qualified_15deg_state_specialist'))
        save(home/'qualified_policy.json',{'sha':'abc'})
        stages=self.snapshot()['sweep']['stages']
        self.assertEqual(stages[0]['run_id'],home.name)
        self.assertEqual(stages[1]['status'],'not_started')
        other=self.root/'r27_recipe_contact_point_other90'
        save(other/'config.json',dict(cfg,target_deg=90,sweep_dir=str(self.root/'other')))
        save(other/'state.json',dict(status='running',iteration=95))
        new=reader.snapshot(self.root,other.name,{})
        self.assertEqual(new['algorithm'],'state_ppo')
        self.assertIn('+90°',new['target'])
        self.assertEqual(new['settings']['contact_point_offset_deg'],7)
        self.assertIsNone(self.snapshot()['sweep']['stages'][1]['run_id'])

    def test_process_attribution_matches_runner_defaults_and_relative_paths(self):
        cwd=self.root.parent
        self.assertEqual(reader.process_run(['python','natural_face90_contact_point.py'],cwd,self.root),self.sweep)
        self.assertIsNone(reader.process_run(['python','natural_face90_contact_point.py','--check'],cwd,self.root))
        self.assertEqual(reader.process_run(['python','natural_face90_contact_point_train.py','--from-qualified15','foo'],cwd,self.root),
                         self.root/'r27_recipe_contact_point_ppo90')
        rel = Path(self.root.name)/'r27_recipe_custom'/'request.json'
        self.assertEqual(reader.process_run(['python','natural_face90_contact_point_worker.py','--request',str(rel)],cwd,self.root),
                         cwd/rel)

    def test_hover_cd_and_followup_are_separately_discovered_frozen_experiments(self):
        home=self.root/'r27_recipe_hover_d70'
        plan=dict(schema=reader.HOVER_CD,seeds=list(range(16)))
        save(home/'plan.json',plan)
        save(home/'treatment'/'report.json',dict(rows=[]))
        comparison={'group':{'C':{'unsafe':2},'D':{'unsafe':2}},'per_seed':[]}
        save(home/'hover_cd_summary.json',dict(plan=plan,updated=False,independent=False,comparison=comparison))
        result=reader.snapshot(self.root,home.name,{})
        self.assertEqual((result['kind'],result['status'],result['report_accepted']),('hover_cd','completed',True))
        self.assertEqual(result['hover_cd'],comparison)
        follow=self.root/'r27_recipe_approach_anchor_plus7'
        plan=dict(schema=reader.ANCHOR,seeds=list(range(16)))
        save(follow/'plan.json',plan)
        save(follow/'probe'/'attempt_abc123ab'/'request.json',dict(role='validate',iteration=70))
        self.assertFalse(reader.snapshot(self.root,follow.name,{})['approach_anchor'])
        save(follow/reader.ANCHOR_RESULT,dict(plan=plan,updated=False,independent=False,
             baseline={'fixed_11':{'episodes':11}},candidate={'fixed_11':{'episodes':11}},
             new_unsafe_seeds=[],qualifying_fixed_seeds=[]))
        result=reader.snapshot(self.root,follow.name,{})
        self.assertEqual((result['kind'],result['status']),('approach_anchor','completed'))
        self.assertEqual(result['approach_anchor']['candidate']['fixed_11']['episodes'],11)
        self.assertEqual(result['settings']['contact_point_offset_deg'],7)

    def test_pad_orientation_uses_three_paired_arms_and_reference_safety(self):
        home=self.root/'r27_recipe_pad_orientation_plus7'
        plan=dict(schema=reader.PAD_ORIENTATION,seeds=list(range(16)))
        save(home/'plan.json',plan)
        self.assertEqual(reader.process_run(['python','natural_face90_pad_orientation_probe.py'],
                                          self.root.parent,self.root),home)
        pending=reader.snapshot(self.root,home.name,{home.name:[dict(pid=77,state='R')]})
        self.assertEqual((pending['kind'],pending['status'],pending['pad_orientation']),('pad_orientation','running',{}))
        def arm(unsafe):
            return dict(fixed_11=dict(episodes=11),all_16=dict(episodes=16,unsafe=unsafe))
        save(home/reader.PAD_RESULT,dict(plan=plan,updated=False,independent=False,baseline=arm(0),
             previous=arm(3),candidate=arm(1),new_unsafe_seeds=[7],qualifying_fixed_seeds=[],
             paired_contact_gate_met=False,pad_orientation=dict(adjusted_by_lane=[True]*16)))
        result=reader.snapshot(self.root,home.name,{})
        self.assertEqual(result['status'],'completed')
        panel=result['pad_orientation']
        self.assertEqual([panel[k]['all_16']['unsafe'] for k in ('baseline','previous','candidate')],[0,3,1])
        self.assertEqual(panel['new_unsafe_seeds'],[7])
        self.assertFalse(result['qualified'])

        preshape=self.root/'r27_recipe_pad_preshape_plus7'
        save(preshape/'plan.json',dict(plan,schema=reader.PRESHAPE))
        self.assertEqual(reader.process_run(['python','natural_face90_pad_preshape_probe.py'],
                                          self.root.parent,self.root),preshape)
        s=reader.snapshot(self.root,preshape.name,{preshape.name:[dict(pid=78,state='R')]})
        self.assertEqual((s['kind'],s['status'],s['settings']['contact_point_offset_deg']),
                         ('pad_preshape','running',7))

    def test_cartesian_probe_and_training_use_their_own_destinations(self):
        script=['python','natural_face90_cartesian.py']
        cwd=self.root.parent
        self.assertEqual(reader.process_run(script+['probe'],cwd,self.root),self.root/'r27_recipe_cartesian_probe')
        self.assertEqual(reader.process_run(script+['train'],cwd,self.root),self.root/'r27_recipe_cartesian_ppo15')
        probe=Path(self.root.name)/'r27_recipe_custom_probe'
        train=Path(self.root.name)/'r27_recipe_custom_train'
        options=['--run-dir',str(train),'--probe-dir='+str(probe)]
        self.assertEqual(reader.process_run(script+options+['probe'],cwd,self.root),cwd/probe)
        self.assertEqual(reader.process_run(script+['train']+options,cwd,self.root),cwd/train)
        self.assertIsNone(reader.process_run(script+['check']+options,cwd,self.root))
        req=probe/'probe'/'request.json'
        self.assertEqual(reader.process_run(['python','natural_face90_cartesian_worker.py','--request',str(req)],
                                           cwd,self.root),cwd/req)

    def test_cartesian_probe_reports_measured_gate_and_distinct_safety_cohort(self):
        home=self.root/'r27_recipe_cartesian_probe'
        plan=dict(schema=reader.CARTESIAN_PROBE,seeds=list(range(16)))
        save(home/'plan.json',plan)
        slot=home/'probe'
        attempt=slot/'attempt_1234abcd'
        save(slot/'request.json',dict(role='validate',iteration=0))
        save(attempt/'progress.json',dict(step=123))
        pending=reader.snapshot(self.root,home.name,{home.name:[dict(pid=10,state='S',attempt=str(attempt))]})
        self.assertEqual((pending['kind'],pending['status'],pending['role'],pending['iteration'],pending['lanes']),
                         ('cartesian_probe','running','validate',0,16))
        self.assertEqual((pending['completed'],pending['planned'],pending['cartesian_probe']),(0,1,{}))
        def arm(first,driven,unsafe):
            return dict(fixed_11=dict(episodes=11,safe_qualified_index_first=first,safe_drive_2deg=driven),
                        all_16=dict(episodes=16,unsafe=unsafe))
        contract=dict(actor_dim=19,critic_dim=19,action_dim=3,residual_norm_cap_m=.0005)
        result=dict(schema=reader.CARTESIAN_PROBE,plan=plan,updated=False,independent=False,
                    baseline=arm(0,0,2),previous=arm(1,0,3),candidate=arm(4,2,3),cartesian=contract,
                    qualifying_fixed_seeds=[1,3],new_unsafe_seeds=[9],paired_contact_gate_met=False)
        save(home/reader.CARTESIAN_RESULT,result)
        s=reader.snapshot(self.root,home.name,{})
        self.assertEqual((s['completed'],s['planned'],s['status'],s['cartesian']),(1,1,'completed',contract))
        panel=s['cartesian_probe']
        self.assertEqual(panel['candidate']['fixed_11']['safe_drive_2deg'],2)
        self.assertEqual(panel['candidate']['all_16']['unsafe'],3)
        self.assertEqual((panel['qualifying_fixed_seeds'],panel['new_unsafe_seeds']),([1,3],[9]))
        self.assertFalse(panel['paired_contact_gate_met'])
        save(home/reader.CARTESIAN_RESULT,dict(result,plan=dict(plan,rng=17)))
        self.assertFalse(reader.snapshot(self.root,home.name,{})['cartesian_probe'])

    def test_cartesian_ppo_reads_committed_counters_and_validation_waves(self):
        home=self.root/'r27_recipe_cartesian_ppo15'
        save(home/'config.json',dict(schema=reader.CARTESIAN_PPO,lanes=16,iterations=100,
                                    target_deg=15,contact_point_offset_deg=7,pad_preshape_offset_deg=-6))
        save(home/'state.json',dict(iteration=4,status='running'))
        previous=dict(episodes=16,successes=12,index_driven_successes=11,unsafe=1)
        save(home/'iteration_0001'/'result.json',dict(iteration=1,validation_after_update=previous))
        slot=home/'iteration_0005'/'validate'/'wave_0000'
        attempt=slot/'attempt_1234abcd'
        save(slot/'request.json',dict(role='validate',iteration=5))
        save(slot/'accepted.json',dict(attempt=attempt.name))
        contract=dict(actor_dim=19,critic_dim=19,action_dim=3)
        save(attempt/'report.json',dict(rows=[dict(lane=0,success=False,unsafe=True,final_face_deg=.25)],
                                       expert_learning=dict(cartesian=contract,actor_updated=False)))
        live={home.name:[dict(pid=2,state='R',attempt=str(attempt))]}
        s=reader.snapshot(self.root,home.name,live)
        self.assertEqual((s['kind'],s['algorithm'],s['iteration'],s['completed'],s['planned']),
                         ('cartesian_training','state_ppo',5,4,100))
        self.assertEqual((s['role'],s['settings']['pad_preshape_offset_deg'],s['target_deg']),('validate',-6,15))
        self.assertTrue(s['report_accepted'])
        self.assertEqual(s['cartesian'],contract)
        self.assertEqual(s['summary']['unsafe'],1)
        self.assertEqual(s['history'][0]['validation']['index_driven_successes'],11)
        self.assertEqual({r['kind'] for r in reader.catalog(self.root,live)},{'sweep','cartesian_training'})

    def test_contact_curriculum_cli_tracks_outputs_and_paired_evaluation(self):
        cli=['python','natural_face90_cartesian_contact.py']
        cwd=self.root.parent
        for mode,default in [('train','r27_recipe_cartesian_contact_ppo'),
                             ('train15','r27_recipe_cartesian_contact_to15')]:
            self.assertEqual(reader.process_run(cli+[mode],cwd,self.root),self.root/default)
            rel=Path(self.root.name)/'r27_recipe_custom'
            self.assertEqual(reader.process_run(cli+[mode,'--probe-dir','prior','--contact-dir=short',
                                                    '--run-dir',str(rel)],cwd,self.root),cwd/rel)
        self.assertIsNone(reader.process_run(cli+['check','--probe-dir','prior'],cwd,self.root))
        self.assertIsNone(reader.process_run(cli+['long'],cwd,self.root))
        self.assertEqual(reader.process_run(cli+['train','--contact-dir',str(rel)],cwd,self.root),cwd/rel)
        home=self.root/'r27_recipe_cartesian_contact_ppo'
        slot=home/'iteration_0005/validate/wave_0000'
        req=slot/'request.json'
        worker=['python','natural_face90_cartesian_contact_worker.py','--request',str(req)]
        self.assertEqual(reader.process_run(worker,cwd,self.root),req)
        save(home/'config.json',dict(schema=reader.CARTESIAN_CONTACT,iterations=15,lanes=16))
        save(req,dict(role='validate',iteration=5))
        attempt=slot/'attempt_abc12345'
        save(attempt/'progress.json',dict(step=17))
        result=reader.snapshot(self.root,home.name,{home.name:[dict(pid=9,state='R',attempt=str(attempt))]})
        self.assertEqual((result['kind'],result['algorithm'],result['status']),('cartesian_contact','state_ppo','running'))
        self.assertEqual((result['role'],result['iteration'],result['progress']['step']),('validate',5,17))
        self.assertFalse(result['contact_ready'])

    def test_contact_readiness_uses_committed_pair_and_matching_transfer_receipt(self):
        home=self.root/'r27_recipe_cartesian_contact_ppo'
        save(home/'config.json',dict(schema=reader.CARTESIAN_CONTACT,target_deg=15,iterations=15,lanes=16))
        gate=dict(fixed_11=dict(episodes=11,safe_qualified_index_first=3,safe_drive_2deg=2),
                  all_16=dict(episodes=16,unsafe=0),qualifying_fixed_seeds=[101,106],new_unsafe_seeds=[],
                  paired_contact_gate_met=True,fixed_peak_drive_deg=2.3)
        state=dict(iteration=2,checkpoint='two.pt',checkpoint_sha='two_sha',status='running')
        save(home/'iteration_0002/result.json',dict(iteration=2,paired_gate=gate))
        save(home/'iteration_0003/result.json',dict(iteration=3,paired_gate=dict(gate,qualifying_fixed_seeds=[99])))
        save(home/'state.json',state)
        get=lambda:reader.snapshot(self.root,home.name,{})
        self.assertEqual(get()['contact_gate'],dict(gate,iteration=2))
        self.assertFalse(get()['contact_ready'])
        save(home/'state.json',dict(state,status='contact_ready'))
        cert=dict(iteration=2,checkpoint='two.pt',sha='two_sha',paired_gate=gate)
        for changed in (dict(sha='old'),dict(iteration=1),dict(paired_gate=dict(gate,all_16={'unsafe':3})),{}):
            save(home/'contact_ready.json',dict(cert,**changed))
            self.assertEqual(get()['contact_ready'],not changed)
        self.assertFalse(get()['qualified'])
        self.assertEqual(get()['history'][0]['paired_gate']['all_16']['unsafe'],0)
        followup=self.root/'r27_recipe_cartesian_contact_to15'
        save(followup/'config.json',dict(schema=reader.CARTESIAN_TO15,target_deg=15))
        save(followup/'state.json',dict(status='running',iteration=1))
        save(followup/'contact_ready.json',cert)
        snap=reader.snapshot(self.root,followup.name,{})
        self.assertEqual((snap['kind'],snap['algorithm'],snap['qualified']),('cartesian_contact_to15','state_ppo',False))
        save(followup/'state.json',dict(status='qualified_15deg_state_specialist',iteration=3))
        save(followup/'qualified_policy.json',dict(sha='new_policy'))
        self.assertTrue(reader.snapshot(self.root,followup.name,{})['qualified'])
