"""Development PhysX evaluation: neural left residuals, fixed phase/right schedule.

No teacher left target or reference left state enters policy inference. This does
not evaluate right turns or promote any stage gate.
"""
from pathlib import Path
import argparse,json,time,shutil
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    import torch
    from phase_absolute_policy_v2 import accepted_prefix,phase_progress,features,restored,sha,SCHEMA
    rows,names,left,source=accepted_prefix(a.source);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);meta=saved['metadata']
    if meta['schema']!=SCHEMA or meta['joint_names']!=names or meta['output_indices']!=left:raise ValueError('Checkpoint schema mismatch')
    if source['seed']!=a.seed:raise ValueError('Phase/right schedule seed mismatch')
    if not any(s['manifest_sha256']==source['manifest_sha256'] for s in meta['sources']):raise ValueError('Checkpoint source mismatch')
    if any(meta['source_sha256'][n]!=sha(Path(__file__).parent/n) for n in ('phase_contact_policy.py','phase_absolute_policy_v2.py')):raise ValueError('Policy source changed after training')
    manifest=json.loads((a.source/'artifact_manifest.json').read_text())
    if sha(a.source/'center_request.json')!=manifest['sha256']['center_request.json']:raise ValueError('Center coordination request hash mismatch')
    protocol=json.loads((a.source/'center_request.json').read_text())['coordination_protocol']
    a.output.mkdir(parents=True,exist_ok=False);snapshot=a.output/'source_snapshot';snapshot.mkdir()
    sources=set(json.loads((a.source/'experiment.json').read_text()).get('source_sha256',{}));sources.update(('evaluate_phase_absolute_v2.py','phase_contact_policy.py','phase_absolute_policy_v2.py','center_j56_audit.py'))
    source_hashes={}
    for n in sorted(sources):shutil.copy2(Path(__file__).parent/n,snapshot/n);source_hashes[n]=sha(snapshot/n)
    model=restored(saved);model.eval();progress=phase_progress(rows);plan=json.loads((a.source/'center_plan.json').read_text())
    from isaaclab.app import AppLauncher
    from runtime_platform import platform_kit_args
    app=AppLauncher(headless=True,enable_cameras=False,kit_args=platform_kit_args('')).app
    env=None
    try:
        from articulated_cube_env import DynamicPickupEnv
        from isaac_compat import tensor,quaternion_from_native
        from finger_contact_audit import FingerContactAudit
        from palm_contact_audit import DetailedSelfContactAudit
        from palm_handover_audit import measured_hand_frames
        from elbow_audit import measured_elbow
        from centered_transport_audit import assess_center_row
        from center_j56_audit import joint_coordination
        from layer_turn_metrics import measure_layers
        env=DynamicPickupEnv(a.output,friction=(.9,.8),seed=a.seed,vision=False,preview=False,auto_reset=False,arm_damping=8.)
        env.max_episode_length=len(rows)+1;env.cfg['success']['hold_seconds']=3.;env.rng=np.random.default_rng(a.seed);env.reset();env.metrics={};env.pair_impulses={}
        if env.cfg['joint_names']!=names:raise ValueError('Runtime joint schema mismatch')
        (a.output/'config.json').write_text(json.dumps(env.cfg,indent=2));(a.output/'asset_contract.json').write_text(json.dumps(env.contract,indent=2))
        fingers=FingerContactAudit();audit=DetailedSelfContactAudit();reset=env.cube.pose_reset_count
        hold=longest=0;failure=None;lifted=False;maximum_lift=0.;returned=False;started=time.monotonic();contact_points=[];center_initial_q=None
        gain_step=next(i for i,r in enumerate(rows) if r['phase']=='tilt_hold');finger_ids=[env.ids[i] for i,n in enumerate(names) if n.startswith('left_')]
        with (a.output/'trajectory.jsonl').open('w') as journal:
            for step,reference in enumerate(rows):
                if step==gain_step:env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(finger_ids)),20.,dtype=torch.float32),joint_ids=finger_ids)
                root=tensor(env.cube.data.root_state_w)[0]
                previous={'q':tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),'joint_velocity':tensor(env.robot.data.joint_vel)[0,env.ids].tolist(),'applied_target':env.targets[0].tolist(),'cube_position':root[:3].tolist(),'cube_quaternion_wxyz':quaternion_from_native(root[3:7]).tolist(),'finger_contact_points':contact_points}
                if reference['phase']=='left_center_transport' and center_initial_q is None:center_initial_q=list(previous['q'])
                x=torch.from_numpy(features(previous,reference['phase'],progress[step])).unsqueeze(0)
                prior=torch.tensor(np.asarray(previous['applied_target'])[left],dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():prediction=model.targets(x,prior)[0]
                if not torch.isfinite(prediction).all():raise ValueError('Nonfinite policy output')
                target=np.asarray(reference['applied_target'],dtype=np.float32).copy();target[left]=prediction.numpy()
                fingers.reset_step();audit.reset_step();env.step((torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales)
                row=dict(step=step,phase=reference['phase'],**env.metrics);row.update(fingers.metrics());row.update(audit.metrics());contact_points=row['finger_contact_points']
                row.update(nominal_target=target.tolist(),applied_target=env.targets[0].tolist(),q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),cube_quaternion_wxyz=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).tolist(),measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env),policy_phase_progress=progress[step])
                row['layer_measurement']=measure_layers(env.cube.states())
                if row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit']):failure='robot_safety_gate'
                maximum_lift=max(maximum_lift,float(row['height_m']));lifted=lifted or row['height_m']>=.08
                if lifted and row['height_m']<.05:returned=True;failure='cube_returned_to_table_after_lift'
                if row['drop']:failure='cube_drop'
                if env.cube.pose_reset_count!=reset:raise RuntimeError('Unexpected cube pose write')
                if row['phase']=='center_hold':
                    if center_initial_q is None:raise ValueError('Missing measured center initial state')
                    report=assess_center_row(row,names,plan['center_target_world_m'],plan['cube_target_quaternion_wxyz']);coordination=joint_coordination(row['q'],center_initial_q,names,protocol);hold=hold+1 if report['qualified'] and coordination['qualified'] and not failure else 0;longest=max(longest,hold);row['center_audit']=report;row['joint_coordination']=coordination
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                if step%120==0 or failure:journal.flush();print('PHASE_ABSOLUTE_V2_EVAL',step,row['phase'],'hold',hold,'failure',failure,flush=True)
                if failure:break
        result={'scope':'development_learned_left_absolute_recorded_phase_and_right_schedule','seed':a.seed,'steps':step+1,'accepted':bool(not failure and longest>=90),'failure':failure or ('center_hold_gate_not_met' if longest<90 else None),'qualified_center_hold_s':longest/30.,'maximum_lift_m':maximum_lift,'cube_returned_to_table_after_lift':returned,'elapsed_s':time.monotonic()-started,'checkpoint_sha256':sha(a.checkpoint),'source_manifest_sha256':source['manifest_sha256'],'evaluation_source_sha256':source_hashes,'phase_schedule_source':a.source.name,'right_schedule_seed':source['seed'],'right_turn_trained':False,'stage_gate_unlocked':False,'cube_pose_writes_during_control':env.cube.pose_reset_count-reset,'model_input_state':'measured pre-action state; previous measured contacts and applied targets; scheduled phase/progress','left_teacher_targets_used':False,'left_finger_hold_gain_step':gain_step,'left_finger_hold_stiffness':20.}
        (a.output/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    except Exception as e:
        import traceback
        (a.output/'error.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
    finally:
        from isaac_compat import close_application
        if env is not None:env.close()
        close_application(app)

if __name__=='__main__':main()
