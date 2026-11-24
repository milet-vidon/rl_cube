"""Development closed-loop check: learned LEFT targets, recorded RIGHT schedule.

Not a pure bimanual neural policy, not face-turn validation, not a promotion gate.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=9901);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app=AppLauncher(headless=True,enable_cameras=False,kit_args=platform_kit_args('')).app
env=None
try:
    import torch
    from teacher_bootstrap import dataset,network,sha
    from articulated_cube_env import DynamicPickupEnv
    from isaac_compat import tensor,quaternion_from_native
    from finger_contact_audit import FingerContactAudit
    from palm_contact_audit import DetailedSelfContactAudit
    from palm_handover_audit import measured_hand_frames
    from elbow_audit import measured_elbow
    from centered_transport_audit import assess_center_row
    from layer_turn_metrics import measure_layers
    dataset(a.source) # recheck source acceptance and manifest integrity
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['metadata']['source_manifest_sha256']!=sha(a.source/'artifact_manifest.json'):
        raise ValueError('Checkpoint/source mismatch')
    model=network(checkpoint['metadata'].get('bounded',False),checkpoint['metadata'].get('box_bounds'));model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=json.loads((a.source/'trajectory.json').read_text())
    plan=json.loads((a.source/'center_plan.json').read_text())
    env=DynamicPickupEnv(a.output,friction=(.9,.8),seed=a.seed,vision=False,preview=False,auto_reset=False,arm_damping=8.)
    env.max_episode_length=len(rows)+1;env.cfg['success']['hold_seconds']=3.
    env.rng=np.random.default_rng(a.seed);env.reset();env.metrics={};env.pair_impulses={}
    names=env.cfg['joint_names'];assert names==checkpoint['metadata']['joint_names']
    fingers=FingerContactAudit();audit=DetailedSelfContactAudit();resets=env.cube.pose_reset_count
    longest=0;hold=0;failure=None;started=time.monotonic();left=checkpoint['metadata']['output_indices']
    with (a.output/'trajectory.jsonl').open('w') as journal:
        for step,reference in enumerate(rows):
            q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist()
            root=tensor(env.cube.data.root_state_w)[0]
            xyz=root[:3].tolist();quat=quaternion_from_native(root[3:7]).tolist()
            state=torch.tensor(q+xyz+quat+[step/3000.],dtype=torch.float32)
            with torch.no_grad():prediction=model((state-checkpoint['mean'])/checkpoint['std'])*checkpoint['scale']+checkpoint['center']
            if not torch.isfinite(prediction).all():raise ValueError('Nonfinite policy output')
            target=np.asarray(reference['nominal_target'],dtype=np.float32).copy();target[left]=prediction.numpy()
            fingers.reset_step();audit.reset_step()
            env.step((torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales)
            row=dict(step=step,phase=reference['phase'],**env.metrics)
            row.update(fingers.metrics());row.update(audit.metrics())
            row.update(q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),
                       cube_quaternion_wxyz=quaternion_from_native(tensor(env.cube.data.root_state_w)[0,3:7]).tolist(),
                       measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env))
            row['layer_measurement']=measure_layers(env.cube.states())
            if row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit']):failure='robot_safety_gate'
            if row['drop']:failure='cube_drop'
            if env.cube.pose_reset_count!=resets:raise RuntimeError('Unexpected cube pose write')
            if row['phase']=='center_hold':
                report=assess_center_row(row,names,plan['center_target_world_m'],plan['cube_target_quaternion_wxyz'])
                hold=hold+1 if report['qualified'] and not failure else 0;longest=max(longest,hold)
                row['center_audit']=report
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            if step%120==0 or failure:journal.flush();print('BC_EVAL',step,row['phase'],'hold',hold,'failure',failure,flush=True)
            if failure:break
    result=dict(scope='development_learned_left_recorded_right',seed=a.seed,steps=step+1,
                accepted=bool(not failure and longest>=90),failure=failure or ('center_hold_gate_not_met' if longest<90 else None),
                qualified_center_hold_s=longest/30.,elapsed_s=time.monotonic()-started,
                checkpoint_sha256=sha(a.checkpoint),source_manifest_sha256=sha(a.source/'artifact_manifest.json'),
                evaluator_sha256=sha(__file__),right_turn_trained=False,stage_gate_unlocked=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
except Exception as e:
    import traceback
    (a.output/'error.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
finally:
    from isaac_compat import close_application
    if env is not None:env.close()
    close_application(app)
