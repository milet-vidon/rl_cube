"""Recompute the original bounded palm correction at its rejected query, CPU only."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import kinematics
from diagnose_hold_increment_actual_contact_v8 import ROOT,CODE,WORK,read,sha


def main():
    run=CODE/'runs/left_hold_increment_v8_seed20261105_eval_9501_20261104'
    out=WORK/'v8_seed20261105_original_pose_envelope_reconstruction_20261104.json'
    assert not out.exists()
    assert sha(kinematics.__file__)==sha(run/'source_snapshot/kinematics.py')
    names=read(run/'projection_parameters.json')['joint_names'];robot=kinematics.RobotKinematics();last=None;query=None;physical=0
    for line in (run/'evaluation_transitions.jsonl').open():
        r=json.loads(line)
        if r['record_type']=='executed_transition':last=r;physical+=1
        elif r['record_type']=='abort':query=r
    assert query['reason']=='evaluation_protocol_error:Post-turn correction exceeded bounded pose envelope'
    assert query['step']==physical and not query['physics_step_completed_for_unfinished_decision']
    state=last['original_controller_audit']['teacher_state_after_query'];post=last['actor_observation_post']
    pathfile=run/'plans'/('tilt_'+state['path_sha256']+'.json');path=read(pathfile)
    assert state['mode']=='tilt'and state['tick']>=len(path)and(state['tick']-len(path))%10==0
    quat=np.array(post['cube_quaternion_wxyz']);reference=np.array(state['reference'])
    actual=Rotation.from_quat(quat[[1,2,3,0]])
    desired=Rotation.from_euler('x',-90.,degrees=True)*Rotation.from_quat(reference[[1,2,3,0]])
    error=(desired*actual.inv()).as_rotvec();angle=np.linalg.norm(error);assert angle>np.radians(1.)
    delta=Rotation.from_rotvec(error*min(.5,np.radians(1.)/angle)).as_matrix()
    palm=robot.fk_link(dict(zip(names,path[-1])),'left_palm_link');center=np.array(state['hold_center']);p0=np.array(state['hold_reference_palm'])
    position=center+delta@(palm[:3,3]-center);rotation=delta@palm[:3,:3]
    previous_angle=np.degrees(Rotation.from_matrix(palm[:3,:3]@p0[:3,:3].T).magnitude())
    new_angle=np.degrees(Rotation.from_matrix(rotation@p0[:3,:3].T).magnitude())
    displacement=np.linalg.norm(position-p0[:3,3]);assert displacement>.04 or new_angle>15.
    report={'schema':'original_palm_envelope_failure_cpu_reconstruction_v1','run':str(run),'physical_steps':physical,'rejected_step':query['step'],
        'original_failure':query['reason'],'teacher_tick':state['tick'],'original_path_length':len(path),
        'current_cube_tilt_error_deg':float(np.degrees(angle)),'this_proposed_correction_angle_deg':float(np.degrees(Rotation.from_matrix(delta).magnitude())),
        'previous_command_accumulated_rotation_deg':float(previous_angle),'proposed_accumulated_rotation_deg':float(new_angle),
        'proposed_position_displacement_m':float(displacement),'rotation_limit_deg':15.,'position_limit_m':.04,
        'rotation_limit_exceeded':bool(new_angle>15.),'position_limit_exceeded':bool(displacement>.04),
        'input_teacher_after_last_executed_query':state,'input_actual_post_state':post,'last_commanded_palm_transform':palm.tolist(),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p)for p in (Path(__file__).resolve(),run/'result.json',run/'evaluation_transitions.jsonl',run/'source_snapshot/palm_handover.py',Path(kinematics.__file__).resolve(),pathfile,ROOT/'revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf')},
        'no_failed_query_action_was_executed':True,'no_source_gate_or_history_modified':True,'isaac_started':False,
        'inference_limit':'Original candidate pose reconstructed from last completed snapshot and unchanged last post-physics state; exception has no after-query snapshot and no projected label is invented.'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('input_teacher_after_last_executed_query','input_actual_post_state','source_sha256','last_commanded_palm_transform')},indent=2));print('sha256',sha(out))


if __name__=='__main__':main()
