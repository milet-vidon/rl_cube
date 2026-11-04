"""Independent actual-state/projection/termination qualification; no simulator."""
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,struct,sys
from pathlib import Path
import torch

REPO=Path('/home/brainco/rl_cube/github_project');CODE=REPO/'revotron_cube_rl'
OUT=REPO/'docs/evidence/left_dagger_training_20261104'
RUN=CODE/'runs/left_dagger_baseline_mixture_v3_9501_20261104'
sys.path.insert(0,str(CODE))
from dagger_projected_collection_v3 import expand_record_provenance
from dagger_collection_core_v1 import STATE_FIELDS,canonical,validate_gate,physics_fingerprint
from joint_safety import project_target
from archive_left_dagger_20261104 import sha,write,rows


def bits(a,b):
    a=torch.as_tensor(a,dtype=torch.float32);b=torch.as_tensor(b,dtype=torch.float32)
    return a.shape==b.shape and bool(torch.equal(a.view(torch.int32),b.view(torch.int32)))


def main():
    manifest=json.loads((RUN/'artifact_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(RUN/name)!=digest:raise ValueError('Manifest mismatch '+name)
    result=json.loads((RUN/'result.json').read_text());trajectory=list(rows(RUN/'trajectory.jsonl'))
    parameters=json.loads((RUN/'projection_parameters.json').read_text())
    first=next(rows(RUN/'transitions.jsonl'));full=expand_record_provenance(first,RUN)['run_provenance'];reference=first['run_provenance']
    if parameters['source_sha256']['joint_safety.py']!=sha(CODE/'joint_safety.py'):raise ValueError('Original projection changed')
    names=parameters['joint_names'];left=[i for i,n in enumerate(names) if n.startswith('left_') or n.endswith('_L')];right=[i for i in range(56) if i not in left]
    param={k:torch.tensor(parameters[k],dtype=torch.float32) for k in ('scales','limits','step_limit','command_velocity_limits')}
    counts=Counter();phases=defaultdict(Counter);sync_joints=Counter();firsts={};gain_events=[];terminal_steps=[];failures=[]
    prior=None;terminal=False
    for line,record in enumerate(rows(RUN/'transitions.jsonl')):
        if record['run_provenance']!=reference:raise ValueError('Provenance reference changed')
        if terminal:raise ValueError('Record after terminal')
        if record['step']!=line:raise ValueError('Step gap')
        if record.get('record_type')=='abort':raise ValueError('Unexpected abort in completed passed mixture')
        if record['supervision_schema']!='baseline_left_projected_expert_v3':raise ValueError('Wrong projected schema')
        pre=record['actor_observation_pre'];post=record['actor_observation_post'];expert=record['expert_supervision'];row=trajectory[line]
        source=record['executed_source'];phase=pre['phase'];counts[source]+=1;phases[phase][source]+=1
        validate_gate(record['pre_gates']);validate_gate(record['post_gates'],stage_required=record['stage_complete'])
        if record['failure_reason'] or expert['failure_reason']:failures.append(line)
        if pre['phase']!=expert['public_same_action_schedule']['phase'] or pre['phase_progress']!=expert['public_same_action_schedule']['phase_progress']:raise ValueError('Wrong action schedule')
        if prior is None:
            if pre['previous_executed_source']!='reset':raise ValueError('Not from reset')
        else:
            if pre['previous_executed_source']!=prior['executed_source']:raise ValueError('Previous execution identity mismatch')
            for key in STATE_FIELDS:
                if canonical(pre[key])!=canonical(prior['actor_observation_post'][key]):raise ValueError('Actual state discontinuity '+key)
        if expert['projection_parameters_file_sha256']!=sha(RUN/'projection_parameters.json') or expert['projection_source_sha256']!=parameters['source_sha256']:raise ValueError('Projection provenance mismatch')
        old=torch.tensor(pre['applied_target'],dtype=torch.float32).reshape(1,56)
        action=torch.tensor(expert['normalized_action'],dtype=torch.float32).reshape(1,56)
        requested=old+param['scales']*action
        projected,hit=project_target(old,requested,param['limits'],param['step_limit'],param['command_velocity_limits'],parameters['control_dt'])
        if not bits(expert['projection_previous_applied_target'],old[0]) or not bits(expert['requested_target'],requested[0]) or not bits(expert['projected_target'],projected[0]):raise ValueError('Expert projection differs bitwise')
        if expert['projection_soft_limit_hit']!=hit[0].tolist():raise ValueError('Projection hit mismatch')
        counts['recomputed_projected_labels']+=1
        if source=='teacher':
            if record['expert_projection_matches_teacher_actual_bitwise'] is not True or not bits(projected[0],record['actual_applied_target']):raise ValueError('Teacher actual differs bitwise')
            counts['teacher_projected_actual_bitwise_verified']+=1
        else:
            firsts.setdefault('student_step',line)
            if record['expert_projection_matches_teacher_actual_bitwise'] is not None:raise ValueError('Student actual substituted for expert verification')
            student=torch.tensor(record['student_proposed_left_target'],dtype=torch.float32).reshape(1,28)
            action[:,left]=(student-old[:,left])/param['scales'][:,left]
        if not bits(row['native_action'],action[0]):raise ValueError('Executed native action differs from declared selection')
        if not bits(torch.tensor(row['native_action'])[right],torch.tensor(expert['normalized_action'])[right]):raise ValueError('Right action changed')
        executed,_=project_target(old,old+param['scales']*action,param['limits'],param['step_limit'],param['command_velocity_limits'],parameters['control_dt'])
        if not bits(executed[0],record['actual_applied_target']) or not bits(record['actual_applied_target'],row['applied_target']):raise ValueError('Actual selected action projection mismatch')
        counts['selected_action_projection_verified']+=1
        for key in ('q','joint_velocity','applied_target','cube_position','finger_contact_impulses'):
            if canonical(post[key])!=canonical(row[key]):raise ValueError('Post state vs physical row mismatch '+key)
        mode=expert['execution_memory_mode'];counts['memory_mode_'+mode]+=1
        changes=expert['execution_memory_changes']
        if changes:
            counts['memory_sync_changed_frames']+=1;counts['memory_sync_changed_entries']+=len(changes)
            firsts.setdefault('memory_sync_changed_step',line)
            for change in changes:
                joint=change['path'].split('_applied_',1)[1];sync_joints[joint]+=1
                if change['after']!=pre['applied_target'][names.index(joint)]:raise ValueError('Memory sync does not use actual command')
        if expert['teacher_runtime_control_setting_change']:
            before=expert['control_settings_before'];after=expert['control_settings_after'];changes=[]
            for key in before:
                for i,(a,b) in enumerate(zip(before[key],after[key])):
                    if a!=b:changes.append({'field':key,'joint':names[i],'before':a,'after':b})
            gain_events.append({'step':line,'phase':phase,'changes':changes})
        term=row['termination']
        for key in ('raw_done','derived_timeout_truncated','derived_drop_terminated','generic_pickup_success','unknown_done_cause'):
            counts[key]+=int(term[key])
        if term['raw_truncated'] is not None or not term['done_matches_original_causes']:raise ValueError('Raw termination API mismatch')
        terminal=bool(record['terminal'])
        if terminal:terminal_steps.append(line)
        if row['drop'] or term['sampling_terminal']:
            if not terminal or line!=len(trajectory)-1:raise ValueError('Physical steps after terminal condition')
        prior=record
    if len(trajectory)!=sum(counts[k] for k in ('teacher','student')) or result['steps']!=len(trajectory):raise ValueError('Physical/transition count mismatch')
    if terminal_steps!=[len(trajectory)-1] or not prior['stage_complete'] or failures:raise ValueError('Not a single successful final stage terminal')
    if sha(RUN/'collection_provenance.json')!=reference['sha256']:raise ValueError('Provenance changed during audit')
    v2=CODE/'runs/left_dagger_baseline_preserve_v2_9501_20261104/physics_identity.json'
    old_identity=json.loads(v2.read_text());identity=full['physics_identity'];domain_diff={}
    for key in identity:
        if canonical(identity[key])!=canonical(old_identity[key]):domain_diff[key]={'v2':old_identity[key],'v3':identity[key]}
    report={'schema':'independent_actual_state_mixture_v3_development_label_qualification_v1','passed':True,
        'run':RUN.name,'artifact_manifest_sha256':sha(RUN/'artifact_manifest.json'),'files_sha256':manifest['sha256'],
        'scope':'Development imitation supervision from one previously seen baseline seed; not a pure student success, holdout, or cross-physics dataset',
        'eligible_for_development_transition_supervision':True,'eligible_training_episode':False,
        'pure_student_episode_claim':False,'independent_generalization_claim':False,
        'frames':len(trajectory),'counts':dict(counts),'student_coverage_by_phase':{k:dict(v) for k,v in phases.items()},
        'first_events':firsts,'memory_sync_changed_entries_by_joint':dict(sync_joints),'gain_events':gain_events,
        'terminal_steps':terminal_steps,'last_physical_step':trajectory[-1]['step'],'no_records_or_physical_rows_after_terminal':True,
        'source_labels':'Only recomputed expert_projected_target; student actual actions never serve as expert labels',
        'right_native_action_bits_preserved_every_frame':True,'original_grasp_gate':result['original_grasp_gate'],
        'physics_identity_sha256':physics_fingerprint(identity),'v2_physics_identity_sha256':physics_fingerprint(old_identity),
        'v2_to_v3_identity_differences':domain_diff,'recovery_or_new_run_physics_steps_not_in_episode':True}
    write(OUT/'mixture_v3_data_qualification.json',report)
    copy=OUT/'diagnostics'/Path(__file__).name
    copy.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'passed':True,'frames':len(trajectory),'counts':dict(counts),'phases':report['student_coverage_by_phase'],
        'terminal_steps':terminal_steps,'identity_difference_keys':list(domain_diff)},indent=2))


if __name__=='__main__':main()
