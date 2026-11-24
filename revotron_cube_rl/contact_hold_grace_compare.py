"""Compare archived round-15 control and actual diagnostic trajectories, without accessing a GPU.

python3 contact_hold_grace_compare.py [--treatment RUN21611 RUN21612] [--output NEW.json]
The retained-grasp/steady-grasp ranges are diagnostics; they never award benchmark success.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import takewhile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIT = ROOT / 'runs/contact_hold_cuda_long_v1/round_015/fit'
PANEL = FIT.parent / 'validation/panel.json'
PANEL_SHA = 'dbc6ecf51f3ef755eb0745cc8e4a2255d56809c934bd0e68c8d3b0eece98b6cd'
MODEL = '13953a5c8bf4522fa9735b5807b082ae51cdfaac2044c80c30ee2f0839d116c2'
SEEDS = (21611, 21612)
FINGERS = ('thumb','index','middle','ring','little')
IMPULSE = 1e-5
UNCHANGED = ('config.json','source_config.json','opposition_grasp_v20.json','student_left_projection.json',
             'face_contact_calibration_v1.json','cube_asset.json','solver_verification.json')
PHYSICAL_KEYS = ('step','phase','q','joint_velocity','cube_position','cube_quaternion_wxyz',
                 'finger_contact_impulses','height_m','cube_table_contact','applied_target')
SOURCE_FILES = ('opposition_teacher.py','environment.py','joint_safety.py','contact_hold_student_eval.py',
                'single_face_experiment.py','settled_wrist_teacher.py','contact_hold_teacher.py')


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def require(ok,msg):
    if not ok: raise ValueError(msg)


def contacts(r):
    c=r['finger_contact_impulses']['left']
    return tuple(f for f in FINGERS if c.get(f,0.)>IMPULSE)


def retained(r):
    fingers=contacts(r)
    return 'thumb' in fingers and len(fingers)>=4


def safe(r):
    return (not r['drop'] and r['robot_table_impulse']<=1e-6 and r['self_contact_impulse']<=1e-6
            and not r['arm_soft_limit_hit'] and not r['actual_arm_soft_limit_violation']
            and not any(r['hard_limit_hit']) and r['layer_measurement']['maximum_relative_rotation_deg']<=30.)


def first_counterfactual(rows,n):
    # Input to action t is the contact report from the executed transition t-1.
    streak=0
    for i in range(1,len(rows)):
        if rows[i]['phase']=='primary_lift':
            streak=0 if retained(rows[i-1]) else streak+1
            if streak>=n: return i
        else: streak=0
    return None


def longest(rows,predicate):
    first=None;span=None;size=0
    for r in rows:
        if predicate(r):
            if first is None: first=r['step']
            length=r['step']-first+1
            if length>size:size=length;span=(first,r['step'])
        else:first=None
    return dict(frames=size,step_range=span)


def finger_row(r):
    return dict(step=r['step'],phase=r['phase'],height_m=r['height_m'],on_table=r['cube_table_contact'],
                fingers=contacts(r),linear_m_s=r['linear_speed'],angular_rad_s=r['angular_speed'])


def load_episode(path,seed,baseline_record,expected_frames):
    path=path.resolve();mp=path/'artifact_manifest.json';m=json.loads(mp.read_bytes());h=m['sha256']
    def checked(name):
        data=(path/name).read_bytes()
        require(sha256(data).hexdigest()==h[name],f'Invalid manifest hash: {path/name}')
        return json.loads(data)
    md=digest(mp)
    if path==Path(baseline_record['run']).resolve():
        require(expected_frames==10 and md==baseline_record['manifest'],'Archived control identity mismatch')
    require(digest(path/'evaluated_best.pt')==h['evaluated_best.pt']==MODEL,'Different checkpoint in episode')
    r=checked('result.json');pr=checked('student_evaluation.json');rt=checked('runtime.json')
    rows=checked('trajectory.json')
    require(len(rows)>0 and len(rows)==r['steps']==r['student_executed_steps'] and
            all(v['step']==i and v['executed_left_source']=='student' for i,v in enumerate(rows)), 'Incomplete action history')
    require(r['seed']==pr['seed']==m['seed']==seed and r['checkpoint_sha256']==pr['checkpoint_sha256']==MODEL
            and r['best_update']==270000 and r['full_episode'] and m['full_episode'] and not r['execution_interrupted']
            and not r['cube_pose_writes_during_control'] and len({x['pose_reset_count'] for x in rows})==1
            and not r['eligible_training_episode'] and not r['corrective_labels_qualified_for_training'], 'Wrong or incomplete episode')
    require(rt['control_hz']==30 and rt['physics_hz']==240 and rt['physics_device']=='cpu' and rt['num_envs']==1,
            'Physics/timebase differs from fixed control')
    originals={}
    for name in UNCHANGED:
        checked(name); originals[name]=h[name]
    for name in SOURCE_FILES:
        require(digest(path/name)==h[name],'Archived source changed: '+name)
        originals[name]=h[name]
    mutation=None
    if expected_frames==20 or (path/'grace_intervention.json').is_file():
        mutation=checked('grace_intervention.json')
        require(mutation['seed']==seed and mutation['checkpoint_sha256']==MODEL and mutation['phase']=='primary_lift'
                and mutation['failure']=='lost_primary_grasp' and mutation['baseline_frames']==10
                and mutation['treatment_frames']==expected_frames and mutation['baseline_manifest_sha256']==baseline_record['manifest'],
                'Wrong intervention')
        require(digest(path/'grace_worker_source.py')==h['grace_worker_source.py']==mutation['worker_sha256']
                and mutation['original_module_sha256']==originals['opposition_teacher.py']
                and mutation['baseline_panel_sha256']==PANEL_SHA
                and r['diagnostic_only'] and pr['diagnostic_only']
                and r['grace_intervention_sha256']==pr['grace_intervention_sha256']==h['grace_intervention.json'],
                'Unbound experiment source or report')
    elif path!=Path(baseline_record['run']).resolve():
        require(pr['evaluation_role']=='validation' and not (path/'grace_intervention.json').exists(), 'Unidentified 10-frame control')
    return dict(run=str(path),result=r,rows=rows,originals=originals,manifest=md,
                trajectory_sha=h['trajectory.json'],mutation=mutation,cfg=checked('config.json'))


def summarize(e):
    r=e['result'];rows=e['rows'];final=rows[-1];trigger=first_counterfactual(rows,10)
    safe_all=all(safe(x) for x in rows)
    report=dict(run=e['run'],manifest_sha256=e['manifest'],trajectory_sha256=e['trajectory_sha'],
                intervention=e['mutation'],seed=r['seed'],steps=len(rows),phase_steps=dict(Counter(x['phase'] for x in rows)),
                failure=r['failure'],max_height_m=max(x['height_m'] for x in rows),last=finger_row(final),
                original_safety_pass=safe_all,wrist_hold_seconds=r['joint_wrist_hold_s'],
                original_wrist_criteria_met=bool(safe_all and r['dynamic_grasp_passed'] and not r['failure']
                         and r['wrist']['accepted'] and r['measured']['accepted'] and r['joint_wrist_hold_s']>=3.),
                ten_frame_trigger_action_step=trigger,twenty_frame_trigger_action_step=first_counterfactual(rows,20))
    if trigger is not None:
        after=rows[trigger:];s=e['cfg']['success']
        air=lambda v:not v['cube_table_contact'] and v['height_m']>=.08
        supported=lambda v:air(v) and retained(v) and safe(v) and v['layer_measurement']['assembled_for_grasp']
        steady=lambda v:supported(v) and v['linear_speed']<s['linear_speed_max'] and v['angular_speed']<s['angular_speed_max']
        report['after_ten_frame_trigger']=dict(trigger_inputs=[finger_row(v) for v in rows[trigger-10:trigger]],
                first_retained_step=next((v['step'] for v in after if retained(v)),None),
                lift_loss_input_streak_at_end=(
                    len(list(takewhile(lambda v:v['phase']=='primary_lift' and not retained(v),reversed(rows[:-1]))))
                    if final['phase']=='primary_lift' and r['failure']=='lost_primary_grasp' else None),
                final_postframe_count=len(after),frames_with_no_measured_left_finger=sum(not contacts(v) for v in after),
                first_table_contact_step=next((v['step'] for v in after if v['cube_table_contact']),None),
                max_height_m=max(v['height_m'] for v in after),min_height_m=min(v['height_m'] for v in after),
                longest_retained=longest(after,retained),longest_airborne_retained=longest(after,supported),
                longest_airborne_five_contact=longest(after,lambda v:supported(v) and len(contacts(v))==5),
                longest_airborne_steady_retained=longest(after,steady),
                later_phases=list(dict.fromkeys(v['phase'] for v in after)))
    return report


def compare(control,treatment):
    require(control['originals']==treatment['originals'],'Mismatch in archived physical/control configuration or source: '+
            ', '.join(k for k in control['originals'] if control['originals'][k]!=treatment['originals'][k]))
    a,b=control['rows'],treatment['rows'];ta,tb=first_counterfactual(a,10),first_counterfactual(b,10)
    n=min(t for t in [len(a),len(b),ta,tb] if t is not None)
    first_any=next((i for i in range(n) if a[i]!=b[i]),None)
    first={k:next((i for i in range(n) if a[i][k]!=b[i][k]),None) for k in PHYSICAL_KEYS}
    def max_delta(key):
        return max((max(abs(x-y) for x,y in zip(a[i][key],b[i][key])) for i in range(n)),default=None)
    return dict(same_checkpoint=True,same_recorded_controller_and_physics_config=True,prefix_frames_compared=n,
                comparison_excludes_first_intervened_action=True,first_any_row_difference_step=first_any,
                first_physical_difference_by_field=first,exact_physical_prefix=all(x is None for x in first.values()),
                prefix_max_abs_q_rad=max_delta('q'),prefix_max_abs_applied_target_rad=max_delta('applied_target'),
                prefix_max_cube_position_component_m=max_delta('cube_position'),
                trigger_steps_10_control_vs_treatment=[ta,tb],total_extra_steps=len(b)-len(a))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,nargs=2)
    p.add_argument('--treatment',type=Path,nargs=2);p.add_argument('--output',type=Path)
    a=p.parse_args()
    require(digest(PANEL)==PANEL_SHA and digest(FIT/'best.pt')==MODEL, 'Pinned round-15 baseline changed')
    ep=json.loads(PANEL.read_bytes())['episodes'];require(tuple(i['seed'] for i in ep)==SEEDS,'Validation seed order changed')
    output=dict(kind='frozen_round15_diagnostic_not_training_or_independent_test',checkpoint_sha256=MODEL,
                control_panel_sha256=PANEL_SHA,definitions={'phase_loss':'previous 10 or 20 primary_lift observations lacking thumb + >=3 other fingers at >1e-5 Ns',
                'airborne_retained':'original four-finger topology, >=0.08 m, no table, original safety and assembled cube',
                'airborne_steady_retained':'airborne_retained plus original configured pickup speed limits; diagnostic, not wrist success',
                'prefix':'same-seed traces compared from reset, excluding whichever run first reaches the 10-frame trigger action',
                'rate_hz':30},episodes=[])
    for i,(seed,rec) in enumerate(zip(SEEDS,ep)):
        ref=load_episode(Path(rec['run']),seed,rec,10)
        c=load_episode(a.control[i],seed,rec,10) if a.control else ref
        require(ref['originals']==c['originals'],'New control differs from historical control')
        entry={'seed':seed,'control':summarize(c)}
        if a.control:entry['historical_vs_current_control']=compare(ref,c)
        if a.treatment:
            t=load_episode(a.treatment[i],seed,rec,20);entry.update(treatment=summarize(t),pair=compare(c,t))
        output['episodes'].append(entry)
    text=json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(text)
    else: print(text,end='')


if __name__=='__main__':main()
