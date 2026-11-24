"""Offline round-15 comparison: physical grasp evidence without a finger-count requirement.

The formal 10-frame results are immutable. New runs describe diagnostic outcomes;
a matching prefix supports a same-state continuation, not a new evaluation seed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from contact_hold_grace_compare import (FIT, MODEL, PANEL, PANEL_SHA, PHYSICAL_KEYS, ROOT,
        SEEDS, digest, first_counterfactual, load_episode, require)
from contact_hold_physical_metrics import summarize_hold

RUNS = ROOT/'runs'
OLD20 = RUNS/'contact_hold_grace_r15_lift20_20261111_1932'
MODES = ('matched_lift','from_reset')
PHYSICS = ('runtime.json','solver_verification.json','config.json','source_config.json',
           'asset_contract.json','runtime_robot_asset.json','cube_asset.json','student_left_projection.json',
           'opposition_grasp_v20.json','face_contact_calibration_v1.json','center_j56_protocol_v3.json')
# Include the measured support, moving hand/cube and rejection variables. Raw contact point lists
# and cubelet dictionaries are checked as well; if exact reproducibility fails, say so.
KEYS = tuple(dict.fromkeys((*PHYSICAL_KEYS,'contact_impulse','table_impulse','linear_speed','angular_speed',
    'measured_hand_frames','cubelets','layer_measurement','finger_contact_points','robot_table_impulse',
    'self_contact_impulse','drop','hard_limit_hit','arm_soft_limit_hit','actual_arm_soft_limit_violation','pose_reset_count')))


def checked_file(directory, manifest, name):
    p=directory/name
    require(p.is_file() and digest(p)==manifest['sha256'][name],f'Changed or missing archived artifact: {p}')
    return p


def checked_json(directory, manifest, name):
    return json.loads(checked_file(directory,manifest,name).read_bytes())


def verify_configuration(directory, baseline):
    """Compare runtime, solver, physics assets, controller config and original harness sources."""
    m=json.loads((directory/'artifact_manifest.json').read_bytes())
    ref=json.loads((baseline/'artifact_manifest.json').read_bytes())
    for key in PHYSICS:
        # The original runtime serializes some USD layer maps in encounter order. Compare
        # validated JSON values, not text order; keep each artifact's own manifest digest.
        require(checked_json(directory,m,key)==checked_json(baseline,ref,key),'Different archived configuration: '+key)
    for key in (n for n in ref['sha256'] if n.endswith('.py')):
        checked_file(directory,m,key);checked_file(baseline,ref,key)
        require(m['sha256'][key]==ref['sha256'][key],'Different archived original source: '+key)
    a=checked_json(directory,m,'experiment.json');b=checked_json(baseline,ref,'experiment.json')
    require({k:v for k,v in a.items() if k!='name'}=={k:v for k,v in b.items() if k!='name'},
            'Original experiment options differ between paired runs')
    return m


def functional_episode(directory, seed, mode, ref, reference):
    m=verify_configuration(directory,Path(ref['run']));h=m['sha256'];md=digest(directory/'artifact_manifest.json')
    r=checked_json(directory,m,'result.json');p=checked_json(directory,m,'student_evaluation.json')
    receipt=checked_json(directory,m,'functional_intervention.json')
    require(r['seed']==p['seed']==m['seed']==seed and r['split']==m['split']=='development' and
            r['best_update']==270000 and r['full_episode'] and m['full_episode'] and not r['execution_interrupted']
            and not r['cube_pose_writes_during_control'] and r['checkpoint']=='evaluated_best.pt', 'Wrong or interrupted diagnostic')
    for data in (r,p):
        require(data['evaluation_role']=='evaluation' and data['diagnostic_only'] is True and data['functional_mode']==mode
                and data['kind']=='contact_hold_student_functional_diagnostic'
                and data['training_run']==str(FIT) and data['checkpoint_sha256']==MODEL
                and not data['eligible_training_episode'] and not data['corrective_labels_qualified_for_training']
                and data['functional_intervention_sha256']==h['functional_intervention.json'], 'Wrong diagnostic identity/eligibility')
    require(digest(checked_file(directory,m,'evaluated_best.pt'))==MODEL,'Model changed')
    for archive,source in [('functional_worker_source.py','contact_hold_functional_worker.py'),
                           ('physical_metrics_source.py','contact_hold_physical_metrics.py')]:
        checked_file(directory,m,archive)
        require(h[archive]==digest(ROOT/source),'Wrong diagnostic code: '+archive)
    require(receipt==json.loads((directory.parent/f'preflight_{seed}.json').read_bytes()) and
            receipt['kind']=='functional_hold_intervention_v1' and receipt['baseline_run']==ref['run'] and
            receipt['support']=='total measured left hand/cube impulse above cfg.success.contact_impulse_min; no named finger or count' and
            receipt['lift_postlift_loss_grace_frames']==20 and receipt['seed']==seed and receipt['mode']==mode and receipt['diagnostic_only'] is True and
            not receipt['eligible_training_episode'] and receipt['worker_sha256']==h['functional_worker_source.py'] and
            receipt['metrics_sha256']==h['physical_metrics_source.py'] and receipt['checkpoint_sha256']==MODEL
            and receipt['baseline_panel_sha256']==PANEL_SHA and receipt['baseline_manifest_sha256']==reference['manifest'],
            'Unbound diagnostic intervention')
    rows=checked_json(directory,m,'trajectory.json')
    require(rows and len(rows)==r['steps']==r['student_executed_steps'] and
            all(x['step']==i and x['executed_left_source']=='student' for i,x in enumerate(rows)) and
            len({x['pose_reset_count'] for x in rows})==1, 'Incomplete diagnostic trajectory')
    cfg=checked_json(directory,m,'config.json');summary=summarize_hold(rows,cfg)
    side=json.loads((directory/'physical_hold.json').read_bytes())
    require(side['source_manifest_sha256']==md and side['metrics_sha256']==h['physical_metrics_source.py'] and
            side['kind']=='functional_hold_diagnostic_only' and side['seed']==seed and side['mode']==mode
            and side['summary']==summary,'Physical hold sidecar does not match frozen metrics on archived observations')
    return dict(rows=rows,cfg=cfg,manifest=md,trajectory_sha=h['trajectory.json'],run=str(directory),
                result=r,sidecar_sha=digest(directory/'physical_hold.json'),mode=mode)


def summary(episode, rows):
    cfg=episode['cfg'];r=episode['result'];last=rows[-1];stats=summarize_hold(rows,cfg)
    end=stats['physical_steady']['ending_streak_frames']
    # These are distinct questions. Ever meeting the time window says nothing about what happens later.
    return dict(physical_hold=stats,ever_3s_stable=stats['physical_hold_observed'],
        ending_3s_stable=stats['physical_hold_at_end'],ending_steady_frames=end,steps=len(rows),final_step=last['step'],
        final_height_m=last['height_m'],final_linear_m_s=last['linear_speed'],final_angular_rad_s=last['angular_speed'],
        final_left_hand_impulse_Ns=last['contact_impulse'][0],final_table_contact=bool(last['cube_table_contact']),
        final_drop_flag=bool(last['drop']),end_reason=r['failure'],final_phase=last['phase'],
        phases=dict(Counter(x['phase'] for x in rows)))


def pair(ref,test,mode):
    a,b=ref['rows'],test['rows']
    ta=first_counterfactual(a,10)
    require(ta is not None and len(a)==ta+1 and ref['result']['failure']=='lost_primary_grasp','Archived trigger was not reproduced by contact history')
    # Use the full overlap for observing divergence, but separately compare through ACTION ta-1.
    # Row ta is a post-action observation, not an input to action ta.
    limit=min(len(a),len(b)); n=min(ta,limit)
    first={}
    for k in KEYS:
        first[k]=next((i for i in range(limit) if a[i][k]!=b[i][k]),None)
    first_step=min((v for v in first.values() if v is not None),default=None)
    prefix_ok=(len(b)>ta and all(v is None or v>=ta for v in first.values()))
    # No physically aligned post-cutoff statistics are presented if an earlier action/state diverged.
    differences={k:v for k,v in first.items() if v is not None}
    phase = ('primary_lift' if mode=='matched_lift' else 'squeeze' if mode=='from_reset' else None)
    bound = next((r['step'] for r in b if r['phase']==phase),None) if phase else first_counterfactual(b,10)
    return dict(mode=mode,earliest_potentially_modified_phase=phase,conservative_affected_phase_start=bound,
            control_trigger_action_step=ta,control_trigger_input_steps=[ta-10,ta-1],
            first_difference_step_in_overlap=first_step,first_difference_by_field=differences,
            first_action_or_phase_difference_step=min((i for i in (first['phase'],first['applied_target']) if i is not None),default=None),
            control_overlap_frames=limit,physical_prefix_frames_compared=n,
            matched_all_physical_fields_before_control_trigger=prefix_ok,
            allows_same_state_continuation_interpretation=prefix_ok,
            post_trigger_aligned=(summary(test,b[ta:]) if prefix_ok else None),
            independent_whole_rollout=summary(test,b),mode_note=(
                'from_reset may intentionally change squeeze timing; divergent runs are independent trajectories of the same seed'
                if mode=='from_reset' else 'retention can change after entering primary_lift; matching the old stop is measured, never presumed'))


def provenance(ep):
    return dict(run=ep['run'],manifest_sha256=ep['manifest'],trajectory_sha256=ep['trajectory_sha'],
                **({'physical_hold_sidecar_sha256':ep['sidecar_sha']} if 'sidecar_sha' in ep else {}))


def report(job,baseline_only):
    require(digest(PANEL)==PANEL_SHA and digest(FIT/'best.pt')==MODEL,'Frozen model or control panel changed')
    refs=json.loads(PANEL.read_bytes())['episodes'];require(tuple(i['seed'] for i in refs)==SEEDS,'Unexpected control seeds')
    oldstate=json.loads((OLD20/'state.json').read_bytes()) if (OLD20/'state.json').is_file() else {'status':'missing'}
    oldcomplete=oldstate.get('status')=='completed'
    if oldcomplete:
        require(digest(OLD20/'report.json')==oldstate['report_sha256'],'Changed previous 20-frame comparison')
    result=dict(schema='round15_functional_hold_without_finger_counts_v1',diagnostic_only=True,
                eligible_for_training=False,eligible_for_formal_validation=False,checkpoint_sha256=MODEL,
                baseline_panel_sha256=PANEL_SHA,metrics_sha256=digest(ROOT/'contact_hold_physical_metrics.py'),
                old_twenty_frame_status=oldstate.get('status'),baseline_only=baseline_only,
                physical_prefix_fields=KEYS,definition='summarize_hold() using original pickup thresholds, no required finger or pad count; 90 consecutive qualifying control frames at 30 Hz. The original wrist benchmark is separate.',
                episodes=[])
    for seed,rec in zip(SEEDS,refs):
        ref=load_episode(Path(rec['run']),seed,rec,10)
        require(ref['result']['evaluation_role']=='validation','Not a fixed validation baseline')
        t=first_counterfactual(ref['rows'],10)
        require(t is not None and t==len(ref['rows'])-1,'Baseline terminal action differs from original ten-frame trigger')
        entry=dict(seed=seed,control10=dict(**provenance(ref),whole=summary(ref,ref['rows']),original_formal_pass=rec['passed'],
                   trigger_action_step=t,trigger_input_steps=[t-10,t-1],after_trigger=summary(ref,ref['rows'][t:])))
        if oldcomplete:
            gp=OLD20/f'seed_{seed}';g=load_episode(gp,seed,rec,20);verify_configuration(gp,Path(rec['run']))
            entry['control20']=dict(**provenance(g),**pair(ref,g,'four_finger_grace20'))
        if not baseline_only:
            for mode in MODES:
                d=job/mode/f'seed_{seed}';e=functional_episode(d,seed,mode,ref,rec)
                entry[mode]=dict(**provenance(e),**pair(ref,e,mode))
        result['episodes'].append(entry)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();obj=report(args.job.resolve(),args.baseline_only)
    text=json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+'\n'
    with args.output.open('x') as out:out.write(text)
