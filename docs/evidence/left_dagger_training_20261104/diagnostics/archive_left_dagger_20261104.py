"""Read-only run validation and isolated evidence archive; no simulator imports."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

REPO=Path('/home/brainco/rl_cube/github_project')
CODE=REPO/'revotron_cube_rl'
WORK=REPO/'work/teacher_optimization_20261103'
OUT=REPO/'docs/evidence/left_dagger_training_20261104'
RUNS=('left_dagger_baseline_preserve_9501_20261104',
      'left_dagger_baseline_preserve_v2_9501_20261104',
      'left_dagger_baseline_mixture_v3_9501_20261104')
sys.path.insert(0,str(CODE))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def vector_bits(values):return struct.pack('<'+'f'*len(values),*values)


def validate_run(run):
    protocol=json.loads((run/'runtime_protocol.json').read_text())
    checks={};missing=[]
    if (run/'artifact_manifest.json').exists():
        manifest=json.loads((run/'artifact_manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            actual=sha(run/name)
            if actual!=digest:raise ValueError('Artifact manifest mismatch: '+str(run/name))
            checks[name]=actual
    else:missing.append('artifact_manifest.json absent: initialization failed before finalize')
    snapshots={}
    for name,digest in protocol['source_sha256'].items():
        p=run/'source_snapshot'/name
        if sha(p)!=digest:raise ValueError('Runtime source snapshot mismatch: '+str(p))
        snapshots[name]=digest
    plans={}
    for p in (run/'plans').glob('*.json'):
        digest=p.stem.split('_',1)[1]
        if sha(p)!=digest:raise ValueError('Plan filename hash mismatch: '+str(p))
        plans[p.name]=digest
    source=CODE/'runs'/protocol['source_manifest']['run']
    if sha(source/'artifact_manifest.json')!=protocol['source_manifest']['manifest_sha256']:
        raise ValueError('Selected baseline manifest changed')
    checkpoint=CODE/'runs/teacher_phase_basis_gpu_v4_fullframe_20261104/best.pt'
    if sha(checkpoint)!=protocol['checkpoint_validation']['checkpoint_sha256']:
        raise ValueError('Behavior checkpoint changed')
    report={'run':run.name,'artifact_manifest_verified':bool(checks),'verified_artifact_sha256':checks,
            'source_snapshot_verified':True,'source_snapshot_sha256':snapshots,
            'plan_filename_sha256_verified':True,'plan_sha256':plans,'explicit_missing_artifacts':missing,
            'baseline_manifest_sha256':protocol['source_manifest']['manifest_sha256'],
            'behavior_checkpoint_sha256':protocol['checkpoint_validation']['checkpoint_sha256'],
            'physics_scope':protocol['physics_domain'],'teacher_probability':protocol['teacher_probability'],
            'eligible_training_episode':False,'pure_student_episode_claim':False}
    if (run/'error.json').exists():report['error']=json.loads((run/'error.json').read_text())
    if (run/'result.json').exists():report['result']=json.loads((run/'result.json').read_text())
    return report


def preserve_bridge(run):
    from palm_handover_audit import assess_handover
    from wrist_audit import assess_wrist
    from dagger_collection_core_v1 import physics_fingerprint
    trajectory=list(rows(run/'trajectory.jsonl'))
    source=CODE/'runs/center_j56_dev_v3_20261103'
    reference=json.loads((source/'trajectory.json').read_text())
    result=json.loads((run/'result.json').read_text())
    names=json.loads((run/'config.json').read_text())['joint_names']
    counts={'raw_done':0,'drop':0,'timeout':0,'unknown_done_cause':0,'phase_mismatch':0,'unsafe_frames':0}
    unequal={k:[] for k in ('applied_target','q','cube_position','cube_quaternion_wxyz','joint_velocity','finger_contact_impulses')}
    bit_unequal={k:[] for k in ('applied_target','q','cube_position')}
    for i,row in enumerate(trajectory):
        ref=reference[i]
        for k in unequal:
            if row[k]!=ref[k]:unequal[k].append(i)
        for k in bit_unequal:
            if vector_bits(row[k])!=vector_bits(ref[k]):bit_unequal[k].append(i)
        term=row['termination']
        counts['raw_done']+=int(term['raw_done']);counts['drop']+=int(row['drop'])
        counts['timeout']+=int(term['derived_timeout_truncated']);counts['unknown_done_cause']+=int(term['unknown_done_cause'])
        counts['phase_mismatch']+=int(row['phase']!=ref['phase'])
        counts['unsafe_frames']+=int(row['self_contact_impulse']>1e-6 or row['robot_table_impulse']>1e-6 or row['arm_soft_limit_hit'] or row['actual_arm_soft_limit_violation'] or any(row['hard_limit_hit']))
    measured=assess_handover(trajectory,names,regrasp=False,full_episode=True)
    wrist=assess_wrist([r if r['layer_measurement']['assembled_for_grasp'] else r|{'phase':'cube_not_assembled'} for r in trajectory],names)
    events=[];transition_count=0;aborts=[];alignment=[];modes={};teacher_count=0
    identity=json.loads((run/'physics_identity.json').read_text());identity_sha=physics_fingerprint(identity)
    for transition in rows(run/'transitions.jsonl'):
        if transition.get('record_type')=='abort':aborts.append(transition['step']);continue
        i=transition['step'];row=trajectory[i];expert=transition['expert_supervision'];transition_count+=1
        teacher_count+=int(transition['executed_source']=='teacher')
        modes[expert['execution_memory_mode']]=modes.get(expert['execution_memory_mode'],0)+1
        if transition['run_provenance']['physics_identity_sha256']!=identity_sha:raise ValueError('Transition physics domain mismatch')
        if transition['run_provenance']['physics_identity']!=identity:raise ValueError('Transition embedded physics differs')
        if transition['actual_applied_target']!=row['applied_target'] or expert['normalized_action']!=row['native_action'] or expert['requested_target']!=row['requested_target']:
            alignment.append(i)
        for key in ('q','applied_target','cube_position','joint_velocity','finger_contact_impulses'):
            if transition['actor_observation_post'][key]!=row[key]:alignment.append((i,'post',key))
            if i and transition['actor_observation_pre'][key]!=trajectory[i-1][key]:alignment.append((i,'pre',key))
        if expert['teacher_runtime_control_setting_change']:
            before=expert['control_settings_before'];after=expert['control_settings_after'];changed=[]
            for field in before:
                for j,(a,b) in enumerate(zip(before[field],after[field])):
                    if a!=b:changed.append({'field':field,'joint':names[j],'before':a,'after':b})
            events.append({'step':i,'phase':row['phase'],'changed':changed})
    passed=(len(trajectory)==1647 and transition_count==1647 and teacher_count==1647 and not aborts and not alignment
            and all(not v for v in unequal.values()) and all(not v for v in bit_unequal.values()) and not any(counts.values())
            and measured['accepted'] and wrist['accepted'] and result['cube_pose_writes_during_control']==0
            and len(events)==1 and events[0]['phase']=='tilt_hold' and len(events[0]['changed'])==21
            and all(c['field']=='joint_stiffness' and c['joint'].startswith('left_') and c['before']==30. and c['after']==20. for c in events[0]['changed']))
    return {'schema':'independent_baseline_dagger_preserve_bridge_validation_v1','passed':passed,
            'scope':'Original baseline seed 9501 reset through wrist gate only; pure-teacher adapter preservation, not student or generalization success',
            'frozen_result_unmodified':True,'frozen_result_sha256':sha(run/'result.json'),
            'source_run':source.name,'source_trajectory_sha256':sha(source/'trajectory.json'),
            'runtime_run':run.name,'runtime_trajectory_sha256':sha(run/'trajectory.jsonl'),
            'runtime_transitions_sha256':sha(run/'transitions.jsonl'),'frames':len(trajectory),
            'unequal_frames_by_field':unequal,'float32_bitwise_unequal_frames':bit_unequal,'counts':counts,
            'transitions':transition_count,'teacher_interventions':teacher_count,'execution_memory_modes':modes,
            'transition_alignment_failures':alignment,'abort_steps':aborts,'gain_events':events,
            'physics_identity_sha256':identity_sha,'recomputed_handover':measured,'recomputed_wrist':wrist,
            'cube_pose_writes':result['cube_pose_writes_during_control'],
            'eligible_training_episode':False,'pure_student_episode_claim':False,
            'not_validated':['mixed student rollout','other seeds','native articulation physics','center oracle','right-face turn','visual policy','PPO']}


def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    compressed=source.stat().st_size>100_000
    target=destination.with_name(destination.name+'.gz') if compressed else destination
    original_hash=sha(source)
    if compressed:
        with source.open('rb') as inp,target.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as out:shutil.copyfileobj(inp,out,4*1024*1024)
        check=hashlib.sha256()
        with gzip.open(target,'rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):check.update(chunk)
        if check.hexdigest()!=original_hash:raise ValueError('Compressed archive roundtrip mismatch')
    else:
        shutil.copy2(source,target)
        if sha(target)!=original_hash:raise ValueError('Archive copy mismatch')
    return {'source':str(source.relative_to(REPO)),'archive':str(target.relative_to(OUT)),
            'source_sha256':original_hash,'archive_sha256':sha(target),'source_bytes':source.stat().st_size,
            'archive_bytes':target.stat().st_size,'compression':'gzip_mtime0_level6' if compressed else None}


def main():
    OUT.mkdir(parents=True,exist_ok=True);archived=[];pending=[]
    for name in RUNS:
        run=CODE/'runs'/name
        if not (run/'result.json').exists() and not (run/'error.json').exists():pending.append(name);continue
        if (run/'result.json').exists() and not (run/'artifact_manifest.json').exists():pending.append(name);continue
        destination=OUT/name
        if (destination/'archive_file_index.json').exists():
            archived.append(name);continue
        report=validate_run(run)
        if 'preserve_v2' in name:
            bridge=preserve_bridge(run);write(OUT/'bridge_validation_v2.json',bridge)
            if not bridge['passed']:raise ValueError('Independent preserve bridge check failed; see sidecar')
        paths=[p for p in run.rglob('*') if p.is_file()]
        with ThreadPoolExecutor(max_workers=4) as pool:
            index=list(pool.map(lambda p:copy_file(p,destination/p.relative_to(run)),paths))
        write(destination/'archive_file_index.json',{'files':sorted(index,key=lambda x:x['archive'])})
        write(destination/'independent_validation.json',report)
        archived.append(name);print('ARCHIVED',name,'files',len(index),flush=True)
    diagnostics=OUT/'diagnostics';diagnostics.mkdir(exist_ok=True)
    for name in ('dagger_provenance_v3_independent_audit.json','audit_dagger_provenance_v3.py','dagger_v3_peer_review_20261104.json'):
        if (WORK/name).exists():copy_file(WORK/name,diagnostics/name)
    copy_file(Path(__file__),diagnostics/Path(__file__).name)
    for name in ('left_dagger_baseline_preserve','left_dagger_baseline_preserve_v2','left_dagger_baseline_mixture_v3'):
        run=RUNS[('left_dagger_baseline_preserve','left_dagger_baseline_preserve_v2','left_dagger_baseline_mixture_v3').index(name)]
        if run not in archived:continue
        for suffix in ('.log','_status.json'):
            if (WORK/(name+suffix)).exists():copy_file(WORK/(name+suffix),OUT/'logs'/(name+suffix))
    write(OUT/'archive_status.json',{'archived_runs':archived,'pending_runs':pending,'commit_performed':False,
        'scope':'Baseline privileged-state left pickup-to-wrist DAgger development; not a pure student success or full cube solution',
        'behavior_checkpoint_run':'teacher_phase_basis_gpu_v4_fullframe_20261104','behavior_checkpoint_sha256':'d2493b00df27b8d84ae69da3b399b3beb926b77fdd87a9b51aa6a2b1b23a36a4'})
    (OUT/'README.md').write_text('''# Left DAgger evidence, 2026-11-04

The v1 preserve attempt failed before any control action because strict JSON rejected a legitimate unbounded USD joint limit. The original error and zero-row trajectory are retained.

The metadata-only v2 preserve run executed 1647 original teacher actions on the selected baseline ordinary 27-body cube. `bridge_validation_v2.json` independently compares every control target, joint position and cube position with the accepted source, recomputes the original wrist/handover gate, and audits done signals and the original gain change. The frozen result is not edited. This validates the single-seed teacher adapter, not a learned policy or generalization.

The v3 run, once present, is an explicit teacher/student mixture with left-only v4 behavior, actual-state expert queries, projected expert supervision and conservative termination on every original done/drop/timeout. Teacher interventions and failed or incomplete episodes remain visible. `eligible_training_episode` and `pure_student_episode_claim` remain false; transition use requires a separate data audit. These data must not be pooled with D6-damped or native-articulation physics.

Each run contains the original manifest, source snapshot, plans and result/error. Large files use deterministic gzip; `archive_file_index.json` records both compressed and original hashes. Decompress before using the v3 provenance reader. The archive-level manifest verifies archived bytes. `archive_status.json` states whether v3 is still pending. No checkpoint is duplicated: the v4 behavior checkpoint SHA256 is recorded there.

This archive does not claim free face rotation, center-policy completion, visual solving, PPO success, or an independent test seed. No original run or frozen result is overwritten.
''')
    manifest={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'manifest.json'}
    write(OUT/'manifest.json',{'sha256':manifest,'files':len(manifest)})
    print('ARCHIVE_STATUS',json.dumps({'archived':archived,'pending':pending,'files':len(manifest)}),flush=True)


if __name__=='__main__':main()
