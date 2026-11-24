"""Matched v6/v4 left-position-policy evaluation from reset; no left takeover.

One externally scheduled Isaac instance only. --preflight-only performs no
simulator launch. Checkpoint and qualification SHA256 values are mandatory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import signal
import traceback

import torch

import phase_residual_policy_v5
import train_phase_residual_v5
import dagger_residual_dataset_v5
import left_dagger_baseline_runtime_v3
import state_residual_eval_runtime_v6
import phase_gated_residual_policy_v6
import train_phase_gated_residual_v6
import eval_prefix_dataset_v6
from train_phase_residual_v5 import load_baseline
from train_phase_gated_residual_v6 import load_checkpoint, verify_saved_datasets
from left_dagger_baseline_runtime_v3 import source_preflight, checkpoint_preflight
from state_residual_eval_runtime_v6 import EvaluationRuntime, evaluate, validate_joint_contract


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_bytes(source, destination, expected_sha256=None):
    """Freeze the exact evaluated bytes, even if another job writes its input."""
    data=Path(source).read_bytes(); digest=hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Explicit checkpoint/qualification SHA256 mismatch: '+str(source))
    with Path(destination).open('xb') as stream:stream.write(data)
    return digest


def prepare(args):
    preflight=source_preflight(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    snapshots=args.output/'checkpoints';snapshots.mkdir()
    policy_path=snapshots/'evaluated_policy.pt';baseline_path=snapshots/'frozen_v4_baseline.pt'
    policy_hash=snapshot_bytes(args.checkpoint,policy_path,args.expected_checkpoint_sha256)
    baseline_hash=snapshot_bytes(args.baseline,baseline_path)
    qualification_path=args.output/'reviewed_transition_qualification.json'
    qualification_hash=snapshot_bytes(args.qualification,qualification_path,args.qualification_sha256)
    qualification=json.loads(qualification_path.read_text())
    if (qualification.get('schema')!=train_phase_gated_residual_v6.QUALIFICATION_SCHEMA
            or qualification.get('eligible_for_bc_transition_supervision') is not True
            or qualification.get('eligible_for_rl_episode_training') is not False
            or qualification.get('pure_student_episode_success_claim') is not False
            or qualification.get('independent_generalization_claim') is not False
            or qualification.get('original_result_files_modified') is not False):
        raise ValueError('Explicit development-transition qualification required')
    if (qualification['source_sha256']!=train_phase_gated_residual_v6.source_hashes()
            or qualification['model_schema']!=phase_gated_residual_policy_v6.SCHEMA
            or qualification['feature_schema']!=phase_gated_residual_policy_v6.FEATURE_SCHEMA
            or qualification['gate_contract']!=phase_gated_residual_policy_v6.GATE_CONTRACT):
        raise ValueError('Reviewed V6 source/model/feature/public gate qualification changed')
    if len(args.dataset)!=2 or len({str(path.resolve())for path in args.dataset})!=2:
        raise ValueError('Both distinct reviewed development histories are required')
    baseline_saved,checked_baseline_hash=load_baseline(baseline_path)
    if checked_baseline_hash!=baseline_hash or qualification['frozen_baseline_checkpoint_sha256']!=baseline_hash:
        raise ValueError('Qualified frozen v4 baseline differs from evaluation baseline')
    preflight['checkpoint_validation']=checkpoint_preflight(baseline_path,preflight)
    family=qualification['physics_family_sha256']
    if args.policy=='v6':
        model,metadata=load_checkpoint(policy_path,baseline_path,args.dataset,expected_physics_family=family)
        if metadata['reviewed_qualification_sha256']!=qualification_hash:
            raise ValueError('Policy was trained with a different reviewed qualification')
        if metadata['feature_schema']!=phase_residual_policy_v5.FEATURE_SCHEMA or model.mean.numel()!=phase_residual_policy_v5.FEATURES:
            raise ValueError('Actual V6 feature layout differs from checkpoint metadata')
    else:
        if policy_hash!=baseline_hash:raise ValueError('Matched v4 control must use exactly the declared frozen baseline')
        import phase_basis_policy_v4
        model=phase_basis_policy_v4.restored(baseline_saved).eval()
        metadata=baseline_saved['metadata']
        verify_saved_datasets({'dataset_evidence':[d['evidence']for d in qualification['datasets']],
            'physics_family_sha256':family},args.dataset)
    validate_joint_contract(metadata,baseline_saved['metadata'],preflight['names'],preflight['left'])
    source_modules=(phase_residual_policy_v5,train_phase_residual_v5,
                    dagger_residual_dataset_v5,phase_gated_residual_policy_v6,
                    train_phase_gated_residual_v6,eval_prefix_dataset_v6,state_residual_eval_runtime_v6)
    identity={'policy':args.policy,'seed':args.seed,'split':'development_seen_seed',
        'episode_name':args.name,'max_steps':args.max_steps,
        'baseline_runtime_module':str(Path(left_dagger_baseline_runtime_v3.__file__).resolve()),
        'baseline_runtime_source_root':str(left_dagger_baseline_runtime_v3.ROOT.resolve()),
        'selection_basis':'development offline fit only; no independent generalization or closed-loop selection claim',
        'evaluated_policy_checkpoint_sha256':policy_hash,'evaluated_policy_snapshot':str(policy_path),
        'frozen_v4_baseline_checkpoint_sha256':baseline_hash,'baseline_snapshot':str(baseline_path),
        'qualification_sha256':qualification_hash,'expected_physics_family_sha256':family,
        'qualification_schema':qualification['schema'],'development_histories':2,
        'public_residual_gate_contract':phase_gated_residual_policy_v6.GATE_CONTRACT if args.policy=='v6'else None,
        'residual_bound_joint_names':qualification['residual_bound_joint_names'] if args.policy=='v6'else None,
        'residual_bound_rad':qualification['residual_bound_rad'] if args.policy=='v6'else None,
        'left_teacher_position_action_steps':0,'left_policy_begins_at_reset_step':0,
        'right_and_public_stage_controller':'original teacher',
        'original_teacher_left_gain_schedule':'first tilt_hold K20, explicitly audited; not a neural gain policy',
        'actor_state_scope':'actual privileged simulator q/qd/commands/cube/contact history; not vision',
        'actor_feature_schema':metadata.get('feature_schema','frozen_v4_198_phase_basis_features'),
        'actor_feature_count':phase_residual_policy_v5.FEATURES if args.policy=='v6'else 198,
        'actor_feature_layout':{'q':[0,56],'qd':[56,112],'applied_target':[112,168],'cube_position':[168,171],
            'cube_quaternion':[171,175],'point_impulses':[175,185],'phase':[185,197],'phase_progress':[197,198],
            **({'cube_velocity':[198,204],'finger_impulses':[204,214],'contact_presence':[214,224],
                'relative_contact_centroids':[224,254],'contact_normals':[254,284]}if args.policy=='v6'else {})},
        'expert_hidden_or_labels_in_actor':False,'executor_identity_in_actor_features':False,
        'corrective_expert_labels':'counterfactual audit only; separate future qualification required',
        'source_sha256':{Path(module.__file__).name:sha(module.__file__)for module in
            source_modules}|{Path(__file__).name:sha(__file__)}}
    source_directory=args.output/'evaluation_source_snapshot';source_directory.mkdir()
    for module in source_modules:
        (source_directory/Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes())
    (source_directory/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output/'evaluation_preflight.json').write_text(json.dumps(identity,indent=2,allow_nan=False))
    return preflight,model,metadata,baseline_saved['metadata'],identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=('v6','v4'),default='v6')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--qualification-sha256',required=True)
    parser.add_argument('--dataset',type=Path,action='append',required=True)
    parser.add_argument('--source',type=Path,default=Path('runs/center_j56_dev_v3_20261103'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name')
    parser.add_argument('--seed',type=int,default=9501)
    parser.add_argument('--max-steps',type=int,default=2000)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.max_steps<1:parser.error('Positive finite evaluation step budget required')
    if not args.name:args.name=args.output.name
    args.teacher_probability=0.  # original runtime metadata role, never a mixture selection
    stopped=[False]
    def stop_handler(signum,frame):stopped[0]=True
    signal.signal(signal.SIGTERM,stop_handler)
    runtime=None
    try:
        preflight,model,metadata,base_metadata,identity=prepare(args)
        if args.preflight_only:
            print('EVALUATION_PREFLIGHT_PASSED',json.dumps(identity),flush=True);return
        runtime=EvaluationRuntime(args,preflight,model,metadata,base_metadata,identity)
        with (args.output/'evaluation_transitions.jsonl').open('w')as journal:
            summary=evaluate(runtime,journal,max_steps=args.max_steps,stopped=lambda:stopped[0])
        result=runtime.finalize_evaluation(summary)
        print('EVALUATION_RESULT',json.dumps(result),flush=True)
    except BaseException as error:
        if args.output.exists():
            (args.output/'evaluation_error.json').write_text(json.dumps({'type':type(error).__name__,
                'message':str(error),'traceback':traceback.format_exc()},indent=2))
        raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':main()
