"""V7 left-position policy from reset; original phase/right/gain controller.

Only the externally scheduled single Isaac process may execute this entry.
--preflight-only validates snapshots, data and checkpoint without a simulator.
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
import phase_gated_residual_policy_v6
import train_phase_gated_residual_v6
import eval_prefix_dataset_v7
import inspect_residual_dataset_v7
import zero_gated_residual_policy_v7
import train_zero_gated_residual_v7
import state_zero_gated_eval_runtime_v7
from train_phase_residual_v5 import load_baseline
from train_zero_gated_residual_v7 import load_checkpoint
from left_dagger_baseline_runtime_v3 import source_preflight, checkpoint_preflight
from state_zero_gated_eval_runtime_v7 import EvaluationRuntime, evaluate, validate_joint_contract


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_bytes(source,destination,expected_sha256=None):
    data=Path(source).read_bytes();digest=hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Explicit checkpoint/qualification/authorization SHA256 mismatch: '+str(source))
    with Path(destination).open('xb')as stream:stream.write(data)
    return digest


def validate_qualification(qualification,datasets):
    if (qualification.get('schema')!='v7_five_history_bc_integrity_candidate_sidecar_v1'
            or qualification.get('training_authorized')is not False
            or qualification.get('candidate_bc_transitions_passed_integrity')is not True
            or qualification.get('eligible_under_declared_v7_candidate_bounds')is not True
            or qualification.get('zero_class_definition_approved')is not True
            or qualification.get('eligible_for_rl_episode_training')is not False
            or qualification.get('pure_student_episode_success_claim')is not False
            or qualification.get('independent_generalization_claim')is not False
            or qualification.get('original_result_files_modified')is not False):
        raise ValueError('Unmodified reviewed development BC candidate qualification required')
    if (qualification['source_sha256']!=inspect_residual_dataset_v7.source_hashes()
            or qualification['current_candidate_gate']!=zero_gated_residual_policy_v7.PUBLIC_GATE_CONTRACT
            or qualification['classification_contract']!=zero_gated_residual_policy_v7.CLASSIFICATION_CONTRACT
            or qualification['active_max_unreachable_distance_rad']!=0.):
        raise ValueError('Reviewed V7 sources/public gate/classification/reachability changed')
    paths=[Path(path).resolve()for path in datasets]
    if (len(paths)!=5 or len(set(paths))!=5
            or tuple(path.name for path in paths)!=inspect_residual_dataset_v7.RUN_NAMES):
        raise ValueError('Exactly the ordered five reviewed development histories are required')


def validate_candidate_metadata(metadata):
    if (metadata.get('offline_candidate_qualified')is not True
            or metadata.get('best_epoch',0)<1 or metadata.get('optimizer_steps',0)<1
            or metadata.get('final_cpu_sequential_metrics',{}).get('step499_reviewed_zero_histories_bitwise_baseline')is not True
            or metadata.get('final_cpu_sequential_metrics',{}).get('inference')!='sequential_one_frame_hidden_state'):
        raise ValueError('Trained V7 candidate must pass the declared actual-history CPU development regression')
    for key in ('eligible_for_rl_episode_training','independent_generalization_demonstrated',
                'closed_loop_success_demonstrated','right28_learned'):
        if metadata.get(key)is not False:raise ValueError('Unreviewed checkpoint claim: '+key)


def prepare(args):
    preflight=source_preflight(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    snapshots=args.output/'checkpoints';snapshots.mkdir()
    policy_path=snapshots/'evaluated_policy.pt';baseline_path=snapshots/'frozen_v4_baseline.pt'
    policy_hash=snapshot_bytes(args.checkpoint,policy_path,args.expected_checkpoint_sha256)
    baseline_hash=snapshot_bytes(args.baseline,baseline_path)
    qualification_path=args.output/'reviewed_transition_qualification.json'
    qualification_hash=snapshot_bytes(args.qualification,qualification_path,args.qualification_sha256)
    authorization_path=args.output/'reviewed_training_authorization.json'
    authorization_hash=snapshot_bytes(args.authorization,authorization_path,args.authorization_sha256)
    qualification=json.loads(qualification_path.read_text())
    validate_qualification(qualification,args.dataset)
    baseline_saved,checked_baseline_hash=load_baseline(baseline_path)
    if checked_baseline_hash!=baseline_hash or qualification['frozen_baseline_checkpoint_sha256']!=baseline_hash:
        raise ValueError('Qualified frozen baseline differs from evaluation baseline')
    preflight['checkpoint_validation']=checkpoint_preflight(baseline_path,preflight)
    family=qualification['physics_family_sha256']
    model,metadata=load_checkpoint(policy_path,baseline_path,args.dataset,
        qualification_path=qualification_path,expected_qualification_sha256=qualification_hash,
        authorization_path=authorization_path,expected_authorization_sha256=authorization_hash,
        expected_physics_family=family,expected_checkpoint_sha256=policy_hash)
    if (metadata['qualification_sha256']!=qualification_hash
            or metadata['root_authorization_sha256']!=authorization_hash):
        raise ValueError('Policy training qualification/authorization differs from explicit snapshots')
    validate_candidate_metadata(metadata)
    if (metadata['feature_schema']!=zero_gated_residual_policy_v7.FEATURE_SCHEMA
            or model.mean.numel()!=zero_gated_residual_policy_v7.FEATURES):
        raise ValueError('Actual V7 feature layout differs from checkpoint metadata')
    validate_joint_contract(metadata,baseline_saved['metadata'],preflight['names'],preflight['left'])
    names=[preflight['names'][index]for index in preflight['left']]
    bounds=torch.tensor(zero_gated_residual_policy_v7.candidate_bounds(names),dtype=torch.float32)
    if (not torch.equal(model.residual_bound.cpu().view(torch.int32),bounds.view(torch.int32))
            or qualification['current_candidate_bounds_by_name']!=dict(zip(names,bounds.tolist()))):
        raise ValueError('Actual model bounds differ from reviewed named qualification')
    source_modules=(phase_residual_policy_v5,train_phase_residual_v5,dagger_residual_dataset_v5,
        phase_gated_residual_policy_v6,train_phase_gated_residual_v6,eval_prefix_dataset_v7,
        inspect_residual_dataset_v7,zero_gated_residual_policy_v7,train_zero_gated_residual_v7,
        state_zero_gated_eval_runtime_v7)
    identity={'policy':'v7','seed':args.seed,'split':'development_seen_seed',
        'episode_name':args.name,'max_steps':args.max_steps,
        'baseline_runtime_module':str(Path(left_dagger_baseline_runtime_v3.__file__).resolve()),
        'baseline_runtime_source_root':str(left_dagger_baseline_runtime_v3.ROOT.resolve()),
        'selection_basis':'development supervised fit and full actual-history gate regressions; no independent generalization',
        'evaluated_policy_checkpoint_sha256':policy_hash,'evaluated_policy_snapshot':str(policy_path),
        'frozen_v4_baseline_checkpoint_sha256':baseline_hash,'baseline_snapshot':str(baseline_path),
        'qualification_sha256':qualification_hash,'qualification_schema':qualification['schema'],
        'training_authorization_sha256':authorization_hash,'expected_physics_family_sha256':family,
        'development_histories':5,
        'public_residual_gate_contract':zero_gated_residual_policy_v7.PUBLIC_GATE_CONTRACT,
        'learned_zero_gate_contract':zero_gated_residual_policy_v7.ZERO_GATE_CONTRACT,
        'training_classification_contract':zero_gated_residual_policy_v7.CLASSIFICATION_CONTRACT,
        'residual_bound_joint_names':names,'residual_bound_rad':bounds.tolist(),
        'left_teacher_position_action_steps':0,'left_policy_begins_at_reset_step':0,
        'right_and_public_stage_controller':'original teacher',
        'original_teacher_left_gain_schedule':'first tilt_hold K20, explicitly audited; not a neural gain policy',
        'actor_state_scope':'actual privileged simulator q/qd/commands/cube/contact history; not vision',
        'actor_feature_schema':metadata['feature_schema'],'actor_feature_count':284,
        'actor_feature_layout':{'q':[0,56],'qd':[56,112],'applied_target':[112,168],
            'cube_position':[168,171],'cube_quaternion':[171,175],'point_impulses':[175,185],
            'phase':[185,197],'phase_progress':[197,198],'cube_velocity':[198,204],
            'finger_impulses':[204,214],'contact_presence':[214,224],
            'relative_contact_centroids':[224,254],'contact_normals':[254,284]},
        'expert_hidden_or_labels_in_actor':False,'executor_identity_in_actor_features':False,
        'corrective_expert_labels':'counterfactual audit only; separate future qualification required',
        'source_sha256':{Path(module.__file__).name:sha(module.__file__)for module in source_modules}
            |{Path(__file__).name:sha(__file__)}}
    source_directory=args.output/'evaluation_source_snapshot';source_directory.mkdir()
    for module in source_modules:
        (source_directory/Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes())
    (source_directory/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output/'evaluation_preflight.json').write_text(json.dumps(identity,indent=2,allow_nan=False))
    return preflight,model,metadata,baseline_saved['metadata'],identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=('v7',),default='v7')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--qualification-sha256',required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--authorization-sha256',required=True)
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
    args.teacher_probability=0.
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
