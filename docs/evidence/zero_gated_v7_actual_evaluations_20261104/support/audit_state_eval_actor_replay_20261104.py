"""CPU-only replay of logged actor inputs; no teacher labels fed to the actor."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from dagger_collection_core_v1 import STATE_FIELDS
from phase_residual_policy_v5 import EpisodeActor,features
from state_residual_eval_runtime_v5 import V4EpisodeActor
from phase_contact_policy import features as v4_features
from train_phase_residual_v5 import load_checkpoint
from phase_basis_policy_v4 import restored

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output-directory',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(1)
run=args.run;identity=json.loads((run/'evaluation_protocol.json').read_text())
baseline=run/'checkpoints/frozen_v4_baseline.pt';policy=run/'checkpoints/evaluated_policy.pt'
assert sha(policy)==identity['evaluated_policy_checkpoint_sha256']
dataset=Path('runs/left_dagger_baseline_mixture_v3_9501_20261104')
if identity['policy']=='v5':
    model,metadata=load_checkpoint(policy,baseline,[dataset],expected_physics_family=identity['actual_physics_family_sha256'])
    actor=EpisodeActor(model,metadata['output_indices']);feature_function=features
else:
    saved=torch.load(policy,map_location='cpu',weights_only=True);metadata=saved['metadata']
    actor=V4EpisodeActor(restored(saved),metadata['output_indices'])
    feature_function=lambda obs:v4_features(obs,obs['phase'],obs['phase_progress'])
actor.reset(identity['episode_name'])
count=0;feature_error=0.;prediction_error=0.;first_feature_mismatch=None;first_prediction_mismatch=None
previous_post=None;continuity_failures=[];schedule_failures=[];shift_errors=[]
first_eval_features=None
with (run/'evaluation_transitions.jsonl').open()as stream:
    for line in stream:
        try:record=json.loads(line)
        except json.JSONDecodeError:break
        if record['record_type']!='executed_transition':continue
        assert record['step']==count
        obs={**record['actor_observation_pre'],'previous_executed_source':'student'}
        actual=feature_function(obs);logged=np.asarray(record['actual_actor_features'],dtype=np.float32)
        delta=float(np.max(abs(actual-logged)));feature_error=max(feature_error,delta)
        if delta and first_feature_mismatch is None:first_feature_mismatch=count
        if first_eval_features is None:first_eval_features=actual
        expected=actor.act(obs,episode_id=identity['episode_name'],step=count)
        delta=float(np.max(abs(np.asarray(expected)-np.asarray(record['predicted_left_target']))))
        prediction_error=max(prediction_error,delta)
        if delta and first_prediction_mismatch is None:first_prediction_mismatch=count
        schedule=record['original_controller_audit']['public_same_action_schedule']
        if any(obs[k]!=schedule[k]for k in ('phase','phase_progress')):schedule_failures.append(count)
        if previous_post is not None:
            bad=[key for key in STATE_FIELDS if obs[key]!=previous_post[key]]
            if bad:continuity_failures.append({'step':count,'fields':bad})
        previous_post=record['actor_observation_post'];count+=1
with (dataset/'transitions.jsonl').open()as stream:first_training=json.loads(next(stream))['actor_observation_pre']
first_training_features=feature_function(first_training)
first_feature_vs_training=None if first_eval_features is None else float(np.max(abs(first_eval_features-first_training_features)))
report={'scope':'CPU reconstruction of actual logged own-rollout features and recurrent policy output; not a simulator run or training qualification',
    'run':str(run),'policy':identity['policy'],'completed_episode_result_exists':(run/'result.json').exists(),
    'replayed_complete_transitions':count,'explicit_actor_reset_count':1,
    'maximum_feature_error':feature_error,'first_feature_mismatch_step':first_feature_mismatch,
    'maximum_predicted_target_error_rad':prediction_error,'first_prediction_mismatch_step':first_prediction_mismatch,
    'actual_history_continuity_failures':continuity_failures,'same_action_phase_schedule_failures':schedule_failures,
    'first_frame_feature_max_difference_from_training_mixture':first_feature_vs_training,
    'checkpoint_sha256':sha(policy),'actual_expected_physics_family':identity['actual_physics_family_sha256'],
    'actor_expert_label_inputs':False,'executor_identity_in_features':False,
    'all_checks_passed':not(feature_error or prediction_error or continuity_failures or schedule_failures or first_feature_vs_training),
    'later_state_divergence_is_not_evidence_of_schema_shift':True,'eligible_for_training':False,
    'source_sha256':sha(__file__)}
out=args.output_directory/f"{identity['policy']}_actor_replay_{count:04d}_20261104.json"
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report|{'output':str(out),'report_sha256':sha(out)}))
