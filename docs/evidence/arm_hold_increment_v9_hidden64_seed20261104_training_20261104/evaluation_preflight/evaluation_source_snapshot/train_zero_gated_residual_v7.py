"""Qualified five-history development BC; no Isaac or online intervention.

The reviewed candidate qualification remains immutable and unauthorized by
itself. An explicit, hash-pinned root authorization is required to optimize.
Every raw continuous label enters regression. Ambiguous small labels are
excluded only from BCE, never clipped or changed into zero labels.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch

import inspect_residual_dataset_v7 as inspection
import eval_prefix_dataset_v7 as reader
import dagger_residual_dataset_v5 as mixture_reader
import train_phase_residual_v5 as frozen
import train_phase_gated_residual_v6 as prior
import zero_gated_residual_policy_v7 as policy
from phase_contact_policy import PHASES

AUTH_SCHEMA = 'root_reviewed_v7_five_history_development_bc_authorization_v1'
QUAL_SCHEMA = 'v7_five_history_bc_integrity_candidate_sidecar_v1'
BASELINE_SHA = 'd2493b00df27b8d84ae69da3b399b3beb926b77fdd87a9b51aa6a2b1b23a36a4'
FAMILY_SHA = 'b2f9feb645c8d92e2da3bb78653165292a5649e59864dee4b1de82e904854192'
ZERO_BOUNDARY_RUNS = tuple(name for name in inspection.RUN_NAMES if name != 'left_state_residual_v5_long_eval_9501_20261104_v2')
RESULT_FLAGS = ('eligible_training_episode', 'runtime_bridge_validated', 'pure_student_episode_claim',
                'corrective_labels_qualified_for_training', 'original_wrist_gate_qualified', 'failure', 'collection_mode')
SELECTION_CONTRACT = {
    'ordering': ['step499_proposal_native_projected_all28_bitwise_baseline',
                 'strict_zero_false_positive_plus_certain_nonzero_false_negative_count',
                 'hard_decoded_projected_rmse_rad'],
    'initial_untrained_epoch_is_candidate': False,
    'candidate_save_requires_cpu_streamed_boundary_check': True,
    'final_metrics_use_cpu_sequential_actor_history': True,
    'step499_zero_regression_runs': list(ZERO_BOUNDARY_RUNS),
    'step499_v5_failed_history_has_eight_real_corrections_not_required_zero': True,
    'no_held_out_or_closed_loop_success_claim': True,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def source_hashes():
    return prior.source_hashes() | inspection.source_hashes() | {
        Path(__file__).name: sha(__file__), Path(policy.__file__).name: sha(policy.__file__)}


def bits_equal(a, b):
    """Equality including dtype, signed zeros and every frozen tensor bit."""
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def element_bits(a, b):
    require(a.dtype == b.dtype == torch.float32 and a.shape == b.shape, 'Expected matching float32 action tensors')
    return a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)


def load_histories(directories, baseline, baseline_hash):
    require([Path(d).name for d in directories] == list(inspection.RUN_NAMES), 'Exactly the five reviewed histories in fixed order are required')
    episodes, supplements = [], []
    for i, directory in enumerate(directories):
        run = Path(directory)
        data = (mixture_reader if i == 0 else reader).load_run(run, baseline, baseline_hash, FAMILY_SHA)
        data['consumer_schema'] = 'qualified_v3_mixture_with_v7_supplement' if i == 0 else reader.SCHEMA
        supplements.append(inspection.supplement_mixture(run, data) if i == 0 else data['audit'])
        episodes.append(data)
    names, left = episodes[0]['joint_names'], episodes[0]['left_indices']
    require(baseline['metadata']['joint_names'] == names and baseline['metadata']['output_indices'] == left,
            'Baseline/dataset joint order differs')
    require(all(d['joint_names'] == names and d['left_indices'] == left and d['physics_family_sha256'] == FAMILY_SHA
                for d in episodes), 'History joint order or physics family differs')
    return episodes, supplements


def build_model(baseline, episodes, hidden_size):
    values = np.concatenate([d['x'][d['mask']] for d in episodes])
    names = [episodes[0]['joint_names'][i] for i in episodes[0]['left_indices']]
    return policy.ZeroGatedResidualPolicy(baseline, values.mean(0), np.maximum(values.std(0), .001),
                                         policy.candidate_bounds(names), hidden_size)


def verify_device_definition(episodes, model, device):
    """No optimizer: ensure GPU math does not silently redefine exact-zero labels."""
    started=time.monotonic();other=deepcopy(model).to(device).eval();reports=[]
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for data in episodes:
            x,previous,labels,_=frozen.tensor_episode(data,'cpu')
            a=model.baseline_targets(x,previous)
            b=other.baseline_targets(x.to(device),previous.to(device))
            ap=policy.project_left_targets(a,previous,data['parameters'],data['left_indices'],data['projector'])
            bp=policy.project_left_targets(b,previous.to(device),data['parameters'],data['left_indices'],data['projector'])
            ac=policy.classify_correction_labels(labels,ap)
            bc=policy.classify_correction_labels(labels.to(device),bp)
            differences={k:int((ac[k]!=bc[k].cpu()).sum()) for k in ('correction_target','classification_mask','exact_zero','ambiguous')}
            reports.append({'episode_id':data['episode_id'],'baseline_proposal_different_float32_elements':int((~element_bits(a,b.cpu())).sum()),
                'baseline_projected_different_float32_elements':int((~element_bits(ap,bp.cpu())).sum()),
                'baseline_projected_max_abs_difference_rad':float((ap.double()-bp.cpu().double()).abs().max()),
                'classification_differences':differences})
            require(not any(differences.values()),'CPU/execution and training-device classification definitions differ: '+str(reports[-1]))
    memory=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
    del other
    return {'device':str(device),'elapsed_s':time.monotonic()-started,'cuda_peak_allocated_bytes':memory,
            'optimizer_created':False,'training_started':False,'all_classification_arrays_bitwise_identical':True,'runs':reports}


def dataset_entry(data, supplement):
    return {'evidence': data['evidence'], 'consumer_schema': data['consumer_schema'], 'supplemental_audit': supplement,
            'original_result_flags': {k: data['result'].get(k) for k in RESULT_FLAGS},
            'eligible_transition_steps': np.flatnonzero(data['mask']).tolist(), 'excluded_records': data['excluded_labels'],
            'teacher_steps': sum(bool(ok) and s == 'teacher' for ok, s in zip(data['mask'], data['executed_sources'])),
            'student_steps': sum(bool(ok) and s == 'student' for ok, s in zip(data['mask'], data['executed_sources']))}


def validate_qualification(text, expected_sha, episodes, supplements, model, baseline_hash):
    require(digest_bytes(text.encode('utf-8')) == expected_sha, 'Qualification bytes SHA mismatch')
    q = json.loads(text)
    require(q['schema'] == QUAL_SCHEMA, 'Wrong qualification schema')
    for key in ('training_authorized', 'eligible_for_rl_episode_training', 'pure_student_episode_success_claim',
                'independent_generalization_claim', 'original_result_files_modified'):
        require(q[key] is False, 'Original qualification False flag changed: ' + key)
    for key in ('requires_root_review_before_training', 'candidate_bc_transitions_passed_integrity',
                'eligible_under_declared_v7_candidate_bounds', 'zero_class_definition_approved'):
        require(q[key] is True, 'Qualification is not ready: ' + key)
    require(q['source_sha256'] == inspection.source_hashes(), 'Qualification inspection source differs')
    require(q['physics_family_sha256'] == FAMILY_SHA and q['frozen_baseline_checkpoint_sha256'] == baseline_hash == BASELINE_SHA,
            'Unreviewed physics family or frozen baseline')
    require(q['current_candidate_gate'] == policy.PUBLIC_GATE_CONTRACT and q['classification_contract'] == policy.CLASSIFICATION_CONTRACT,
            'Qualification public/classification gate differs')
    names = [episodes[0]['joint_names'][i] for i in episodes[0]['left_indices']]
    require(q['current_candidate_bounds_by_name'] == dict(zip(names, model.residual_bound.tolist())), 'Qualification bounds differ')
    require(q['datasets'] == [dataset_entry(d, s) for d, s in zip(episodes, supplements)], 'Qualification dataset/audit/flags changed')
    require(q['original_projection_source_sha256'] == episodes[0]['parameters']['source_sha256'], 'Projection source differs')
    require(all(episodes[0]['collection_source_sha256'][k] == v for k, v in q['original_gate_source_sha256'].items()), 'Original gate source differs')
    report = prior.inspect(episodes, model)
    prior.assert_reachable(report)
    require(q['active_max_unreachable_distance_rad'] == report['active_max_unreachable_distance_rad'] == 0., 'Active labels unreachable')
    classes = []
    with torch.no_grad():
        for name, data in zip(inspection.RUN_NAMES, episodes):
            x, previous, labels, _ = frozen.tensor_episode(data, 'cpu')
            base = policy.project_left_targets(model.baseline_targets(x, previous), previous,
                                                data['parameters'], data['left_indices'], data['projector'])
            summary = inspection.classification_summary(data['labels'], base[0].numpy(), data['mask'], model.residual_active(x)[0].numpy())
            require(q['classification_by_dataset'][name] == summary, 'Qualified classification bytes/counts differ: ' + name)
            classes.append(policy.classify_correction_labels(labels, base))
    return q, report, classes, prior.prove_boundary(episodes)


def training_config(args):
    return {'epochs': args.epochs, 'learning_rate': args.learning_rate, 'hidden_size': args.hidden_size,
            'seed': args.seed, 'regression_weight_rad2': args.regression_weight,
            'binary_cross_entropy_weight': args.bce_weight, 'weight_decay': 1e-5, 'gradient_clip_norm': 1.,
            'device': args.device, 'episode_shuffle_only': True, 'gru_reset_each_episode': True,
            'candidate_selection_every_epochs': args.candidate_interval,
            'regression_is_ungated_original_float32_projected_all_raw_labels': True,
            'ambiguous_excluded_only_from_classification': True}


def validate_authorization(text, expected_sha, qualification_sha, config):
    require(digest_bytes(text.encode('utf-8')) == expected_sha, 'Root authorization SHA differs')
    auth = json.loads(text)
    expected = {'schema': AUTH_SCHEMA, 'approved_for_development_bc_training': True,
                'qualification_sha256': qualification_sha, 'source_sha256': source_hashes(),
                'training_config': config, 'selection_contract': SELECTION_CONTRACT,
                'eligible_for_rl_episode_training': False, 'independent_generalization_claim': False}
    require(auth == expected, 'Root authorization does not exactly cover this code, data and training configuration')
    return auth


def native_left(proposed, previous, data):
    scales = torch.as_tensor(data['parameters']['scales'], dtype=torch.float32, device=proposed.device).reshape(-1)[data['left_indices']]
    return (proposed - previous) / scales


def boundary_record(step, proposed, base, previous, applied, projected_base, labels, data):
    same_proposal = element_bits(proposed, base)
    same_native = element_bits(native_left(proposed, previous, data), native_left(base, previous, data))
    same_projected = element_bits(applied, projected_base)
    return {'step': step, 'proposal_all28_bitwise_baseline': bool(same_proposal.all()),
            'native_action_all28_bitwise_baseline': bool(same_native.all()),
            'projected_all28_bitwise_baseline': bool(same_projected.all()),
            'expert_projected_all28_bitwise_baseline': bool(element_bits(labels, projected_base).all()),
            'proposal_different_joint_indices': torch.nonzero(~same_proposal.reshape(28)).flatten().tolist(),
            'label_minus_baseline_rad': (labels.double() - projected_base.double()).reshape(28).cpu().tolist()}


def selection_key(metrics):
    return (0 if metrics['step499_reviewed_zero_histories_bitwise_baseline'] else 1,
            metrics['strict_zero_gate_false_positive_count'] + metrics['certain_nonzero_gate_false_negative_count'],
            metrics['hard_decoded_projected_fit']['rmse_rad'])


def assess(episodes, model, device='cpu', *, streaming=False, boundary_only=False):
    """Actual recorded states, never predicted/expert states; reset per episode."""
    model.eval()
    errors, ungated_errors, baseline_errors = [], [], []
    fp = fn = certain_count = zero_count = ambiguous_count = 0
    runs = []
    with torch.no_grad():
        for data in episodes:
            x, previous, labels, mask = frozen.tensor_episode(data, device)
            if boundary_only:
                require(streaming, 'Boundary-only proof must stream the complete prefix')
                x, previous, labels, mask = (v[:, :500] for v in (x, previous, labels, mask))
            if streaming:
                h = None; out = []; diagnostics = []
                for step in range(x.shape[1]):
                    y, h, _, diag = model.forward_with_diagnostics(x[:, step:step+1], previous[:, step:step+1], h)
                    out.append(y); diagnostics.append(diag)
                proposed = torch.cat(out, dim=1)
                diag = {k: torch.cat([d[k] for d in diagnostics], dim=1) for k in diagnostics[0]}
            else:
                proposed, _, _, diag = model.forward_with_diagnostics(x, previous, None)
            project = lambda y: policy.project_left_targets(y, previous, data['parameters'], data['left_indices'], data['projector'])
            applied, base, ungated = project(proposed), project(diag['baseline_proposal']), project(diag['ungated_proposal'])
            classes = policy.classify_correction_labels(labels, base)
            eligible = mask.unsqueeze(-1).expand_as(labels)
            active = diag['public_phase_active'] & eligible
            predicted = diag['predicted_correction_class']
            fp += int((predicted & classes['exact_zero'] & active).sum())
            fn += int((~predicted & classes['correction_target'] & active).sum())
            zero_count += int((classes['exact_zero'] & active).sum())
            certain_count += int((classes['correction_target'] & active).sum())
            ambiguous_count += int((classes['ambiguous'] & active).sum())
            errors.append((applied.double()-labels.double())[mask].cpu().numpy())
            ungated_errors.append((ungated.double()-labels.double())[mask].cpu().numpy())
            baseline_errors.append((base.double()-labels.double())[mask].cpu().numpy())
            require(x.shape[1] > 499 and bool(mask[0, 499]), 'Known real boundary step499 absent or ineligible')
            boundary = boundary_record(499, proposed[0,499], diag['baseline_proposal'][0,499], previous[0,499],
                                       applied[0,499], base[0,499], labels[0,499], data)
            changed = ~element_bits(proposed, diag['baseline_proposal']).all(-1)
            indices = torch.nonzero(changed[0] & mask[0]).flatten()
            first = None
            if len(indices):
                k = int(indices[0])
                first = boundary_record(k, proposed[0,k], diag['baseline_proposal'][0,k], previous[0,k],
                                        applied[0,k], base[0,k], labels[0,k], data)
                first.update(phase=PHASES[int(x[0,k,185:197].argmax())], progress=float(x[0,k,197]),
                             predicted_correction_joint_indices=torch.nonzero(predicted[0,k]).flatten().tolist(),
                             expert_certain_correction_joint_indices=torch.nonzero(classes['correction_target'][0,k]).flatten().tolist(),
                             expert_strict_zero_joint_indices=torch.nonzero(classes['exact_zero'][0,k]).flatten().tolist(),
                             expert_ambiguous_joint_indices=torch.nonzero(classes['ambiguous'][0,k]).flatten().tolist(),
                             projected_prediction_rad=applied[0,k].cpu().tolist(), expert_projected_label_rad=labels[0,k].cpu().tolist())
            required=Path(data['evidence']['directory']).name in ZERO_BOUNDARY_RUNS
            if required:
                require(boundary['expert_projected_all28_bitwise_baseline'],'Reviewed zero-boundary expert is not bitwise baseline')
            runs.append({'episode_id': data['episode_id'], 'frames_processed': int(x.shape[1]),'step499_zero_regression_required':required,
                         'step499': boundary, 'first_nonbaseline_proposal': first})
    def summary(parts):
        arr = np.concatenate(parts)
        return {'rmse_rad': float(np.sqrt(np.mean(arr**2))), 'max_abs_rad': float(np.max(np.abs(arr))),
                'rmse_by_joint_rad': np.sqrt(np.mean(arr**2, axis=0)).tolist()}
    boundary_runs=[r for r in runs if r['step499_zero_regression_required']]
    require(bool(boundary_runs),'No reviewed exact-zero boundary history supplied')
    boundary_pass = all(all(r['step499'][k] for k in ('proposal_all28_bitwise_baseline', 'native_action_all28_bitwise_baseline',
                                                     'projected_all28_bitwise_baseline')) for r in boundary_runs)
    return {'hard_decoded_projected_fit': summary(errors), 'ungated_projected_fit': summary(ungated_errors),
            'frozen_baseline_projected_fit': summary(baseline_errors),
            'strict_zero_gate_false_positive_count': fp, 'certain_nonzero_gate_false_negative_count': fn,
            'public_active_strict_zero_elements': zero_count, 'public_active_certain_nonzero_elements': certain_count,
            'public_active_ambiguous_elements': ambiguous_count, 'step499_reviewed_zero_histories_bitwise_baseline': boundary_pass,
            'runs': runs, 'inference': 'sequential_one_frame_hidden_state' if streaming else 'complete_causal_history_batch',
            'actual_recorded_history': True, 'teacher_labels_used_as_actor_input': False,
            'closed_loop_or_generalization_success_claim': False}


def checkpoint_metadata(model, episodes, q, qtext, qsha, authtext, authsha, config, boundary):
    return {'schema': policy.SCHEMA, 'feature_schema': policy.FEATURE_SCHEMA,
            'physics_family_sha256': FAMILY_SHA, 'frozen_baseline_checkpoint_sha256': BASELINE_SHA,
            'joint_names': episodes[0]['joint_names'], 'output_indices': episodes[0]['left_indices'],
            'source_sha256': source_hashes(), 'dataset_evidence': [d['evidence'] for d in episodes],
            'dataset_consumers': [d['consumer_schema'] for d in episodes],
            'qualification_text': qtext, 'qualification_sha256': qsha,
            'root_authorization_text': authtext, 'root_authorization_sha256': authsha,
            'training_config': config, 'selection_contract': SELECTION_CONTRACT,
            'gate_contract': policy.PUBLIC_GATE_CONTRACT, 'zero_gate_contract': policy.ZERO_GATE_CONTRACT,
            'classification_contract': policy.CLASSIFICATION_CONTRACT, 'bound_contract': q['candidate_bound_contract'],
            'residual_bound_rad': model.residual_bound.tolist(), 'hidden_size': model.hidden_size,
            'normalization_mean': model.mean.tolist(), 'normalization_std': model.std.tolist(),
            'boundary_proof': boundary, 'actor_inputs_are_privileged_actual_state_not_vision': True,
            'eligible_for_rl_episode_training': False, 'independent_generalization_demonstrated': False,
            'closed_loop_success_demonstrated': False, 'right28_learned': False}


def save_candidate(path, model, metadata, epoch, optimizer_steps, metrics, streamed_boundary):
    require(epoch >= 1 and optimizer_steps >= 1, 'Untrained baseline cannot be saved as a learned candidate')
    require(streamed_boundary['inference'] == 'sequential_one_frame_hidden_state', 'Candidate missing streamed boundary evidence')
    torch.save({'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'metadata': metadata | {'best_epoch': epoch, 'optimizer_steps': optimizer_steps,
                    'selection_metrics': metrics, 'cpu_streamed_boundary': streamed_boundary,
                    'offline_candidate_qualified': False}}, path)


def verify_metadata(metadata, baseline, baseline_hash, directories, expected_family):
    require(metadata['schema'] == policy.SCHEMA and metadata['feature_schema'] == policy.FEATURE_SCHEMA, 'V7 checkpoint schema differs')
    require(metadata['source_sha256'] == source_hashes(), 'V7 training/reader/inference source changed')
    require(metadata['physics_family_sha256'] == expected_family == FAMILY_SHA, 'Wrong physical family')
    require(metadata['frozen_baseline_checkpoint_sha256'] == baseline_hash == BASELINE_SHA, 'Wrong frozen baseline')
    for key in ('joint_names','output_indices'):
        require(metadata[key] == baseline['metadata'][key], 'Checkpoint joint order changed: '+key)
    for key, expected in [('gate_contract', policy.PUBLIC_GATE_CONTRACT), ('zero_gate_contract', policy.ZERO_GATE_CONTRACT),
                          ('classification_contract', policy.CLASSIFICATION_CONTRACT), ('selection_contract', SELECTION_CONTRACT)]:
        require(metadata[key] == expected, 'Checkpoint contract changed: '+key)
    require(metadata['best_epoch'] >= 1 and metadata['optimizer_steps'] >= 1, 'Untrained checkpoint')
    qtext=metadata['qualification_text'];qsha=metadata['qualification_sha256']
    require(digest_bytes(qtext.encode('utf-8')) == qsha, 'Embedded qualification altered')
    q=json.loads(qtext)
    require(q['schema'] == QUAL_SCHEMA and q['training_authorized'] is False and q['source_sha256']==inspection.source_hashes(), 'Embedded original qualification changed')
    require(q['classification_contract']==policy.CLASSIFICATION_CONTRACT and q['physics_family_sha256']==FAMILY_SHA
            and q['frozen_baseline_checkpoint_sha256']==BASELINE_SHA, 'Embedded qualification contract differs')
    require(metadata['bound_contract']==q['candidate_bound_contract'], 'Checkpoint bound provenance differs')
    validate_authorization(metadata['root_authorization_text'], metadata['root_authorization_sha256'], qsha, metadata['training_config'])
    require([Path(d).name for d in directories] == list(inspection.RUN_NAMES), 'All five reviewed datasets required')
    require(len(metadata['dataset_evidence'])==len(q['datasets'])==len(directories)==5, 'Five dataset evidence records required')
    for i,(directory,old,qd) in enumerate(zip(directories,metadata['dataset_evidence'],q['datasets'])):
        require(old == qd['evidence'] and metadata['dataset_consumers'][i]==qd['consumer_schema'], 'Checkpoint qualification dataset differs')
        (mixture_reader if i==0 else reader).verify_saved_datasets({'dataset_evidence':[old], 'physics_family_sha256':FAMILY_SHA},[directory])
    names=[metadata['joint_names'][i] for i in metadata['output_indices']]
    bounds=torch.tensor(policy.candidate_bounds(names),dtype=torch.float32)
    require(bits_equal(bounds,torch.tensor(metadata['residual_bound_rad'],dtype=torch.float32)), 'Checkpoint named residual bounds changed')
    require(q['current_candidate_bounds_by_name']==dict(zip(names,bounds.tolist())), 'Qualification named bounds differ')


def load_checkpoint(path, baseline_path, dataset_directories, *, expected_physics_family,
                    expected_checkpoint_sha256, qualification_path, expected_qualification_sha256,
                    authorization_path, expected_authorization_sha256, require_offline_qualified=True):
    require(sha(path)==expected_checkpoint_sha256, 'V7 checkpoint bytes SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True);metadata=saved['metadata'];state=saved['model']
    for path_arg, digest_arg, text_key, hash_key in (
        (qualification_path, expected_qualification_sha256, 'qualification_text', 'qualification_sha256'),
        (authorization_path, expected_authorization_sha256, 'root_authorization_text', 'root_authorization_sha256')):
        raw=Path(path_arg).read_bytes()
        require(digest_bytes(raw)==digest_arg==metadata[hash_key] and raw.decode('utf-8')==metadata[text_key],
                'External reviewed evidence differs: '+hash_key)
    baseline,baseline_hash=frozen.load_baseline(baseline_path)
    verify_metadata(metadata,baseline,baseline_hash,dataset_directories,expected_physics_family)
    if require_offline_qualified:
        require(metadata['offline_candidate_qualified'] is True, 'Candidate failed declared offline selection qualification')
    model=policy.ZeroGatedResidualPolicy(baseline,metadata['normalization_mean'],metadata['normalization_std'],
        metadata['residual_bound_rad'],metadata['hidden_size'],metadata['gate_contract'],metadata['zero_gate_contract'])
    original=model.state_dict()
    for key in ('mean','std','residual_bound','residual_phase_active','squeeze_active_progress',
                'correction_logit_threshold','training_class_confidence_threshold_rad'):
        require(bits_equal(original[key],state[key]), 'Checkpoint static tensor differs bitwise: '+key)
    for key,value in baseline['model'].items():
        require(bits_equal(value,state['baseline.'+key]), 'Frozen baseline tensor differs bitwise: '+key)
    model.load_state_dict(state,strict=True)
    return model.eval(),metadata


def train(model, episodes, classes, args, metadata):
    device=args.device;model.to(device)
    initial=assess(episodes,model,device)
    dump(args.output/'initial_untrained_baseline_metrics.json',initial)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=1e-5)
    rng=random.Random(args.seed);best=None;steps=0;started=time.monotonic()
    with (args.output/'training.jsonl').open('w',encoding='utf-8',newline='\n') as journal:
        for epoch in range(1,args.epochs+1):
            model.train();order=list(range(len(episodes)));rng.shuffle(order);loss_rows=[]
            for i in order:
                data=episodes[i];x,previous,labels,mask=frozen.tensor_episode(data,device)
                optimizer.zero_grad(set_to_none=True)
                proposed,_,_,diag=model.forward_with_diagnostics(x,previous,None)
                c=classes[i]
                losses=policy.supervision_losses(proposed,diag,previous,labels,mask,
                    c['correction_target'].to(device),c['classification_mask'].to(device),data['parameters'],data['left_indices'],data['projector'])
                loss=args.regression_weight*losses['regression_mse_rad2']+args.bce_weight*losses['gate_binary_cross_entropy']
                require(bool(torch.isfinite(loss)), 'Nonfinite training objective')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();steps+=1
                loss_rows.append({'episode_id':data['episode_id'],'objective':float(loss.detach()),
                    **{k:float(losses[k].detach()) for k in ('regression_mse_rad2','gate_binary_cross_entropy')}})
            metrics=assess(episodes,model,device);score=selection_key(metrics)
            selected=False
            consider=epoch==1 or epoch%args.candidate_interval==0 or epoch==args.epochs
            if consider and (best is None or score<best):
                cpu_model=deepcopy(model).cpu()
                streamed=assess(episodes,cpu_model,'cpu',streaming=True,boundary_only=True)
                metrics['step499_reviewed_zero_histories_bitwise_baseline']=streamed['step499_reviewed_zero_histories_bitwise_baseline']
                score=selection_key(metrics)
                if best is None or score<best:
                    best=score;selected=True
                    save_candidate(args.output/'model.pt',model,metadata,epoch,steps,metrics,streamed)
                del cpu_model
            row={'epoch':epoch,'optimizer_steps':steps,'episode_losses':loss_rows,'metrics':metrics,
                 'selected':selected,'selection_key':list(score),'elapsed_s':time.monotonic()-started}
            journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
            if epoch==1 or epoch%10==0:print('V7_DEVELOPMENT_FIT',json.dumps({'epoch':epoch,'selection_key':score,'selected':selected,
                'hard_rmse':metrics['hard_decoded_projected_fit']['rmse_rad'],'gate_fp':metrics['strict_zero_gate_false_positive_count'],
                'gate_fn':metrics['certain_nonzero_gate_false_negative_count'],'elapsed_s':row['elapsed_s']}),flush=True)
    require(metadata['source_sha256']==source_hashes(),'Sources changed during optimization')
    saved=torch.load(args.output/'model.pt',map_location='cpu',weights_only=True)
    model.cpu().load_state_dict(saved['model'])
    final=assess(episodes,model,'cpu',streaming=True)
    initial_errors=initial['strict_zero_gate_false_positive_count']+initial['certain_nonzero_gate_false_negative_count']
    final_errors=final['strict_zero_gate_false_positive_count']+final['certain_nonzero_gate_false_negative_count']
    qualified=(final['step499_reviewed_zero_histories_bitwise_baseline'] and final_errors<initial_errors and
               final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'])
    saved['metadata'].update(offline_candidate_qualified=qualified,final_cpu_sequential_metrics=final,
        qualification_basis='trained candidate: step499 bitwise baseline, fewer class errors and lower decoded RMSE than initial baseline; offline development only')
    torch.save(saved,args.output/'model.pt')
    return {'offline_candidate_qualified':qualified,'final_cpu_sequential_metrics':final,'best_epoch':saved['metadata']['best_epoch'],
        'checkpoint_sha256':sha(args.output/'model.pt'),'closed_loop_success_demonstrated':False,
        'independent_generalization_demonstrated':False,'failure_if_unqualified':'candidate_selection_contract_not_met' if not qualified else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train',action='store_true')
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--qualification-sha256',required=True)
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256')
    p.add_argument('--epochs',type=int,default=2000);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--hidden-size',type=int,default=64);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--regression-weight',type=float,default=1e6);p.add_argument('--bce-weight',type=float,default=1.)
    p.add_argument('--candidate-interval',type=int,default=100)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    require(args.epochs>=1 and args.hidden_size>=1 and args.candidate_interval>=1 and all(math.isfinite(v) and v>0 for v in
        (args.learning_rate,args.regression_weight,args.bce_weight)),'Positive finite training parameters required')
    torch.set_num_threads(1)
    sources=source_hashes();baseline,basehash=frozen.load_baseline(args.baseline)
    require(basehash==BASELINE_SHA,'Unreviewed baseline checkpoint')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    episodes,supplements=load_histories(args.run,baseline,basehash)
    model=build_model(baseline,episodes,args.hidden_size)
    qtext=args.qualification.read_bytes().decode('utf-8')
    q,reachability,classes,boundary=validate_qualification(qtext,args.qualification_sha256,episodes,supplements,model,basehash)
    device_proof=verify_device_definition(episodes,model,args.device)
    timer=time.monotonic()
    initial_boundary=assess(episodes,model,'cpu',streaming=True,boundary_only=True)
    initial_boundary['elapsed_s']=time.monotonic()-timer
    initial_boundary['untrained_reference_only_not_candidate']=True
    require(sources==source_hashes(),'Source changed during qualification')
    config=training_config(args)
    auth_template={'schema':AUTH_SCHEMA,'approved_for_development_bc_training':True,'qualification_sha256':args.qualification_sha256,
        'source_sha256':sources,'training_config':config,'selection_contract':SELECTION_CONTRACT,
        'eligible_for_rl_episode_training':False,'independent_generalization_claim':False}
    authtext=authsha=None
    if args.train:
        require(args.authorization is not None and args.authorization_sha256,'Training requires an explicit root authorization file and SHA')
        authtext=args.authorization.read_bytes().decode('utf-8');authsha=args.authorization_sha256
        validate_authorization(authtext,authsha,args.qualification_sha256,config)
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'training_authorization_template.json',auth_template)
    dump(args.output/'training_device_definition.json',device_proof)
    dump(args.output/'initial_boundary_timing.json',initial_boundary)
    dump(args.output/'verified_inputs.json',{'source_sha256':sources,'qualification_sha256':args.qualification_sha256,
        'bound_and_mask_inspection':reachability,'boundary_proof':boundary,'training_started':bool(args.train),
        'original_candidate_training_authorized':q['training_authorized'],'all_original_continuous_labels_retained':True})
    print('V7_INPUTS_VERIFIED',json.dumps({'eligible_steps':reachability['eligible_steps'],'source_sha256':sources,
        'authorization_template':str(args.output/'training_authorization_template.json')}),flush=True)
    if args.train:
        metadata=checkpoint_metadata(model,episodes,q,qtext,args.qualification_sha256,authtext,authsha,config,boundary)
        result=train(model,episodes,classes,args,metadata);dump(args.output/'result.json',result)
        print('V7_COMPLETE',json.dumps({k:v for k,v in result.items()if k!='final_cpu_sequential_metrics'}),flush=True)


if __name__=='__main__':
    main()
