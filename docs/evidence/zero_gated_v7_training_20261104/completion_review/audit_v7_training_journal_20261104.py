"""Completed-run journal/selection audit, without repeating the peer's inference."""
from pathlib import Path
import argparse,hashlib,json,math
import torch
import train_zero_gated_residual_v7 as trainer

ROOT=Path('/home/brainco/rl_cube/github_project');WORK=ROOT/'work/teacher_optimization_20261103'
p=argparse.ArgumentParser()
p.add_argument('--run-name',default='teacher_zero_gated_residual_v7_20261104')
p.add_argument('--authorization-name',default='v7_root_training_authorization_20261104.json')
p.add_argument('--output-name',default='v7_completed_training_journal_audit_20261104.json')
a=p.parse_args()
for value in (a.run_name,a.authorization_name,a.output_name):
    if Path(value).name!=value:raise ValueError('Only basename paths accepted')
RUN=ROOT/'revotron_cube_rl/runs'/a.run_name
OUTPUT=WORK/a.output_name

def read(path):
    return json.loads(path.read_text(),parse_constant=lambda value:(_ for _ in()).throw(ValueError('Nonfinite JSON '+value)))
def sha(path):return trainer.sha(path)
def require(condition,message):
    if not condition:raise ValueError(message)

result=read(RUN/'result.json');cp=RUN/'model.pt'
require(sha(cp)==result['checkpoint_sha256'],'Final result/checkpoint SHA differs')
saved=torch.load(cp,map_location='cpu',weights_only=True);m=saved['metadata']
require(m['source_sha256']==trainer.source_hashes(),'Final trainer source differs')
require(m['final_cpu_sequential_metrics']==result['final_cpu_sequential_metrics'],'Final metadata/result CPU metrics differ')
require(m['offline_candidate_qualified']==result['offline_candidate_qualified'],'Final qualification metadata differs')
require(m['best_epoch']==result['best_epoch']>=1,'Invalid selected epoch')
auth=read(WORK/a.authorization_name)
require(auth['source_sha256']==m['source_sha256'] and auth['training_config']==m['training_config'],'Root-reviewed source/config differs')
require(sha(WORK/a.authorization_name)==m['root_authorization_sha256'],'Root authorization bytes changed')
initial=read(RUN/'initial_untrained_baseline_metrics.json')
config=m['training_config'];epochs=config['epochs'];interval=config['candidate_selection_every_epochs']
count=0;selected=[];best=None;last=None;loss_ranges={key:[float('inf'),-float('inf')]for key in ('regression_mse_rad2','gate_binary_cross_entropy','objective')}
with (RUN/'training.jsonl').open()as journal:
    for line in journal:
        require(line.endswith('\n'),'Incomplete training journal line')
        row=json.loads(line);count+=1;require(row['epoch']==count,'Missing or duplicated epoch')
        require(row['optimizer_steps']==5*count,'Wrong optimizer step count')
        require(len(row['episode_losses'])==5,'An actual history is omitted from an epoch')
        require(len({v['episode_id']for v in row['episode_losses']})==5,'Duplicated history within epoch')
        for values in row['episode_losses']:
            for key,bounds in loss_ranges.items():
                value=values[key];require(math.isfinite(value)and value>=0,'Invalid loss '+key)
                bounds[0]=min(bounds[0],value);bounds[1]=max(bounds[1],value)
            expected=config['regression_weight_rad2']*values['regression_mse_rad2']+config['binary_cross_entropy_weight']*values['gate_binary_cross_entropy']
            require(abs(expected-values['objective'])<1e-10*max(1.,abs(expected)),'Declared objective weights differ')
        require(row['selection_key']==list(trainer.selection_key(row['metrics'])),'Journal selection key differs')
        if row['selected']:
            require(count==1 or count%interval==0 or count==epochs,'Unreviewed candidate epoch selected')
            key=tuple(row['selection_key']);require(best is None or key<best,'Selected candidate did not improve declared ordering')
            best=key;selected.append({'epoch':count,'selection_key':row['selection_key'],'elapsed_s':row['elapsed_s']})
            last=row
require(count==epochs==5000,'Training incomplete')
require(last is not None and last['epoch']==m['best_epoch'] and m['optimizer_steps']==5*last['epoch'],'Stored best is not journal-selected final best')
require(last['metrics']==m['selection_metrics'],'Checkpoint selection metrics differ from journal')
require(m['cpu_streamed_boundary']['inference']=='sequential_one_frame_hidden_state','Missing CPU streamed candidate proof')
final=m['final_cpu_sequential_metrics']
require(final['public_active_strict_zero_elements']==65953 and final['public_active_certain_nonzero_elements']==14997
        and final['public_active_ambiguous_elements']==1090,'Final classification support differs from reviewed qualification')
boundary_runs=[]
for row in final['runs']:
    name=Path(row['episode_id'].split(':')[0]).name
    expected=name in trainer.ZERO_BOUNDARY_RUNS
    require(row['step499_zero_regression_required']==expected,'Wrong zero-boundary scope')
    if expected:
        require(row['step499']['expert_projected_all28_bitwise_baseline'],'Zero-boundary ground truth changed')
        boundary_runs.append(name)
    else:
        require(name==trainer.inspection.RUN_NAMES[1]and not row['step499']['expert_projected_all28_bitwise_baseline'],'V5 recovery supervision was reinterpreted')
require(sorted(boundary_runs)==sorted(trainer.ZERO_BOUNDARY_RUNS),'Missing real zero-boundary history')
initial_error=initial['strict_zero_gate_false_positive_count']+initial['certain_nonzero_gate_false_negative_count']
final_error=final['strict_zero_gate_false_positive_count']+final['certain_nonzero_gate_false_negative_count']
qualified=(final['step499_reviewed_zero_histories_bitwise_baseline']and final_error<initial_error and
           final['hard_decoded_projected_fit']['rmse_rad']<initial['hard_decoded_projected_fit']['rmse_rad'])
require(qualified==result['offline_candidate_qualified'],'Offline result qualification incorrectly computed')
require(result['closed_loop_success_demonstrated']is False and result['independent_generalization_demonstrated']is False,'Overclaimed training result')
value={'schema':'v7_completed_training_journal_independent_consistency_audit_v1','audit_passed':True,
    'training_run':str(RUN),'checkpoint_sha256':sha(cp),'result_sha256':sha(RUN/'result.json'),
    'training_journal_sha256':sha(RUN/'training.jsonl'),'epochs':count,'optimizer_steps_per_epoch':5,
    'selected_candidates':selected,'best_epoch':m['best_epoch'],'loss_ranges':loss_ranges,
    'initial_hard_decoded_fit':initial['hard_decoded_projected_fit'],'final_hard_decoded_fit':final['hard_decoded_projected_fit'],
    'initial_classification_errors':initial_error,'final_strict_zero_false_positives':final['strict_zero_gate_false_positive_count'],
    'final_certain_nonzero_false_negatives':final['certain_nonzero_gate_false_negative_count'],
    'final_cpu_sequential_metrics':final,'offline_candidate_qualified':qualified,
    'independent_inference_repeated_here':False,'inference_review_owner':'training_path_audit',
    'source_files_or_original_results_modified':False,'closed_loop_success_demonstrated':False,
    'independent_generalization_demonstrated':False}
OUTPUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(OUTPUT),'sha256':sha(OUTPUT),'epochs':count,'best_epoch':m['best_epoch'],
    'checkpoint_sha256':sha(cp),'offline_candidate_qualified':qualified,'final_fp':value['final_strict_zero_false_positives'],
    'final_fn':value['final_certain_nonzero_false_negatives'],'hard_fit':value['final_hard_decoded_fit']},indent=2))
