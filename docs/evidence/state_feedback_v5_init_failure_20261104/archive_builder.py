"""Preserve the zero-control-step V5 launch failure and launch-only correction."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path('/home/brainco/rl_cube/github_project')
CODE=ROOT/'revotron_cube_rl'
WORK=ROOT/'work/teacher_optimization_20261103'
RUN=CODE/'runs/left_state_residual_v5_long_eval_9501_20261104'
OUT=ROOT/'docs/evidence/state_feedback_v5_init_failure_20261104'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
assert (RUN/'trajectory.jsonl').stat().st_size==0
assert read(RUN/'error.json')['message']=='Baseline runtime asset contract differs from source'
OUT.mkdir(parents=True,exist_ok=False)
entries=[]
def preserve(source,relative):
    before=sha(source);target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb')as a,target.open('xb')as b:shutil.copyfileobj(a,b)
    assert before==sha(target)==sha(source)
    entries.append({'source':str(source.relative_to(ROOT)),'archive':str(relative),'bytes':target.stat().st_size,'sha256':before})
for source in sorted(RUN.rglob('*')):
    if source.is_file():preserve(source,Path('failed_run')/source.relative_to(RUN))
source_contract=CODE/'runs/center_j56_dev_v3_20261103/asset_contract.json'
preserve(source_contract,Path('diagnosis/source_asset_contract.json'))
generated_contract=CODE/'asset_contract.json'
expected_failed_base_hash='1217705118af80f13ab54433011674fa7b5034d62fb5cea8b0a2f76a3c3c96c9'
assert sha(generated_contract)==expected_failed_base_hash,'Global generated contract changed after failure; do not attribute a newer contract to this failure'
preserve(generated_contract,Path('diagnosis/failed_base_generated_asset_contract.json'))
for name in ['launch_v5_closed_loop_eval_20261104.sh','launch_v4_closed_loop_eval_20261104.sh',
             'launch_v5_closed_loop_eval_v2_20261104.sh','launch_v4_closed_loop_eval_v2_20261104.sh']:
    preserve(WORK/name,Path('launch')/name)
preserve(Path(__file__),Path('archive_builder.py'))
a=read(source_contract);b=read(generated_contract)
common_differences=[key for key in a.keys()&b.keys()if a[key]!=b[key]]
assert set(common_differences)=={'asset_path_overrides','cube_mode'}
assert set(a)-set(b)=={'dynamic_cube'}
for name in ['launch_v5_closed_loop_eval_v2_20261104.sh','launch_v4_closed_loop_eval_v2_20261104.sh']:
    oldname=name.replace('_v2_20261104','_20261104')
    old=(WORK/oldname).read_text();new=(WORK/name).read_text()
    assert old.split('exec python ',1)[1].strip()+'_v2'==new.split('exec python ',1)[1].strip()
diagnosis={
    'schema':'v5_launch_environment_asset_contract_failure_v1',
    'run':RUN.name,'completed_policy_control_steps':0,'trajectory_jsonl_bytes':0,
    'error':read(RUN/'error.json'),
    'underlying_constructor_physics_steps_claim':None,
    'zero_step_scope':'Zero evaluation actions. Frozen base constructor performs warm-up _physics calls; this is not a zero-total-physics claim.',
    'cause':'Launcher inherited activate.sh old-root REVOTRON_USD_PATH and ISAAC_MUJOCO_CUBE_ROOT instead of the source wrist-mesh entry and github_project asset root.',
    'observed_activation_values':{'REVOTRON_USD_PATH':'/home/brainco/rl_cube/revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd','ISAAC_MUJOCO_CUBE_ROOT':'/home/brainco/rl_cube/isaac_mujoco_cube'},
    'source_asset_path_overrides':a['asset_path_overrides'],
    'failed_base_generated_asset_path_overrides':b['asset_path_overrides'],
    'base_contract_common_field_differences':sorted(common_differences),
    'base_contract_disk_timing':'environment.py writes asset_contract.json before DynamicPickupEnv replaces cube_mode and adds dynamic_cube. Those two differences do not describe the final in-memory contract.',
    'in_memory_full_failed_contract_archived':False,
    'correction':'New launch_v2 explicitly exports the two source asset paths and validates the wrist entry, referenced 56DOF layer, and cube USD hashes before unchanged evaluation startup.',
    'control_source_or_model_changed':False,'contract_assertion_weakened':False,
    'v2_launch_prefix_only_checks_passed':True,'v2_simulator_started_by_archive_author':False,
    'v2_output_run':'left_state_residual_v5_long_eval_9501_20261104_v2',
    'historical_failure_result_modified':False,'evaluation_success_claim':False,
}
(OUT/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
manifest={'schema':'exact_copy_failed_startup_archive_v1','source_files':entries,
          'generated_files_sha256':{'diagnosis.json':sha(OUT/'diagnosis.json')},
          'all_source_files_unchanged':True,'all_archived_bytes_match_source':True}
(OUT/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(OUT),'manifest_sha256':sha(OUT/'archive_manifest.json'),'source_files':len(entries)}))
