"""Compare the frozen student's 90mm trial to archived matched-seed 57mm trials."""
import argparse
import hashlib
import json
from pathlib import Path
from contact_hold_physical_metrics import summarize_hold
from contact_hold_size_face_score import summarize_face
from contact_hold_simple_rotation import summarize_rotation
from contact_hold_grace_worker import CHECKPOINT

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'runs/contact_hold_functional_r15_20261111_1945/from_reset'

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def score(run,mm,seed):
    man=run/'artifact_manifest.json'; m=json.loads(man.read_text())
    def get(n):
        p=run/n
        if digest(p)!=m['sha256'][n]:raise ValueError(f'changed file: {p}')
        return json.loads(p.read_text())
    cfg=get('config.json'); r=get('result.json'); rows=get('trajectory.json')
    if not(m['full_episode'] and r['full_episode'] and r['seed']==seed and not r['execution_interrupted'] and
           r['checkpoint_sha256']==digest(run/'evaluated_best.pt')==CHECKPOINT and r['steps']==len(rows) and
           r['cube_pose_writes_during_control']==0 and all(row['executed_left_source']=='student' for row in rows)):
        raise ValueError(f'Not an intact closed-loop trial: {run}')
    proof=get('cube_asset.json')
    asset=ROOT.parent/'isaac_mujoco_cube/cube_model.usda' if mm==57 else run.parent.parent/'asset90/cube_model.usda'
    if proof['source_sha256']!=digest(asset): raise ValueError('Wrong object asset')
    if len(proof['model']['bodies'])!=27 or proof['model']['runtime_friction']!=[.9,.8]: raise ValueError('Wrong bodies or friction')
    if max(abs(row['height_m']-(row['cube_position'][2]-cfg['table_z']-mm/2000.)) for row in rows)>1e-5:
        raise ValueError('Wrong height reference')
    phase=dict.fromkeys(row['phase'] for row in rows)
    left_contact=[row['step'] for row in rows if row['contact_impulse'][0]>cfg['success']['contact_impulse_min']]
    ready=next((row for row in reversed(rows) if row['phase']=='ready'), None)
    arm=[i for i,n in enumerate(cfg['joint_names']) if n.startswith('Joint')]
    ready_err=max(abs(ready['q'][i]-ready['nominal_target'][i]) for i in arm) if ready else None
    return dict(run=str(run),manifest_sha256=digest(man),checkpoint_sha256=CHECKPOINT,asset_sha256=digest(asset),
                friction=proof['model']['runtime_friction'],steps=len(rows),phases=list(phase),end_reason=r.get('failure'),
                left_contact_frames=len(left_contact),first_left_contact_step=next(iter(left_contact),None),last_ready_arm_error_rad=ready_err,
                ever_commanded_face_push=bool(get('experiment.json').get('push_degrees')),
                first_xyz=rows[0]['cube_position'],first_quaternion_wxyz=rows[0]['cube_quaternion_wxyz'],
                end_table_contact=bool(rows[-1]['cube_table_contact']),original=summarize_hold(rows,cfg),
                practical=summarize_rotation(rows,cfg),single_face=summarize_face(rows,cfg))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path);a=p.parse_args();job=a.job.resolve()
    data=dict(kind='development_only_57_90mm_frozen_round15_from_reset',
              precontact_alignment_probe='approach' in job.name,episodes={})
    for seed in (21611,21612):
        pair={str(mm):score((BASE if mm==57 else job/'from_reset')/f'seed_{seed}',mm,seed) for mm in (57,90)}
        data['episodes'][str(seed)]=pair
    with (job/'size_comparison.json').open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(job/'size_comparison.json')
if __name__=='__main__':main()
