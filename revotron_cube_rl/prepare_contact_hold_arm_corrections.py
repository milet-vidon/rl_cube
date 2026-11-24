"""Prepare review-only, left-arm preshape labels from a failed student rollout.

The full episode remains failed. Only projected teacher requests at nonterminal
student states are exported; these counterfactual commands were not executed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from contact_hold_dataset import sha, verified_json
from joint_safety import max_target_delta, project_target
from phase_contact_policy import features


def prepare(run, out):
    manifest = json.loads((run/'artifact_manifest.json').read_text())
    result, rows, cfg, contract, protocol = [
        verified_json(run, name, manifest) for name in
        ('result.json','trajectory.json','config.json','asset_contract.json','student_evaluation.json')]
    checkpoint_sha = sha(run/'evaluated_best.pt')
    if not (result['kind'] == 'contact_hold_bc_closed_loop_evaluation' and result['eligible_training_episode'] is False
            and result['corrective_labels_qualified_for_training'] is False and not result['execution_interrupted']
            and result['failure'] == 'contact_topology_or_settle_timeout' and rows[-1]['phase'] == 'preshape'
            and result['steps'] == result['student_executed_steps'] == len(rows)
            and result['cube_pose_writes_during_control'] == 0 and len({r['pose_reset_count'] for r in rows}) == 1
            and result['checkpoint_sha256'] == protocol['checkpoint_sha256'] == manifest['sha256']['evaluated_best.pt'] == checkpoint_sha):
        raise ValueError('Expected a complete and identified student preshape timeout')
    names = cfg['joint_names']
    left = [i for i,n in enumerate(names) if n.startswith('left_') or n.endswith('_L')]
    joints = [f'Joint{i}_L' for i in range(1,8)]
    ids = [names.index(n) for n in joints]
    lids = [left.index(i) for i in ids]
    bounds = torch.tensor([[contract['joint_limit_safety']['soft_limits_rad'][n] for n in joints]])
    # Bind these constants to the archived and current original env.step implementation.
    for name in ('environment.py','joint_safety.py'):
        if sha(Path(__file__).with_name(name)) != manifest['sha256'][name]:
            raise ValueError('Projection implementation changed: '+name)
    scales = torch.full((1,7), .025)
    step_limit = torch.minimum(scales,max_target_delta(joints,scales,cfg))
    velocity = torch.full((1,7),3.1416*cfg['arm_velocity_limit_scale'])
    xs, labels, steps = [], [], []
    for i, r in enumerate(rows):
        if i != r['step'] or r['executed_left_source'] != 'student':
            raise ValueError('Missing actual student transition')
        if (r['drop'] or r['self_contact_impulse']>1e-6 or r['robot_table_impulse']>1e-6
                or r['actual_arm_soft_limit_violation'] or r['arm_soft_limit_hit'] or any(r['hard_limit_hit'])):
            raise ValueError('Unsafe trajectory')
        if not i or i == len(rows)-1 or r['phase'] != 'preshape':
            continue
        pre = rows[i-1]
        before = torch.tensor(pre['applied_target'])[None,ids]
        teacher = torch.tensor(r['teacher_left_requested_not_executed'])[None,lids]
        torch.testing.assert_close(teacher,torch.tensor(r['nominal_target'])[None,ids],rtol=0,atol=1e-6)
        def project(request):
            return project_target(before,request,bounds,step_limit,velocity,1./cfg['control_hz'])
        # Replay the ACTUAL student target through this boundary before using it on teacher requests.
        actual, hit = project(torch.tensor(r['student_raw_left_target'])[None,lids])
        torch.testing.assert_close(actual,torch.tensor(r['applied_target'])[None,ids],rtol=0,atol=1e-6)
        target, teacher_hit = project(teacher)
        if bool(teacher_hit.any()) or any(r['soft_limit_hit']):
            continue
        xs.append(features(pre,r['phase'],r['phase_elapsed']))
        labels.append(target[0].tolist())
        steps.append(i)
    if not steps:
        raise ValueError('No reviewable preshape corrections')
    out.mkdir(parents=True,exist_ok=False)
    data = out/'preshape_left_arm.npz'
    np.savez_compressed(data, features=np.stack(xs), projected_teacher_arm_rad=np.asarray(labels,dtype=np.float32),
                        steps=np.asarray(steps), output_names=np.asarray(joints))
    summary = dict(kind='review_only_unexecuted_teacher_arm_corrections', source=str(run.resolve()), seed=result['seed'],
                   checkpoint_sha256=checkpoint_sha, manifest_sha256=sha(run/'artifact_manifest.json'), data_sha256=sha(data),
                   rows=len(steps), first_step=steps[0], last_step=steps[-1], terminal_step_excluded=len(rows)-1,
                   phase='preshape', output_names=joints, original_failure=result['failure'], training_authorized=False,
                   independent_test=False, labels_were_executed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.run,a.output)
