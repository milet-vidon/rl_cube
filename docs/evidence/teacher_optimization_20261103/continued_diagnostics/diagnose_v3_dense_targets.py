"""Read-only source/evaluation comparison; no simulation import or execution."""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
from pxr import Usd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[2] / 'revotron_cube_rl'
source = root / 'runs/center_j56_dev_v3_20261103'
evaluation = root / 'runs/teacher_phase_expert_gpu_v3_dense_eval_9501'
S = json.loads((source / 'trajectory.json').read_text())
E = [json.loads(line) for line in (evaluation / 'trajectory.jsonl').read_text().splitlines()]
names = json.loads((source / 'config.json').read_text())['joint_names']
left = [i for i, name in enumerate(names) if name.startswith('left_') or name.endswith('_L')]
right = [i for i in range(56) if i not in left]
result = {
    'scope': 'read_only_offline_comparison_no_new_physics_run',
    'source_run': str(source), 'evaluation_run': str(evaluation),
    'source_rows': len(S), 'evaluation_rows': len(E),
    'artifact_sha256': {str(p): sha(p) for p in (
        source / 'trajectory.json', source / 'artifact_manifest.json',
        source / 'experiment.json', evaluation / 'trajectory.jsonl',
        evaluation / 'result.json', evaluation / 'source_snapshot/evaluate_phase_expert_v3.py',
    )},
    'phase_mismatches': [{'step': i, 'source': s['phase'], 'eval': e['phase']}
                         for i, (s, e) in enumerate(zip(S, E)) if s['phase'] != e['phase']],
    'evaluation_result': json.loads((evaluation / 'result.json').read_text()),
    'phases': [], 'selected_steps': [],
}
for phase in dict.fromkeys(row['phase'] for row in E):
    indices = [i for i, row in enumerate(E) if row['phase'] == phase]
    delta = np.asarray([E[i]['applied_target'] for i in indices]) - np.asarray([S[i]['applied_target'] for i in indices])
    sample, joint = np.unravel_index(np.argmax(np.abs(delta[:, left])), (len(indices), len(left)))
    q_delta = np.asarray([E[i]['q'] for i in indices]) - np.asarray([S[i]['q'] for i in indices])
    result['phases'].append({
        'phase': phase, 'first_step': indices[0], 'last_step': indices[-1],
        'left_target_rmse_rad': float(np.sqrt(np.mean(delta[:, left] ** 2))),
        'left_target_max_abs_rad': float(np.abs(delta[:, left]).max()),
        'worst_step': indices[sample], 'worst_joint': names[left[joint]],
        'right_target_max_abs_rad': float(np.abs(delta[:, right]).max()),
        'left_measured_q_rmse_rad': float(np.sqrt(np.mean(q_delta[:, left] ** 2))),
        'source_maximum_lift_m': max(S[i]['height_m'] for i in indices),
        'eval_maximum_lift_m': max(E[i]['height_m'] for i in indices),
    })
for i in (0, 1, 100, 300, 360, 400, 450, 480, 500, 520, 540, 560, 580, 600, 620, 650):
    s, e = S[i], E[i]
    difference = np.asarray(e['applied_target']) - s['applied_target']
    result['selected_steps'].append({
        'step': i, 'phase': s['phase'],
        'target_max_abs_rad': float(np.abs(difference).max()),
        'target_worst_joint': names[int(np.argmax(np.abs(difference)))],
        'measured_q_max_abs_difference_rad': float(np.abs(np.asarray(e['q']) - s['q']).max()),
        'cube_position_delta_m': (np.asarray(e['cube_position']) - s['cube_position']).tolist(),
        'source_left_finger_impulses_Ns': s['finger_contact_impulses']['left'],
        'eval_left_finger_impulses_Ns': e['finger_contact_impulses']['left'],
        'source_lift_m': s['height_m'], 'eval_lift_m': e['height_m'],
    })
result['config_equal'] = json.loads((source / 'config.json').read_text()) == json.loads((evaluation / 'config.json').read_text())
result['asset_contract_sha256'] = {label: sha(path / 'asset_contract.json') for label, path in (('source', source), ('evaluation', evaluation))}
experiment = json.loads((source / 'experiment.json').read_text())
evaluation_hashes = result['evaluation_result']['evaluation_source_sha256']
result['source_code_differences'] = {name: {'source': value, 'evaluation': evaluation_hashes.get(name)}
                                     for name, value in experiment['source_sha256'].items()
                                     if value != evaluation_hashes.get(name)}


def physics(path):
    stage = Usd.Stage.Open(str(path))
    output = {}
    for prim in stage.Traverse():
        attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith(('physics:', 'physx'))}
        relationships = {r.GetName(): [str(x) for x in r.GetTargets()] for r in prim.GetRelationships()
                         if r.GetName().startswith('physics:') or r.GetName() == 'material:binding:physics'}
        schemas = [s for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
        if attrs or relationships or schemas:
            output[str(prim.GetPath())] = {'attributes': attrs, 'relationships': relationships,
                                          'physics_schemas': schemas, 'active': prim.IsActive(),
                                          'type': prim.GetTypeName()}
    return output


A = physics(source / 'dynamic_training_scene.usda')
B = physics(evaluation / 'dynamic_training_scene.usda')
result['composed_physics'] = {
    'source_prim_count': len(A), 'evaluation_prim_count': len(B),
    'differences': {key: {'source': A.get(key), 'eval': B.get(key)}
                    for key in sorted(set(A) | set(B)) if A.get(key) != B.get(key)},
    'scope': 'composed physics/physx attributes, physics material bindings, physics schemas, active flag',
}
key_values = {}
for prim in A.values():
    for key, value in prim['attributes'].items():
        if any(token in key.lower() for token in ('gravity', 'offset', 'ccd', 'enabled', 'solver', 'friction')):
            key_values.setdefault(key, set()).add(value)
result['composed_physics']['relevant_attribute_values'] = {key: sorted(value) for key, value in sorted(key_values.items())}
constructors = []
for filename in ('wrist_direction.py', 'elbow_handover.py', 'palm_handover.py',
                 'tilt_regrasp.py', 'opposition_teacher.py', 'opposition_feedback.py'):
    code = (source / filename).read_text()
    for cls in ast.parse(code).body:
        if isinstance(cls, ast.ClassDef) and any(token in cls.name for token in ('Teacher', 'Controller', 'Arrival')):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == '__init__':
                    constructors.append({'file': filename, 'class': cls.name, 'line': fn.lineno,
                                         'source': ast.get_source_segment(code, fn)})
result['teacher_constructor_evidence'] = constructors
result['runtime_audit'] = {
    'constructor_conclusion': 'Teacher constructors only read env state and create local controllers/planners; no gain, target or simulator state writes found.',
    'gain_schedule': {'arms_damping': 8., 'finger_initial_stiffness': 30., 'finger_damping': 1.,
                      'left_finger_hold_stiffness': 20., 'gain_change_step_both': 1553,
                      'right_stiffening_branch': 'set_right_arm_gains(True) requires regrasp=True; absent from center-only source'},
    'audit_subscription_order_both': ['environment', 'FingerContactAudit', 'DetailedSelfContactAudit'],
    'source_render': {'record': True, 'vision': True, 'preview': True,
                      'kit_args': '--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false',
                      'after_reset': 'one overview render after audit construction, then render every third control step'},
    'evaluation_render': {'record': False, 'vision': False, 'preview': False, 'kit_args': ''},
    'render_implementation': 'update_transformations; sim.render; camera.update. No explicit additional physics step, but render timing effects not excluded without controlled replay.',
}
result['conclusions'] = [
    'All common control-step phases and right targets match; no one-step schedule shift found.',
    'Low mean left-target error still contains ready max .00339 rad and squeeze max .00269 rad errors.',
    'Contact state bifurcates during squeeze; at steps500/520 source index/ring have positive impulse while eval has zero; by560 eval has no left finger impulses and never lifts.',
    'Configuration, asset contract and composed physics match; no demonstrated gravity/contactoffset/CCD/activation difference.',
    'Exact reference applied-target replay is required before attributing failure solely to neural approximation; no more NN updates justified yet.',
]
result['next_diagnostic'] = {
    'script': 'evaluate_reference_replay.py', 'sha256': sha(root / 'evaluate_reference_replay.py'),
    'first': 'headless matches learned evaluator; all56 targets from reference through unchanged env.step',
    'if_replay_fails': 'source render mode repeats original camera/render configuration',
    'scope': 'diagnostic only, no learned success claim, all original safety and wrist/center gates retained',
}
destination = Path(__file__).with_name('v3_dense_grasp_failure_diagnosis.json')
destination.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({'output': str(destination), 'sha256': sha(destination),
                  'phase_mismatches': result['phase_mismatches'],
                  'physics_prim_counts': [len(A), len(B)],
                  'physics_differences': result['composed_physics']['differences'],
                  'source_code_difference_names': list(result['source_code_differences'])}))
