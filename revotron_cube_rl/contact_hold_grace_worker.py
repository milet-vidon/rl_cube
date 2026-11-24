"""Isolated round-15 PhysX diagnostic. The scheduler owns the campaign and GPU locks.

Only the primary_lift lost_primary_grasp debounce changes (10 -> 20 observations).
The deployed student, post-lift teacher, and original simulation/audit harness run unchanged.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import textwrap
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
FIT = RUNS / 'contact_hold_cuda_long_v1/round_015/fit'
CHECKPOINT = '13953a5c8bf4522fa9735b5807b082ae51cdfaac2044c80c30ee2f0839d116c2'
PANEL = FIT.parent / 'validation/panel.json'
HASHES = {
    'opposition_teacher.py': 'a9fcfbdf084b7a34fbc2c0ef9b6e6134df0f517d4d68f8742138a3300b3a79f1',
    'single_face_experiment.py': 'ee88d31643223f486f1836c2670f2c4ed8ec6174c905d6c24b1fb9d53e58c002',
    'contact_hold_student_eval.py': '0849ba2834dd09942d08397a258346a16446595f0dffc37ef870c54be10c558d',
    'tilt_regrasp.py': '8061917aa74b94f7fa753b213a6b9601448a5a35f2c13d147be7a5388225bdc1',
}
PANEL_SHA = 'dbc6ecf51f3ef755eb0745cc8e4a2255d56809c934bd0e68c8d3b0eece98b6cd'
ORIGINAL = "if self.lost>=10:self.failure_reason='lost_primary_grasp';"
TREATMENT = ORIGINAL.replace('>=10', '>=20')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patched_action():
    from opposition_teacher import OppositionController
    original = OppositionController.action
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(ORIGINAL) != 1 or TREATMENT in source or original.__closure__:
        raise ValueError('Expected exactly the frozen primary-grasp failure branch')
    changed = source.replace(ORIGINAL, TREATMENT)
    # Separate function and locals; do not modify OppositionController or its module on disk.
    names = {}
    exec(compile(changed, '<contact_hold_grace_primary_lift_only>', 'exec'), original.__globals__, names)
    return names['action'], source, changed


def prepare(seed):
    from contact_hold_loop_data import inspect as inspect_rollout
    for name, digest in HASHES.items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise ValueError('Frozen implementation changed: ' + name)
    if sha(PANEL.read_bytes()) != PANEL_SHA or sha((FIT / 'best.pt').read_bytes()) != CHECKPOINT:
        raise ValueError('Round-15 model or reference validation panel changed')
    ep, = [e for e in json.loads(PANEL.read_text())['episodes'] if e['seed'] == seed]
    observed, *_ = inspect_rollout(Path(ep['run']), FIT, seed, 'validation')
    if ep != observed or ep['checkpoint'] != CHECKPOINT or ep['failure'] != 'lost_primary_grasp':
        raise ValueError('Original 10-frame baseline does not match archived evidence')
    action, original, changed = patched_action()
    receipt = dict(kind='round15_primary_lift_grace_paired_development_diagnostic',
                   seed=seed, checkpoint_sha256=CHECKPOINT, baseline_run=ep['run'],
                   baseline_manifest_sha256=ep['manifest'], baseline_panel_sha256=PANEL_SHA,
                   phase='primary_lift', failure='lost_primary_grasp', baseline_frames=10, treatment_frames=20,
                   original_module_sha256=HASHES['opposition_teacher.py'], original_action_sha256=sha(original.encode()),
                   experimental_action_sha256=sha(changed.encode()), worker_sha256=sha(Path(__file__).read_bytes()),
                   replacement={'old': ORIGINAL, 'new': TREATMENT}, unchanged_source_sha256=HASHES,
                   diagnostic_only=True, eligible_training_episode=False, original_final_success_audits=True)
    return action, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, choices=(21611, 21612), type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true', help='Check pinned baseline and intervention; no writes or simulation')
    args = parser.parse_args()
    out = args.output.resolve()
    if (out.parent.parent != RUNS or not re.fullmatch(r'contact_hold_grace_[A-Za-z0-9_]+', out.parent.name)
            or out.name != f'seed_{args.seed}' or out.exists()):
        parser.error('Choose a fresh runs/contact_hold_grace_<id>/seed_<seed> diagnostic directory')
    action, receipt = prepare(args.seed)
    if args.preflight:
        print(json.dumps(receipt, indent=2))
        return

    from contact_hold_student_eval import StudentEvaluation
    import contact_hold_student_eval as adapter

    class DiagnosticStudent(StudentEvaluation):
        def __init__(self, training_run, opts, directory):
            super().__init__(training_run, opts, directory)
            if self.digest != CHECKPOINT or directory.resolve() != out:
                raise ValueError('Wrong model or diagnostic destination')
            path = directory / 'grace_intervention.json'
            path.write_text(json.dumps(receipt, indent=2))
            shutil.copyfile(__file__, directory / 'grace_worker_source.py')
            self.protocol.update(diagnostic_only=True, primary_lift_lost_grace_frames=20,
                                 grace_intervention_sha256=sha(path.read_bytes()))
            (directory / 'student_evaluation.json').write_text(json.dumps(self.protocol, indent=2))

        def teacher_class(self):
            scheduler = super().teacher_class()
            import tilt_regrasp
            from opposition_teacher import OppositionController
            if tilt_regrasp.OppositionController is not OppositionController:
                raise RuntimeError('Unexpected pickup controller binding')
            class GraceController(OppositionController):
                pass
            GraceController.action = action
            # TiltTeacher explicitly calls OppositionController.action(self.base); overriding
            # self.base.action would have no effect. Only rebind that name in this interpreter.
            tilt_regrasp.OppositionController = GraceController
            return scheduler

    out.parent.mkdir(exist_ok=True)
    os.environ['CONTACT_HOLD_EVAL_ROLE'] = 'evaluation'
    sys.argv = [str(ROOT / 'single_face_experiment.py'), '--bc-evaluation-run', str(FIT), '--seed', str(args.seed),
                '--name', str(out.relative_to(RUNS)), '--controller', 'opposed', '--push-degrees', '0',
                '--adaptive-wrist', '--settled-hold', '--position-iterations', '64', '--split', 'development',
                '--max-steps', '3500']
    with patch.object(adapter, 'StudentEvaluation', DiagnosticStudent):
        runpy.run_path(str(ROOT / 'single_face_experiment.py'), run_name='__main__')


if __name__ == '__main__':
    main()
