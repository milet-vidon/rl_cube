"""Run an isolated hold candidate through the unchanged physical experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=11601)
    p.add_argument('--hold-damping', type=float, choices=(1., 2., 4.), default=1.)
    p.add_argument('--max-steps', type=int, default=3500)
    p.add_argument('--record', action='store_true')
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('name must be a new run directory name')
    run = ROOT / 'runs' / args.name
    if run.exists():
        raise FileExistsError(run)
    lock = (ROOT.parent / 'work/teacher_optimization_20261103/isaac_gui_evaluation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader'], text=True).strip()
    if jobs:
        raise RuntimeError('Another CUDA workload is active: ' + jobs)
    import settled_wrist_teacher
    from hold_contact_teacher_v12 import ContactHoldTeacher
    ContactHoldTeacher.hold_damping = args.hold_damping
    original_init = ContactHoldTeacher.__init__

    def archive_init(self, env, *positional, **kwargs):
        original_init(self, env, *positional, **kwargs)
        sources = [Path(__file__), ROOT / 'hold_contact_teacher_v12.py']
        for src in sources:
            (run / src.name).write_bytes(src.read_bytes())
        manifest = dict(schema='contact_hold_candidate_v12', seed=args.seed,
                        settings=dict(stiffness=30., damping=args.hold_damping, ramp_steps=30),
                        acceptance_thresholds_changed=False, trained_policy=False,
                        source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
        (run / 'hold_candidate_v12.json').write_text(json.dumps(manifest, indent=2))
    ContactHoldTeacher.__init__ = archive_init
    settled_wrist_teacher.SettledWristTeacher = ContactHoldTeacher
    sys.argv = [str(ROOT / 'single_face_experiment.py'), '--name', args.name,
                '--seed', str(args.seed), '--controller', 'opposed', '--push-degrees', '0',
                '--adaptive-wrist', '--settled-hold', '--max-steps', str(args.max_steps),
                '--split', 'development'] + (['--record'] if args.record else [])
    runpy.run_path(str(ROOT / 'single_face_experiment.py'), run_name='__main__')


if __name__ == '__main__':
    main()
