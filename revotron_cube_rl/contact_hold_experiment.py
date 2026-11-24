"""Run the frozen single-face harness with an explicit hold-controller override.

Only the left hold controller class changes. The original harness still checks
certificates, safety, contacts, speed, pose, and the three-second hold window.
The actual controller and this launcher are snapshotted before the first action.
"""
import argparse
import fcntl
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', type=int, default=11601)
    parser.add_argument('--max-steps', type=int, default=3500)
    parser.add_argument('--hold-damping', type=float, choices=(1., 2., 4.), default=1.)
    parser.add_argument('--opposing-pair-stiffness', type=float, choices=(10., 15., 20., 30.), default=30.,
                        help='Final thumb/little stiffness after a one-second ramp; index/middle/ring stay at 30')
    parser.add_argument('--source-joint-damping', action='store_true',
                        help='Use the existing source-MJCF linear passive joint damping migration')
    parser.add_argument('--position-iterations', type=int, choices=(32, 64))
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single directory name')
    root = Path(__file__).resolve().parent
    lock = (root.parent/'work/teacher_optimization_20261103/isaac_gui_evaluation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if jobs:
        raise RuntimeError('Another CUDA workload is active: '+jobs)
    import settled_wrist_teacher
    from contact_hold_teacher import ContactHoldTeacher
    ContactHoldTeacher.hold_damping = args.hold_damping
    ContactHoldTeacher.opposing_pair_stiffness = args.opposing_pair_stiffness
    ContactHoldTeacher.source_joint_damping = args.source_joint_damping
    ContactHoldTeacher.position_iterations = args.position_iterations
    settled_wrist_teacher.SettledWristTeacher = ContactHoldTeacher
    script = Path(__file__).with_name('single_face_experiment.py')
    sys.argv = [str(script), '--name', args.name, '--seed', str(args.seed),
                '--max-steps', str(args.max_steps), '--controller', 'opposed',
                '--push-degrees', '0', '--adaptive-wrist', '--settled-hold',
                '--split', 'development']
    if args.source_joint_damping:
        sys.argv.append('--source-joint-damping')
    if args.position_iterations is not None:
        sys.argv.extend(('--position-iterations', str(args.position_iterations)))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
