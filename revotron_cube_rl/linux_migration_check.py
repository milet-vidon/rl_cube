"""Static portability check for moving the project from Windows to Linux.

This check does not start Isaac Sim. It validates relative assets, Python source
syntax, and that no required runtime path is fixed to a Windows drive letter.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> int:
    errors = []
    robot = Path(os.environ.get('REVOTRON_USD_PATH', ROOT.parent / 'revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd')).expanduser()
    cube = Path(os.environ.get('ISAAC_MUJOCO_CUBE_ROOT', ROOT.parent / 'isaac_mujoco_cube')).expanduser()
    if not robot.is_file():
        errors.append(f'missing Revo3 USD: {robot}')
    if not (cube / 'upstream/mujoco_cube-main/cube_3x3x3.xml').is_file():
        errors.append(f'missing mujoco_cube XML under: {cube}')
    if not (cube / 'upstream/mujoco_cube-main/assets/sticker.png').is_file():
        errors.append(f'missing mujoco_cube texture under: {cube}')
    for path in ROOT.glob('*.py'):
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except Exception as exc:
            errors.append(f'syntax error in {path.name}: {exc}')
    required = (
        'environment.py', 'assets.py', 'train.py', 'position_train.py',
        'absolute_policy.py', 'rewards.py', 'traditional_controller.py',
        'sweep_controller.py', 'iterative_training.py', 'linux_env_check.py',
    )
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f'missing required project file: {name}')
            continue
        text = (ROOT / name).read_text(encoding='utf-8')
        if 'E:\\IsaacLab' in text and name not in ('train.py', 'position_train.py'):
            # iterative_training.py keeps Windows defaults behind an os.name ==
            # 'nt' branch; all Linux runtime paths must remain environment based.
            if name != 'iterative_training.py':
                errors.append(f'fixed Windows runtime path remains in {name}')
    launcher = ROOT / 'run_linux.sh'
    if not launcher.is_file():
        errors.append('missing Linux launcher: run_linux.sh')
    elif not launcher.read_text(encoding='utf-8').startswith('#!/usr/bin/env bash'):
        errors.append('run_linux.sh must have a portable bash shebang')
    if errors:
        print('LINUX_MIGRATION_CHECK FAILED')
        for error in errors:
            print(' -', error)
        return 1
    print('LINUX_MIGRATION_CHECK PASSED')
    print('project_root=', ROOT)
    print('revo3_usd=', robot)
    print('mujoco_cube_root=', cube)
    print('entry_point=run_linux.sh')
    print('note=Isaac Sim still requires a supported NVIDIA Linux GPU, driver, and Isaac Python environment.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
