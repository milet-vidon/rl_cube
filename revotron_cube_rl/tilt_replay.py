"""Render measured simulation states; never use replay to calculate success.

The source trajectory is produced by tilt_experiment.py with live physics,
contact sensing, and the unchanged control safety layer. This viewer restores
recorded states only to visualize that evidence without rerunning the controller.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import threading

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--start-step', default=0, type=int)
args = parser.parse_args()
run = Path(args.run).resolve()
source = run / f'trajectory_{args.seed}.json'
rows = json.loads(source.read_text())
selected = [r for r in rows if r['step'] >= args.start_step and r['step'] % 3 == 0]
if not selected or selected[-1]['step'] != rows[-1]['step']:
    selected.append(rows[-1])

from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
from isaac_compat import tensor, quaternion_to_native, close_application

app = AppLauncher(headless=True, enable_cameras=True, kit_args=platform_kit_args(
    '--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false')).app
env = None
try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from environment import CubePickupEnv
    from side_video import encode_mp4
    from tilt_audit import tilt_error

    env = CubePickupEnv(vision=True, preview=True, auto_reset=False, arm_damping=8.)
    env.overview.set_world_poses_from_view(torch.tensor([[1.30, -.80, 1.50]]), torch.tensor([[.40, -.04, 1.12]]))
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
    phases = {'ready':'准备', 'preshape':'左手预抓形', 'lower':'左手下降', 'approach':'左手靠近',
              'squeeze':'左手夹紧', 'primary_lift':'左手抬起', 'right_clear':'右手让开',
              'tilt_90':'向右侧倾翻 90°', 'align_right':'移向接抓位置', 'tilt_hold':'翻转后保持',
              'right_preshape':'右手预抓形', 'right_route':'右手绕行靠近', 'right_approach':'右手侧向进入',
              'right_close':'右手拇指对四指闭合', 'bilateral_hold':'双手保持'}
    closing_start = next((r['step'] for r in rows if r['phase']=='right_close'), rows[-1]['step'])
    phase_images = {}
    frames = []
    for index, row in enumerate(selected):
        q = tensor(env.robot.data.default_joint_pos).clone()
        q[0, env.ids] = torch.tensor(row['q'])
        velocity = torch.zeros_like(q)
        velocity[0, env.ids] = torch.tensor(row['joint_velocity'])
        env.robot.write_joint_state_to_sim(q, velocity)
        env.robot.set_joint_position_target(q)
        state = tensor(env.cube.data.default_root_state).clone()
        state[0, :3] = torch.tensor(row['cube_position'])
        state[0, 3:7] = quaternion_to_native(torch.tensor(row['cube_quaternion_wxyz']))
        state[0, 7:] = 0.
        env.cube.write_root_state_to_sim(state)
        env.robot.write_data_to_sim()
        env.cube.write_data_to_sim()
        env.sim.forward()
        env.robot.update(env.control_dt)
        env.cube.update(env.control_dt)
        zoom = np.clip((row['step'] - closing_start) / 60., 0., 1.)
        eye = (1-zoom)*np.array([1.30, -.80, 1.50]) + zoom*np.array([.95, -.50, 1.36])
        target = (1-zoom)*np.array([.40, -.04, 1.12]) + zoom*np.asarray(row['cube_position'])
        env.overview.set_world_poses_from_view(torch.tensor([eye],dtype=torch.float32), torch.tensor([target],dtype=torch.float32))
        env._render()
        image = Image.fromarray(tensor(env.overview.data.output['rgb'])[0, :, :, :3].numpy())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 44), fill='black')
        draw.text((6, 2), '传统控制 · 物理仿真轨迹回放 | 整体倾翻魔方', font=font, fill='white')
        draw.text((6, 22), f'{phases.get(row["phase"],row["phase"])} | 种子 {args.seed} | 仿真时间 {row["step"]/30:.1f} 秒', font=font, fill='white')
        draw.rectangle((0, 474, 640, 512), fill='black')
        left = sum(v > 1e-5 for v in row['finger_contact_impulses']['left'].values())
        right = sum(v > 1e-5 for v in row['finger_contact_impulses']['right'].values())
        draw.text((6, 474), f'目标 90° | 姿态误差 {tilt_error(row):.1f}° | 抬升 {row["height_m"]*100:.1f} 厘米', font=font, fill='white')
        draw.text((6, 493), f'接触手指：左 {left}/5  右 {right}/5 | 自碰 {row["self_contact_impulse"]:.3f} / 碰桌 {row["robot_table_impulse"]:.3f} N·s', font=font, fill='white')
        frames.append(image)
        phase_images[row['phase']] = image.copy()
        if index % 60 == 0:
            print('RENDER', index, '/', len(selected), flush=True)
    frames[-1].save(run / f'replay_final_{args.seed}.png')
    for phase in ('primary_lift','tilt_hold','bilateral_hold'):
        if phase in phase_images:
            phase_images[phase].save(run / f'replay_{phase}_{args.seed}.png')
    encode_mp4(run / f'replay_{args.seed}.mp4', frames)
    frames[0].save(run / f'replay_{args.seed}.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    manifest = {
        'kind': 'visualization_of_measured_physics_trajectory',
        'source_trajectory': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'renderer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recorded_steps': [r['step'] for r in selected],
        'fps': 10,
        'frames': len(frames),
        'frozen_tail_frames': 0,
        'replay_used_for_acceptance': False,
    }
    (run / f'replay_manifest_{args.seed}.json').write_text(json.dumps(manifest, indent=2))
    print('REPLAY COMPLETE', len(frames), flush=True)
finally:
    code = 1 if sys.exc_info()[0] else 0
    threading.Timer(15, lambda: os._exit(code)).start()
    if env:
        env.close()
    close_application(app)
