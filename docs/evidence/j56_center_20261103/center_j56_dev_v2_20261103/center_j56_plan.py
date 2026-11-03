"""Center the cube with outward J5 roll, inward J6 bend and reduced J7 flexion."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from centered_transport_plan import CenterScene
from center_j56_audit import joint_coordination
from face_push_plan import timed


class CoordinatedCenterScene(CenterScene):
    def valid_full(self, q):
        if not super().valid_full(q):
            return False
        frames = self.k.fk(q)
        fingers = ('thumb', 'index', 'middle', 'ring', 'little')
        shapes = [(name, mesh, frames[name] @ transform, center, radius)
                  for name, mesh, transform, center, radius in self.geometry
                  if name.startswith('left_')]
        for index, first in enumerate(shapes):
            first_finger = next((finger for finger in fingers if first[0].startswith('left_' + finger + '_')), None)
            if first_finger is None:
                continue
            for second in shapes[index + 1:]:
                second_finger = next((finger for finger in fingers if second[0].startswith('left_' + finger + '_')), None)
                if second_finger and second_finger != first_finger and self.distance(first, second) < .003:
                    return False
        return True


def plan(request):
    protocol = request['coordination_protocol']
    initial = request['initial']
    names = request['joint_names']
    arm = [f'Joint{i}_L' for i in range(1, 8)]
    scene = CoordinatedCenterScene(initial, request['cube_position'], request['cube_quaternion'], request['torso_colliders'])
    kinematics = scene.k
    palm = kinematics.fk_link(initial, 'left_palm_link')
    local_cube = scene.cube_local
    target = np.asarray(protocol['center_root_m'])
    reference = np.array([initial[name] for name in arm])
    limits = np.array([kinematics.limits[name] for name in arm])
    lower, upper = limits[:, 0] + .035, limits[:, 1] - .035
    preferred = reference[4:].copy()
    preferred[:2] += np.radians([protocol['j5_outward_delta_deg'], protocol['j6_inward_delta_deg']])
    preferred[2] = np.sign(reference[6]) * np.radians(min(protocol['j7_preferred_abs_deg'], max(abs(np.degrees(reference[6])) - 10., 0.)))
    preferred = np.clip(preferred, lower[4:], upper[4:])

    def residual(values):
        frames = kinematics.fk(initial | dict(zip(arm, values)))
        hand = frames['left_palm_link']
        shoulder, elbow, wrist = [frames[name][:3, 3] for name in ('Link2_L', 'Link4_L', 'Link7_L')]
        forearm = hand[:3, :3].T @ ((wrist - elbow) / np.linalg.norm(wrist - elbow))
        upper_arm_angle = np.arctan2((elbow - shoulder)[1], -(elbow - shoulder)[2])
        return np.r_[
            150 * ((hand @ local_cube)[:3, 3] - target),
            15 * max(hand[1, 0] + .75, 0),
            .25 * Rotation.from_matrix(palm[:3, :3] @ hand[:3, :3].T).as_rotvec(),
            55 * max(.18 - (elbow - shoulder)[1], 0),
            55 * max(.06 - (elbow - wrist)[1], 0),
            25 * max(.45 - forearm[2], 0),
            10 * max(np.radians(38) - upper_arm_angle, 0),
            30 * max(.24 - frames['Link5_L'][0, 3], 0),
            np.array([1.5, 3., 2.]) * (values[4:] - preferred),
            .10 * (values[:4] - reference[:4]),
        ]

    fit = least_squares(residual, np.clip(reference, lower + 1e-7, upper - 1e-7), bounds=(lower, upper), max_nfev=240)
    final = initial | dict(zip(arm, fit.x))
    finger_offsets = protocol.get('finger_target_offsets_rad', {})
    for name, offset in finger_offsets.items():
        if name not in ('left_index_MPR_joint', 'left_index_PIP_joint') or abs(offset) > .015:
            raise ValueError('Unsupported finger compensation')
        final[name] += offset
        if not kinematics.limits[name][0] < final[name] < kinematics.limits[name][1]:
            raise ValueError('Finger compensation violates joint limits')
    frames = kinematics.fk(final)
    cube = frames['left_palm_link'] @ local_cube
    if np.linalg.norm(cube[:3, 3] - target) > .002:
        raise ValueError('Coordinated center position error exceeds 2 mm')
    shoulder, elbow, wrist = [frames[name][:3, 3] for name in ('Link2_L', 'Link4_L', 'Link7_L')]
    forearm = frames['left_palm_link'][:3, :3].T @ ((wrist - elbow) / np.linalg.norm(wrist - elbow))
    bend = float(np.degrees(np.arctan2(forearm[1], forearm[2])))
    upper_angle = float(np.degrees(np.arctan2((elbow - shoulder)[1], -(elbow - shoulder)[2])))
    if elbow[1] - shoulder[1] < .16 or elbow[1] - wrist[1] < .05 or np.degrees(fit.x[5]) < 5 or bend > -5 or frames['left_palm_link'][1, 0] > -.6 or forearm[2] <= .3 or upper_angle < 35:
        raise ValueError('Coordinated center violates accepted wrist, elbow or palm posture')
    start = np.array([initial[name] for name in names])
    finish = np.array([final[name] for name in names])
    coordination = joint_coordination(finish, start, names, protocol)
    if not coordination['qualified']:
        raise ValueError('Requested J5/J6 coordination not achieved: ' + json.dumps(coordination))
    path = timed([start, finish], protocol['transport_seconds'], names, kinematics)
    previous = start
    checks = 0
    for step, point in enumerate(path):
        samples = max(1, int(np.ceil(np.max(np.abs(point - previous)) / .01)))
        for fraction in np.linspace(0, 1, samples + 1)[1:]:
            checks += 1
            state = dict(zip(names, (1 - fraction) * previous + fraction * point))
            if not scene.valid_full(state):
                return {'passed': False, 'phase': 'left_center_transport', 'step': step, 'pair': scene.last_pair, 'checks': checks, 'physics_validated': False}
        previous = point
    return {
        'passed': True,
        'style': protocol['version'],
        'phases': [('left_center_transport', path.tolist()), ('center_hold', np.repeat(finish[None], protocol['hold_steps'], axis=0).tolist())],
        'checks': checks,
        'center_target_world_m': (cube[:3, 3] + [0, 0, 1.15]).tolist(),
        'cube_target_quaternion_wxyz': Rotation.from_matrix(cube[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(),
        'palm_target_root': frames['left_palm_link'].tolist(),
        'initial_cube_world_m': request['cube_position'],
        'joint_coordination': coordination,
        'arm_joint_delta_deg': dict(zip(arm, np.degrees(fit.x - reference).tolist())),
        'side_bend_deg': bend,
        'forearm_local_z': float(forearm[2]),
        'finger_configuration_changed': bool(finger_offsets),
        'finger_target_offsets_rad': finger_offsets,
        'physics_validated': False,
    }


if __name__ == '__main__':
    try:
        result = plan(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as error:
        result = {'passed': False, 'error': str(error), 'physics_validated': False}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
    print('J56_CENTER_PLAN', result['passed'], result.get('checks'), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
