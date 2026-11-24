"""Deployment adapter for the pickup-to-wrist contact-hold BC checkpoint."""
from pathlib import Path
import json
import os
import torch

from contact_hold_dataset import audit_demo, sha
from contact_hold_policy import restored
from phase_contact_policy import features

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


class StudentEvaluation:
    def __init__(self, training_run, args, out):
        training = Path(training_run).resolve()
        report = read(training/'result.json')
        path = training/'best.pt'
        if not (report['completed'] and report['checkpoint_reload_verified']
                and report['checkpoint_sha256'] == sha(path)):
            raise ValueError('Checkpoint does not match the completed fit')
        (out/'evaluated_best.pt').write_bytes(path.read_bytes())
        self.digest = sha(out/'evaluated_best.pt')
        if self.digest != report['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during snapshot')
        ckpt = torch.load(out/'evaluated_best.pt', map_location='cpu', weights_only=True)
        meta = ckpt['metadata']
        if meta != read(training/'experiment.json') or ckpt['metrics']['update'] != report['best_update']:
            raise ValueError('Training metadata or best update mismatch')
        for name, digest in meta['source_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Changed training dependency: ' + name)
        self.model = restored(ckpt)
        self.names = meta['joint_names']
        self.left = meta['output_indices']
        self.phase = None
        self.phase_start = 0
        self.sources = [Path(s['path']) for s in meta['sources']]
        self.source = self.sources[0]
        for p, s in zip(self.sources, meta['sources']):
            if audit_demo(p)[-1] != s:
                raise ValueError('Training demonstration no longer reproduces its audit')
        candidate = meta['sources'][0]['candidate']
        for name, digest in candidate['source_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Changed teacher code: ' + name)
        src = read(self.source/'experiment.json')
        # This harness adds the student action hook; use the original source hashes
        # for every other simulator/controller/audit module in the demonstration.
        for name, digest in src['source_sha256'].items():
            if name != 'single_face_experiment.py' and sha(ROOT/name) != digest:
                raise ValueError('Changed experiment dependency: ' + name)
        role = os.getenv('CONTACT_HOLD_EVAL_ROLE', 'evaluation')
        if role not in ('collection','validation','test','evaluation') or getattr(args,'split','development') != ('independent' if role == 'test' else 'development'):
            raise ValueError('Evaluation role and split disagree')
        trained = {s['seed'] for s in meta['sources']} | set(meta.get('training_seeds', []))
        if role == 'test' and args.seed in trained:
            raise ValueError('A training seed cannot be an independent test')
        for key, value in vars(args).items():
            if key not in ('bc_evaluation_run','name','seed','split') and value != src[key]:
                raise ValueError('Different demonstration setting: ' + key)
        self.candidate = candidate
        self.protocol = dict(kind='contact_hold_bc_closed_loop_evaluation',
                             training_run=str(training), checkpoint_sha256=self.digest,
                             seed=args.seed, evaluation_role=role, seen_training_seed=args.seed in trained,
                             best_update=report['best_update'], left_position_actions='student_from_reset',
                             right_actions_phase_and_gains='original_teacher', initial_action_supervised=False,
                             eligible_training_episode=False, corrective_labels_qualified_for_training=False,
                             actor_source_sha256=sha(Path(__file__)), harness_sha256=sha(ROOT/'single_face_experiment.py'),
                             training_harness_sha256=src['source_sha256']['single_face_experiment.py'])
        (out/'student_evaluation.json').write_text(json.dumps(self.protocol,indent=2))

    def teacher_class(self):
        from contact_hold_teacher import ContactHoldTeacher
        c = self.candidate
        class Scheduler(ContactHoldTeacher):
            hold_stiffness = c['left_finger_hold_stiffness']
            hold_damping = c['left_finger_hold_damping']
            opposing_pair_stiffness = c['thumb_little_final_stiffness']
            gain_ramp_steps = c['thumb_little_ramp_steps']
            position_iterations = c['position_iterations']
            source_joint_damping = c['source_joint_linear_damping_enabled']
        return Scheduler

    def bind(self, env, out):
        for name in ('config.json','asset_contract.json','solver_verification.json','runtime_robot_asset.json','cube_asset.json'):
            if read(self.source/name) != read(out/name):
                raise ValueError('Runtime differs from training demonstration: ' + name)
        names = env.cfg['joint_names']
        left = [i for i,n in enumerate(names) if n.startswith('left_') or n.endswith('_L')]
        if names != self.names or left != self.left or len(left) != 28 or env.auto_reset:
            raise ValueError('Wrong joint order or reset semantics')
        self.cap = torch.minimum(torch.minimum(env.scales,env.max_target_delta),
                                 env.command_velocity_limits*env.control_dt)[:,left]
        # Exact runtime values, including all 21 finger limits. Replay must not
        # substitute nominal arm waypoints for state-dependent finger feedback.
        (out/'student_left_projection.json').write_text(json.dumps(dict(
            names=[names[i] for i in left], bounds=env.limits[0,left].tolist(),
            step=torch.minimum(env.scales,env.max_target_delta)[0,left].tolist(),
            velocity=env.command_velocity_limits[0,left].tolist(), dt=float(env.control_dt)), allow_nan=False))

    @torch.inference_mode()
    def action(self, env, phase, step, previous, native):
        from isaac_compat import tensor, quaternion_from_native
        if previous is None:
            state = tensor(env.cube.data.root_state_w)[0]
            previous = dict(q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),
                            joint_velocity=tensor(env.robot.data.joint_vel)[0,env.ids].tolist(),
                            applied_target=env.targets[0].tolist(), cube_position=state[:3].tolist(),
                            cube_quaternion_wxyz=quaternion_from_native(state[3:7]).tolist(), finger_contact_points=[])
        if self.phase != phase:
            self.phase_start, self.phase = step, phase
        clock = min((step-self.phase_start)/300.,1.)
        x = torch.from_numpy(features(previous,phase,clock)).unsqueeze(0)
        before = env.targets[:,self.left]
        teacher_left = (before + env.scales[:,self.left] * native[:,self.left])[0].tolist()
        requested = self.model(x)
        target = before + (requested-before).clamp(-self.cap,self.cap)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite student target')
        action = native.clone()
        action[:,self.left] = (target-before)/env.scales[:,self.left]
        return action, dict(executed_left_source='student', phase_elapsed=clock,
                            student_raw_left_target=requested[0].tolist(),
                            student_left_target=target[0].tolist(), teacher_left_requested_not_executed=teacher_left)
