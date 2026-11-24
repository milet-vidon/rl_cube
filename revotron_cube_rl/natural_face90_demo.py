"""A human controls the right wrist, thumb and index after the physical R27 pickup."""
import argparse
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import time
import uuid

from natural_face90_v5_worker import inputs, PARENT_HASH, ROOT, require, save, sha

DEFAULT_SEED = 66002220
MANUAL_KEYS = ('move', 'rotate', 'thumb', 'index', 'opposition')


def assets(seed):
    # The archived loader validates a block of at least two seeds. The second is
    # only a provenance check; the demonstration creates and resets a single lane.
    companion = 66002208 if seed != 66002208 else 66002211
    return inputs(dict(role='validate', target_deg=15, parent_sha=PARENT_HASH,
                       seeds=[seed, companion], parameters=[[0.]*6 for _ in range(2)]))


class Pickup:
    def __init__(self, env, grasp, meta, model):
        import torch
        from contact_hold90_cuda_domain import make_plan, make_teacher
        from contact_hold90_cuda_sample import HoldMeter, local_palm_reader
        from phase_contact_policy import PHASES
        self.env = env
        self.model = model.to(env.home.device).eval()
        self.ids = meta['output_indices']
        require(env.names == meta['joint_names'] and self.ids == [
            i for i,k in enumerate(env.names) if k.startswith('left_') or k.endswith('_L')], 'Wrong R27 joint map')
        plan = make_plan(env.core(), grasp)
        require(bool(plan.accepted[0]), 'R27 rejected this reset')
        self.teacher = make_teacher(plan, env.cfg, meta['sources'][0]['candidate'], grasp, env.batch_gain)
        self.meter = HoldMeter(1, env.home.device)
        self.palm = local_palm_reader(env)
        self.nph = len(PHASES)
        self.last_phase = -1
        self.start = self.hold = 0
        self.cap = torch.minimum(env.step, env.vel*env.dt*env.dec)

    def command(self, step, active, target, q, dq):
        import torch
        from contact_hold90_cuda_domain import FAILURES90
        e = self.env
        feedback = e.contacts.clone()
        if step == 0:
            feedback[:, :10] = 0.
        core = e.core()
        raw, ph, fail, accepted = self.teacher.action(active,q,dq,core,target,feedback,
                                                      e.views['Core'].get_velocities(),e.hand_impulse)
        if int(fail[0]):
            raise RuntimeError('R27 pickup: '+FAILURES90.get(int(fail[0]),'invalid teacher failure'))
        if int(ph[0]) != self.last_phase:
            self.start, self.last_phase = step, int(ph[0])
        contacts = torch.log1p(e.feature_contacts.to(q.dtype)/.001) if step else q.new_zeros((1,10))
        obs = torch.cat((q,dq,target,core[:,:3],core[:,[6,3,4,5]],contacts,
                         torch.nn.functional.one_hot(ph,self.nph).to(q.dtype),
                         q.new_tensor([[min((step-self.start)/300.,1.)]])),1)
        require(obs.shape == (1,198) and bool(obs.isfinite().all()), 'Invalid pickup observation')
        pred = self.model(obs)
        old, cap = target[:,self.ids], self.cap.expand_as(target)[:,self.ids]
        command = raw.clone()
        command[:,self.ids] = old + (pred-old).clamp(-cap,cap)
        return command, bool(accepted[0])

    def endpoint(self, step, active, accepted, safety):
        flags, _, assembled, height, table, support = safety
        e = self.env
        self.meter.update(step,active,support & assembled,self.palm(e.core()))
        orig = e.cfg['success']
        v = e.views['Core'].get_velocities()
        stable = ((e.hand_impulse > orig['contact_impulse_min']).all(1) & (height >= orig['lift_m'])
                  & ~table & ~flags[2] & (v[:,:3].norm(dim=1) < orig['linear_speed_max'])
                  & (v[:,3:].norm(dim=1) < orig['angular_speed_max']))
        self.hold = self.hold + 1 if bool(stable[0]) else 0
        if accepted:
            require(bool((self.meter.end & support & assembled)[0]), 'R27 endpoint without supported hold')
            return True
        require(self.hold < round(orig['hold_seconds']*30), 'Environment finished before R27 wrist endpoint')
        return False


def read_safety(env, command, soft, arms, init_xy, phase, lost):
    import torch
    from natural_face90_v5_worker import measurements
    poses = torch.stack([view.get_transforms() for view in env.views.values()], 1)
    safety = measurements(env,command,soft,arms,init_xy,env.core(),poses)
    flags, deformed, _, _, table, support = safety
    reasons = ('hard_joint_limit','soft_arm_limit','cube_drop','robot_self_contact','robot_table_contact')
    reason = next((s for s,flag in zip(reasons,flags) if bool(flag[0])),None)
    if phase == 'pickup':
        if not reason and bool(deformed[0]):
            reason = 'layer_deviation'
    else:
        lost = 0 if bool(support[0]) else lost+1
        if not reason and bool(table[0]):
            reason = 'cube_returned_to_table'
        if not reason and lost >= env.domain_receipt['loss_grace_frames']:
            reason = 'primary_support_lost'
    return safety, reason, lost


def state_for_page(row, phase, reason=None):
    fields = ('step', 'cube', 'bodies', 'angle_deg', 'thumb_Ns', 'index_Ns', 'other_Ns',
              'thumb_index_same_substeps')
    return dict(phase=phase, reason=reason or '', **{k: row[k] for k in fields})


def demonstrate(ui, seed, out, loaded):
    import torch
    from joint_safety import project_target
    from natural_face90_demo_scene import open_demo_scene
    from natural_face90_recipe_contacts import step_physics
    from natural_face90_v5_controller import NaturalFace90Controller
    from natural_face90_v5_metrics import NaturalFace90Observer
    from natural_face90_demo_control import ManualControl
    from natural_face90_demo_record import Recorder

    source, grasp, meta, model = loaded
    e = open_demo_scene(source, grasp)
    e.reset([seed])
    require(e.n == 1 and bool(e.contacts[0,12] > 0), 'Missing reset table contact')
    # Install the project observer first, then the recording tap that delegates to it.
    monitor = NaturalFace90Observer(e, target_deg=90.)
    prepare = NaturalFace90Controller(e, [0.]*6, target_deg=90.)
    pickup = Pickup(e, grasp, meta, model)
    active = torch.ones(1,dtype=torch.bool,device=e.home.device)
    false = ~active
    target = e.home.clone()
    arms = [i for i,n in enumerate(e.names) if n.startswith('Joint')]
    xy = target.new_tensor(e.cfg['cube_xyz'][:2])
    phase, step, prep_steps, lost = 'pickup', 0, 0, 0
    pending, remaining, row, obs = {}, 0, None, None
    with ExitStack() as stack:
        control = ManualControl(e)
        stack.callback(control.close)
        files = [*sorted(ROOT.glob('natural_face90_demo*.py')), ROOT/'natural_face90_demo.html',
                 ROOT/'run_two_finger_demo.sh']
        recorder = Recorder(out, e, dict(seed=seed, r27_sha256=PARENT_HASH, source=str(source),
                            grasp=str(grasp),
                            software_sha256={f.name: sha(f) for f in files},
                            input_frame='robot base; wrist rotation about palm; orbit about measured Core nZ',
                            solver=control.placo.receipt))
        def end_recording(exc_type, exc_value, _tb):
            if exc_type is None:
                recorder.close()
            else:
                recorder.finish(str(exc_value) or exc_type.__name__, failed=True)
        stack.push(end_recording)
        while True:
            if phase == 'manual' and not remaining:
                ui.publish(state_for_page(row, 'manual'))
                packet = ui.next_action()
                if packet.get('finish'):
                    summary = recorder.finish('operator_saved')
                    break
                remaining = packet['ticks']
                pending = {k:packet[k] for k in MANUAL_KEYS}
                pending['orbit'] = packet.get('orbit', 0)
            q = e.robot.get_dof_positions()[:,e.ids].clone()
            accepted = False
            pre_failure = None
            if phase == 'pickup':
                desired, accepted = pickup.command(step,active,target,q,e.robot.get_dof_velocities()[:,e.ids])
            elif phase == 'prepare':
                desired, _, failed, reasons = prepare.action(target,active,obs)
                pre_failure = reasons[0] if bool(failed[0]) else None
            else:
                desired = control.step(target,pending)
                remaining -= 1
            cmd, soft = project_target(target,desired,e.limits,e.step,e.vel,e.dt*e.dec)
            step_physics(e,cmd,out)
            target = cmd
            safety, reason, lost = read_safety(e,cmd,soft,arms,xy,phase,lost)
            post = phase != 'pickup'
            obs = monitor.observe(active,active if post else false,active if reason is None else false,false,step)
            if post and not reason:
                if not bool(obs['structure_ok'][0]) or not bool(obs['allowed_layer_state'][0]):
                    reason = 'layer_structure_or_pose'
                elif bool(obs['wrong_contact'][0]):
                    reason = 'right_contact_wrong_piece_or_link'
            next_phase = phase
            stop = reason or pre_failure
            if not stop and phase == 'pickup':
                try:
                    ready = pickup.endpoint(step,active,accepted,safety)
                except ValueError as exc:
                    stop = str(exc)
                else:
                    if ready:
                        prepare.begin(active,target,obs)
                        next_phase = 'prepare'
                    elif step+1 >= 3500:
                        stop = 'pickup_horizon'
            elif not stop and phase == 'prepare':
                prep_steps += 1
                if bool((obs['natural_ready'] & prepare.detour_done & (prepare.phase>=2))[0]):
                    next_phase = 'manual'
                elif prep_steps >= 2200:
                    stop = 'preparation_horizon'
            row = recorder.record(step,phase,pending if phase=='manual' else {},q,cmd,
                                  safe=reason is None,end_reason=stop)
            if stop:
                summary = recorder.finish(stop, failed=True)
                break
            if next_phase=='manual' and phase!='manual':
                recorder.begin_handoff(step+1,target)
                print('Manual control ready. Select an action in the browser.',flush=True)
            elif phase!='manual' and (step%20==0 or phase!=next_phase):
                ui.publish(state_for_page(row, next_phase))
            phase, step = next_phase, step+1
        final = 'failed' if summary['status']=='failed' else 'finished'
        ui.publish(dict(state_for_page(row, final, summary['reason']), summary_path=str(out/'demo_summary.json')))
        return summary


def check(seed):
    from natural_face90_placo_client import self_check
    from natural_face90_recipe_contacts import check_capacity_source
    from natural_face90_recipe_frames import check_source_scene
    from natural_face90_demo_usd import check_demo_usd
    from natural_face90_demo_web import DemoUI
    from natural_face90_demo_control import ManualControl
    from natural_face90_demo_record import Recorder
    loaded=assets(seed)
    check_capacity_source(1)
    check_source_scene(loaded[0])
    mounted = check_demo_usd(loaded[0])
    info=self_check()
    print(f"CPU CHECK OK: R27, mounted USD {mounted['articulation_path']}, "
          f"{mounted['cube_bodies']} cube bodies, contact capacity and Placo {info['version']}")
    return loaded


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='CPU-only preflight')
    modes.add_argument('--replay', type=Path, metavar='RUN_DIR', help='Replay an existing recording without Isaac or a GPU')
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--port',type=int,default=8766)
    args=p.parse_args(argv)
    if not 1024 <= args.port <= 65535 or not 61_000_000 <= args.seed < 71_000_000:
        p.error('Use an unprivileged TCP port and a development seed in [61000000,71000000)')
    if args.check:
        check(args.seed)
        return
    from natural_face90_demo_web import DemoUI
    if args.replay:
        from natural_face90_demo_replay import replay
        with DemoUI(args.port) as ui:
            try:
                replay(ui, args.replay)
            except KeyboardInterrupt:
                pass
        return
    from natural_face90_v5_long import gpu
    with ExitStack() as resources:
        try:
            resources.enter_context(gpu())
        except RuntimeError as exc:
            if 'GPU 0' in str(exc) or '占用' in str(exc):
                raise SystemExit('GPU 0 正由其他任务使用。空闲后重新运行 bash run_two_finger_demo.sh。') from None
            raise
        ui = resources.enter_context(DemoUI(args.port))
        print(f'Open http://127.0.0.1:{args.port}/ on this host or through SSH forwarding.',flush=True)
        out = ROOT/'runs'/'human_demos'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=True, exist_ok=False)
        print('Recording to', out, flush=True)
        app = None
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True, device='cuda:0').app
            import torch
            with torch.no_grad():
                result=demonstrate(ui,args.seed,out,assets(args.seed))
            print('Saved',out/'demo_summary.json',result['reason'],flush=True)
            # Kit may exit the process in close_application, before the UI context exits.
            time.sleep(2)
        except BaseException as exc:
            import traceback
            trace = traceback.format_exc()
            save(out/'demo_error.json', dict(reason=repr(exc), seed=args.seed, traceback=trace))
            print(trace, flush=True)
            ui.publish(dict(phase='failed', reason=str(exc) or type(exc).__name__, run_dir=str(out)))
            time.sleep(1)
            raise
        finally:
            if app is not None:
                from isaac_compat import close_application
                close_application(app)


if __name__=='__main__':
    main()
