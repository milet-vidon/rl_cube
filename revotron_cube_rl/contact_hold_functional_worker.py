"""Run one round-15 student under count-independent online contact gates.

The scheduler owns the device locks. The official wrist audit keeps its own
criteria; task-level steady holding is measured separately from the trajectory.
"""
import argparse
from contextlib import ExitStack
from difflib import unified_diff
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import textwrap
from unittest.mock import patch

from contact_hold_grace_worker import CHECKPOINT, FIT, HASHES, PANEL, PANEL_SHA, ROOT, RUNS, sha

MODES = ('matched_lift', 'from_reset')
SEEDS = (21611, 21612)
SUPPORT = "(self.env.contact_impulse[0] > self.env.cfg['success']['contact_impulse_min'])"


def transform(fn, replacements):
    original = textwrap.dedent(inspect.getsource(fn))
    if fn.__closure__:
        raise ValueError('Only the pinned standalone controller methods are supported')
    changed = original
    for old, new in replacements:
        if changed.count(old) != 1:
            raise ValueError(f'Unexpected {fn.__qualname__} code: {old}')
        changed = changed.replace(old, new)
    namespace = {}
    exec(compile(changed, f'<functional:{fn.__qualname__}>', 'exec'), fn.__globals__, namespace)
    hunk = ''.join(unified_diff(original.splitlines(True), changed.splitlines(True), 'reference', 'diagnostic'))
    return namespace['action'], dict(original=sha(original.encode()), diagnostic=sha(changed.encode()), diff=hunk)


def methods(mode):
    from opposition_teacher import OppositionController
    from tilt_regrasp import TiltTeacher
    if mode not in MODES:raise ValueError(mode)
    stages = "('squeeze','primary_lift')" if mode=='from_reset' else "('primary_lift',)"
    old = 'self.primary_streak=self.primary_streak+1 if all_primary else 0'
    pickup, pdiff = transform(OppositionController.action, (
        (old, f'self.primary_streak=self.primary_streak+1 if ({SUPPORT} if self.name in {stages} else all_primary) else 0'),
        ('self.lost=0 if topology else self.lost+1', f'self.lost=0 if {SUPPORT} else self.lost+1'),
        ("if self.lost>=10:self.failure_reason='lost_primary_grasp'", "if self.lost>=20:self.failure_reason='lost_primary_support'"),
    ))
    tilt, tdiff = transform(TiltTeacher.action, (
        ("all_fingers=all(primary.get(f,0)>1e-5 for f in ('thumb','index','middle','ring','little'))", f'left_supported={SUPPORT}'),
        ("retained=primary.get('thumb',0)>1e-5 and sum(primary.get(f,0)>1e-5 for f in ('index','middle','ring','little'))>=3", 'retained=left_supported'),
        ("if self.lost>=10:self.failure_reason='primary_grasp_lost_during_tilt'", "if self.lost>=20:self.failure_reason='primary_support_lost_during_tilt'"),
        (' and all_fingers and safe_pose', ' and left_supported and safe_pose'),
    ))
    return pickup,tilt,{'OppositionController.action':pdiff,'TiltTeacher.action':tdiff}


def prepare(mode, seed):
    from contact_hold_loop_data import inspect as inspect_rollout
    for name, digest in HASHES.items():
        if sha((ROOT/name).read_bytes())!=digest:raise ValueError('Changed frozen source: '+name)
    if sha(PANEL.read_bytes())!=PANEL_SHA or sha((FIT/'best.pt').read_bytes())!=CHECKPOINT:
        raise ValueError('Changed reference panel or checkpoint')
    ep, = [e for e in json.loads(PANEL.read_bytes())['episodes'] if e['seed']==seed]
    observed, *_=inspect_rollout(Path(ep['run']), FIT, seed, 'validation')
    if observed!=ep or ep['failure']!='lost_primary_grasp':raise ValueError('Invalid reference trial')
    pickup,tilt,diffs=methods(mode)
    met_path=ROOT/'contact_hold_physical_metrics.py'
    receipt=dict(kind='functional_hold_intervention_v1', mode=mode, seed=seed,
            checkpoint_sha256=CHECKPOINT,baseline_run=ep['run'],baseline_manifest_sha256=ep['manifest'],
            baseline_panel_sha256=PANEL_SHA,original_source_sha256=HASHES,method_diffs=diffs,
            support='total measured left hand/cube impulse above cfg.success.contact_impulse_min; no named finger or count',
            squeeze_gate='original' if mode=='matched_lift' else 'total left contact for 10 steps and original arm/path settling',
            lift_postlift_loss_grace_frames=20,
            dimensions_not_changed='observations, policy, finger feedback, motion and gain schedules, settling/height/speed/orientation/safety thresholds, offline wrist audits',
            diagnostic_only=True, eligible_training_episode=False,
            worker_sha256=sha(Path(__file__).read_bytes()),metrics_sha256=sha(met_path.read_bytes()))
    return pickup,tilt,receipt


def simulate(out, seed, mode):
    pickup,tilt,receipt=prepare(mode, seed)
    import contact_hold_student_eval as adapter
    import tilt_regrasp
    from opposition_teacher import OppositionController
    from tilt_regrasp import TiltTeacher
    if tilt_regrasp.OppositionController is not OppositionController:
        raise RuntimeError('Unexpected controller binding')
    met_path=ROOT/'contact_hold_physical_metrics.py'

    class DiagnosticStudent(adapter.StudentEvaluation):
        def __init__(self,training_run,opts,directory):
            super().__init__(training_run,opts,directory)
            if self.digest!=CHECKPOINT or directory.resolve()!=out:raise ValueError('Wrong episode or student')
            rec=directory/'functional_intervention.json'
            rec.write_text(json.dumps(receipt,indent=2,allow_nan=False))
            shutil.copyfile(__file__,directory/'functional_worker_source.py')
            shutil.copyfile(met_path,directory/'physical_metrics_source.py')
            self.protocol.update(kind='contact_hold_student_functional_diagnostic',diagnostic_only=True,functional_mode=mode,
                                 functional_intervention_sha256=sha(rec.read_bytes()))
            (directory/'student_evaluation.json').write_text(json.dumps(self.protocol,indent=2))

        def teacher_class(self):
            scheduler=super().teacher_class()
            class DiagnosticScheduler(scheduler):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    if self.regrasp or not self.base.finger_arrival:
                        raise ValueError('Expected the frozen primary-grasp/whole-cube wrist protocol')
                    marker=self.env.run_dir/'contact_hold_candidate.json'
                    j=json.loads(marker.read_bytes())
                    j.update(original_contact_and_speed_acceptance_retained=False, original_speed_limits_retained=True,
                             diagnostic_only=True,functional_mode=mode)
                    marker.write_text(json.dumps(j,indent=2))
            return DiagnosticScheduler

    os.environ['CONTACT_HOLD_EVAL_ROLE']='evaluation'
    sys.argv=[str(ROOT/'single_face_experiment.py'),'--bc-evaluation-run',str(FIT),'--seed',str(seed),'--name',str(out.relative_to(RUNS)),
              '--controller','opposed','--push-degrees','0','--adaptive-wrist','--settled-hold','--position-iterations','64',
              '--split','development','--max-steps','3500']
    # TiltTeacher explicitly calls OppositionController.action on self.base. Patching
    # these methods in this one process covers both the normal and settled-hold calls.
    with ExitStack() as s:
        s.enter_context(patch.object(adapter,'StudentEvaluation',DiagnosticStudent))
        s.enter_context(patch.object(OppositionController,'action',pickup))
        s.enter_context(patch.object(TiltTeacher,'action',tilt))
        runpy.run_path(str(ROOT/'single_face_experiment.py'),run_name='__main__')


def finalize(out,mode,seed):
    from contact_hold_physical_metrics import summarize_hold
    m=(out/'artifact_manifest.json').read_bytes()
    manifest=json.loads(m)
    def checked(name):
        b=(out/name).read_bytes()
        if sha(b)!=manifest['sha256'][name]:raise ValueError('Corrupt diagnostic file: '+name)
        return b
    r=json.loads(checked('result.json')); receipt=json.loads(checked('functional_intervention.json'))
    prot=json.loads(checked('student_evaluation.json'));rows=json.loads(checked('trajectory.json'))
    cfg=json.loads(checked('config.json')); runtime=json.loads(checked('runtime.json'))
    if not (manifest['seed']==r['seed']==receipt['seed']==seed and manifest['full_episode'] and r['full_episode'] and
            manifest['split']==r['split']=='development' and not r['execution_interrupted'] and r['evaluation_role']=='evaluation' and
            r['kind']=='contact_hold_student_functional_diagnostic' and not (r['eligible_training_episode'] or r['corrective_labels_qualified_for_training']) and
            r['diagnostic_only'] and r['functional_mode']==receipt['mode']==mode and r['functional_intervention_sha256']==prot['functional_intervention_sha256']==manifest['sha256']['functional_intervention.json'] and
            r['checkpoint_sha256']==receipt['checkpoint_sha256']==sha(checked('evaluated_best.pt'))==CHECKPOINT and
            runtime['physics_device']=='cpu' and runtime['num_envs']==1 and runtime['control_hz']==30 and runtime['physics_hz']==240 and
            len(rows)==r['steps']==r['student_executed_steps'] and rows and not r['cube_pose_writes_during_control'] and
            all(row['step']==i and row['executed_left_source']=='student' for i,row in enumerate(rows)) and len({row['pose_reset_count'] for row in rows})==1):
        raise ValueError('Wrong or incomplete diagnostic trial')
    if sha(checked('physical_metrics_source.py'))!=receipt['metrics_sha256'] or sha((ROOT/'contact_hold_physical_metrics.py').read_bytes())!=receipt['metrics_sha256']:
        raise ValueError('Changed physical metric definition')
    summary=summarize_hold(rows,cfg)
    dst=out/'physical_hold.json'
    with dst.open('x') as f:
        json.dump(dict(kind='functional_hold_diagnostic_only',seed=seed,mode=mode,source_manifest_sha256=sha(m),
                       metrics_sha256=receipt['metrics_sha256'],summary=summary),f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    p.add_argument('--mode',required=True,choices=MODES);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve()
    if (out.parent.parent.parent!=RUNS or not re.fullmatch(r'contact_hold_functional_[A-Za-z0-9_]+',out.parent.parent.name) or
            out.parent.name!=a.mode or out.name!=f'seed_{a.seed}' or out.exists()):
        p.error('Expected a fresh runs/contact_hold_functional_<id>/<mode>/seed_<seed>')
    if a.preflight:
        _,_,rec=prepare(a.mode,a.seed);print(json.dumps(rec,indent=2));return
    if a.native_child:
        simulate(out,a.seed,a.mode);return
    # Isaac's app.close can terminate the interpreter; verify and derive the
    # summary outside the simulator. Inherit any device leases from the scheduler.
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--mode',a.mode,'--seed',str(a.seed),'--output',str(out),'--native-child'],
                   close_fds=False,check=True)
    finalize(out,a.mode,a.seed)


if __name__=='__main__':main()
