"""Supplementary 90 mm probe: wider arm-settling tolerance exclusively before first hand/cube contact.

Never use this probe to label the fixed-control size comparison. The simulated
contact signals, planner, student and physical verification are shared with the
ordinary size worker; the per-boundary ledger records the real controller use.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import contact_hold_size_worker as size
import contact_hold_functional_worker as functional
from contact_hold_grace_worker import ROOT, RUNS

BASE_WORKER='b8d3ffee1fbce07464fd3f6efebfcf996d9747da23290cfbd1c0e1039d1d6c9e'
CONTROL_JOB=RUNS/'contact_hold_size_r15_90mm_20261112_082542'
PHASES=('ready','preshape','lower','approach')
ORIGINAL=.02
PROBE=.08
LEDGER='approach_settle_events.jsonl'
MARKER='approach_intervention.json'


def receipt(seed,asset):
    if size.file_sha(ROOT/'contact_hold_size_worker.py')!=BASE_WORKER:raise ValueError('Different size worker')
    _,s=size.prepare(seed,asset)
    baseline=CONTROL_JOB/size.MODE/f'seed_{seed}'
    mf=baseline/'artifact_manifest.json';hashes=json.loads(mf.read_bytes())['sha256']
    for n in ('size_intervention.json','result.json','cube_asset.json','trajectory.json'):
        if size.file_sha(baseline/n)!=hashes[n]:raise ValueError('Changed 90 mm paired control: '+n)
    ctrl=json.loads((baseline/'result.json').read_bytes())
    if not (ctrl['size_diagnostic'] and ctrl['seed']==seed and ctrl['cube_edge_m']==size.EDGE and
            ctrl['checkpoint_sha256']==size.CHECKPOINT and ctrl['failure']=='contact_topology_or_settle_timeout' and
            json.loads((baseline/'cube_asset.json').read_bytes())['source_sha256']==s['asset_sha256']):
        raise ValueError('Different paired control')
    grasp=ROOT/'opposition_grasp_v20.json'
    if json.loads(grasp.read_bytes())['arm_settle_tolerance_rad']!=ORIGINAL:raise ValueError('Original settling tolerance changed')
    return dict(kind='size90_precontact_arm_settle_probe_v1',seed=seed,base_size_worker_sha256=BASE_WORKER,
        worker_sha256=size.file_sha(__file__),grasp_source_sha256=size.file_sha(grasp),cube_edge_m=size.EDGE,
        checkpoint_sha256=size.CHECKPOINT,asset_sha256=s['asset_sha256'],original_tolerance_rad=ORIGINAL,
        precontact_tolerance_rad=PROBE,unchanged_max_arm_speed_rad_s=.18,phases=list(PHASES),
        precondition='No positive measured left or right hand/cube impulse has occurred. Permanently revert after any measured contact.',
        control_run=str(baseline),control_manifest_sha256=size.file_sha(mf),diagnostic_only=True,eligible_training_episode=False)


def observed_impulses(controller):
    totals=[float(x) for x in controller.env.contact_impulse]
    fingers=[float(x) for side in controller.finger_contacts.values() for x in side.values()]
    if len(totals)!=2 or any(x<0 or not math.isfinite(x) for x in totals+fingers):raise ValueError('Invalid measured hand impulses')
    return totals, max(totals+fingers)>0.


def monitor(pickup, emit):
    from isaac_compat import tensor
    touched=False
    def monitored(self):
        nonlocal touched
        if self.settle_tolerance!=ORIGINAL:raise ValueError('Wrong original tolerance on live controller')
        phase,path=self.phases[self.index];before=self.index
        step=int(self.env.episode_length_buf[0]);totals,contact=observed_impulses(self)
        if contact and not touched:
            emit(dict(kind='first_measured_contact',input_step=step,phase=phase,impulses_Ns=totals))
        touched=touched or contact
        boundary=(not self.failure_reason and self.tick+1>=len(path) and self.index<len(self.phases)-1)
        eligible=bool(boundary and not touched and phase in PHASES)
        # Only this exact comparison in the original controller reads the field.
        # The field is restored before any subsequent call, even on failure.
        if eligible:self.settle_tolerance=PROBE
        try:
            action=pickup(self)
            if boundary:
                arms=[i for i,n in enumerate(self.env.cfg['joint_names']) if n.startswith('Joint')]
                # No finger feedback changes an arm goal in the four allowed phases.
                q=tensor(self.env.robot.data.joint_pos)[:,self.env.ids];v=tensor(self.env.robot.data.joint_vel)[:,self.env.ids]
                err=float((q-self.nominal_target)[:,arms].abs().max());speed=float(v[:,arms].abs().max())
                advanced=self.index!=before
                if eligible and advanced!=(err<PROBE and speed<.18):raise ValueError('Live transition does not agree with scoped tolerance')
                emit(dict(kind='settle_boundary',input_step=step,phase=phase,original_tolerance_rad=ORIGINAL,
                     controller_tolerance_rad=self.settle_tolerance,contact_seen=touched,hand_impulses_Ns=totals,
                     activated=eligible,arm_error_rad=err,max_arm_speed_rad_s=speed,phase_advanced=advanced,
                     transition_required_probe=bool(eligible and advanced and err>=ORIGINAL)))
        finally:self.settle_tolerance=ORIGINAL
        return action
    return monitored


def simulate(out,seed,asset):
    proof=receipt(seed,asset)
    import contact_hold_student_eval as adapter
    original_init=adapter.StudentEvaluation.__init__
    original_sim= functional.simulate
    def student_init(self,*args):
        original_init(self,*args)
        if args[2].resolve()!=out or self.digest!=size.CHECKPOINT:raise ValueError('Wrong probe student')
        marker=out/MARKER
        marker.write_text(json.dumps(proof,indent=2,allow_nan=False))
        shutil.copyfile(__file__,out/'approach_worker_source.py')
        with (out/LEDGER).open('x') as f:f.write('')
        self.protocol.update(approach_tolerance_diagnostic=True,baseline_fixed_approach_control=False,
                             approach_intervention_sha256=size.file_sha(marker))
    def emit(event):
        with (out/LEDGER).open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
        if event.get('transition_required_probe'):print('APPROACH_RELAXED_TRANSITION',event,flush=True)
    def instrument(actual_out,actual_seed,mode):
        if (actual_out,actual_seed,mode)!=(out,seed,size.MODE):raise ValueError('Wrong evaluation binding')
        orig_prepare=functional.prepare
        def wrapped(mode,seed):
            pickup,tilt,base=orig_prepare(mode,seed)
            return monitor(pickup,emit),tilt,base
        with patch.object(functional,'prepare',wrapped):original_sim(out,seed,mode)
    with patch.object(adapter.StudentEvaluation,'__init__',student_init),patch.object(functional,'simulate',instrument):
        size.simulate(out,seed,asset)


def finalize(out,seed,asset):
    size.finalize(out,seed,asset)
    proof=receipt(seed,asset);mf=out/'artifact_manifest.json';hashes=json.loads(mf.read_bytes())['sha256']
    for n in (MARKER,LEDGER,'approach_worker_source.py','trajectory.json','result.json'):
        if size.file_sha(out/n)!=hashes[n]:raise ValueError('Modified supplemental artifact: '+n)
    r=json.loads((out/'result.json').read_bytes())
    if not (proof==json.loads((out/MARKER).read_bytes()) and r['approach_tolerance_diagnostic'] and not r['baseline_fixed_approach_control']
            and hashes[MARKER]==r['approach_intervention_sha256'] and hashes['approach_worker_source.py']==proof['worker_sha256']):
        raise ValueError('Wrong supplemental intervention')
    rows=json.loads((out/'trajectory.json').read_bytes());events=[json.loads(line) for line in (out/LEDGER).read_text().splitlines()]
    boundary=[e for e in events if e['kind']=='settle_boundary'];first=[e for e in events if e['kind']=='first_measured_contact']
    first_post=next((r['step'] for r in rows if max(r['contact_impulse'])>0),None)
    if len(first)>1 or any(e['input_step']>=len(rows) for e in events):raise ValueError('Invalid controller timestamps')
    if first_post is not None and first and first[0]['input_step']>first_post+1 and rows[first_post+1]['phase'] in PHASES:
        raise ValueError('Missed first measured precontact hand impulse')
    if not boundary:raise ValueError('No real settling boundary was evaluated')
    prev_step=-1
    for e in boundary:
        step=e['input_step'];row=rows[step]
        if step<=prev_step or e['phase']!=row['phase']:raise ValueError('Ledger and real trajectory phase disagree')
        prev_step=step
        expected=e['phase'] in PHASES and not e['contact_seen']
        if e['activated']!=expected or e['controller_tolerance_rad']!=(PROBE if expected else ORIGINAL):
            raise ValueError('Tolerance applied outside its declared scope')
        if expected and (any(e['hand_impulses_Ns']) or (first_post is not None and step>first_post)):
            raise ValueError('Tolerance remained enabled after real hand contact')
        if step and e['hand_impulses_Ns']!=rows[step-1]['contact_impulse']:raise ValueError('Sensor input differs from post-step record')
    activ=[e for e in boundary if e['activated']];needed=[e for e in activ if e['transition_required_probe']]
    with (out/'approach_summary.json').open('x') as f:
        json.dump(dict(seed=seed,source_manifest_sha256=size.file_sha(mf),original_tolerance_rad=ORIGINAL,
            precontact_tolerance_rad=PROBE,activated_at_boundaries=len(activ),activated_by_phase={p:sum(e['phase']==p for e in activ) for p in PHASES},
            transitions_requiring_relaxation=needed,first_contact_input=first[0] if first else None,first_poststep_contact=first_post,
            last_phase=rows[-1]['phase'],failure=r['failure'],steps=len(rows),diagnostic_only=True),f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=size.BASE_DIGEST,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve();asset=a.asset.resolve();job=out.parent.parent
    if (job.parent!=RUNS or not re.fullmatch(r'contact_hold_size_r15_[A-Za-z0-9_]*approach[A-Za-z0-9_]*',job.name) or out.exists()
        or out.parent.name!=size.MODE or out.name!=f'seed_{a.seed}' or asset!=job/'asset90/cube_model.usda'):
        p.error('Expected a new supplemental ...size_r15_...approach.../from_reset/seed_<seed> and its own sibling asset90/cube_model.usda')
    if a.preflight:print(json.dumps(receipt(a.seed,asset),indent=2));return
    if a.native_child:simulate(out,a.seed,asset);return
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--seed',str(a.seed),'--output',str(out),
                    '--asset',str(asset),'--native-child'],check=True,close_fds=False)
    finalize(out,a.seed,asset)


if __name__=='__main__':main()
