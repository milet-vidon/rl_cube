"""Paired 90 mm retarget with consistent 57 mm actor input/output coordinates.

For observations taken after command i, express q and target as x - D_i, and
joint velocity as v - (D_i - D_{i-1})/dt, where D is the phase-aligned plan
coordinate offset. The virtual aligned cube center is 16.5 mm lower. The
physical scene, contact, observed orientation, actual state, target projection,
right-hand guide and all pass/fail criteria remain exactly as in the control.
"""
import argparse
from difflib import unified_diff
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch
import contact_hold_size_retarget_worker as parent
from contact_hold_grace_worker import ROOT, RUNS, CHECKPOINT
from contact_hold_size_worker import file_sha
from phase_contact_policy import features

SOURCE='canonical_student_plus_size_plan_delta'
KIND='size90_canonical_observation_bc_and_guided_right_face_development'
PARENT_SHA='52f20cdd820e23cacc7d3b348b2a0a51db82c1f0719d087a49f1219dc1db6230'
CONTROL=RUNS/'contact_hold_size_r15_90mm_retarget_approach_20261112_084150_b'
CTRL_MANIFEST={21611:'37012d1873f3c7578ecf22e83f50813535b0a8286cc7cfd3abdaad4572e77170',21612:'5116e56be6a8af2e947c3af6c54a2e03b430e05ddf6a79e70c5b8dfc69eb4d6a'}
CUBE_Z_SHIFT=.0165


def derive(fn,repl,**bindings):
    if file_sha(ROOT/'contact_hold_size_retarget_worker.py')!=PARENT_SHA:raise ValueError('Changed output-only paired worker')
    orig=inspect.getsource(fn);s=orig
    for before,after in repl:
        if s.count(before)!=1:raise ValueError('Changed parent binding: '+before)
        s=s.replace(before,after)
    ns={};exec(compile(s,'<canonical:'+fn.__name__+'>','exec'),dict(fn.__globals__,**bindings),ns)
    return ns[fn.__name__],''.join(unified_diff(orig.splitlines(True),s.splitlines(True),'output_only','canonical'))


def bindings(proof=None):
    native,diff=derive(parent.simulate,[
        ('action,row=student_action(self,env,phase,step,previous,native)',
         'action,row=canonical_actor(student_action,self,env,phase,step,previous,native,context,out,seed,asset,canonical_proof)'),
        ("raw=cast(row['student_raw_left_target'])", "raw=cast(row['canonical_student_raw_left_target'])")],
        SOURCE=SOURCE,KIND=KIND,canonical_actor=canonical_actor,canonical_proof=proof)
    verify,diff_verify=derive(parent.finalize,[
        ("raw=np.asarray(row['student_raw_left_target'])", "raw=np.asarray(row['canonical_student_raw_left_target'])"),
        ("('unretargeted_student_left_target',plain)","('canonical_unretargeted_student_left_target',plain)"),
        ("per-step frozen output with no residual", "per-step frozen canonical-observation output with no output residual")],
        SOURCE=SOURCE,KIND=KIND)
    return native,verify,{'native':diff,'verifier':diff_verify}


def receipt(seed,asset):
    src=parent.receipt(seed,asset);ctrl=CONTROL/'from_reset'/f'seed_{seed}'
    if file_sha(ctrl/'artifact_manifest.json')!=CTRL_MANIFEST[seed]:raise ValueError('Different paired control')
    for n in ('grasp90.json','geometry_retarget.json'):
        if file_sha(CONTROL/'calibration'/n)!=file_sha(asset.parent.parent/'calibration'/n):raise ValueError('New calibration in paired test')
    for n in ('cube_model.usda','geometry.json'):
        if file_sha(CONTROL/'asset90'/n)!=file_sha(asset.parent/n):raise ValueError('Cube changed in paired test')
    return dict(kind=KIND,seed=seed,executed_left_source=SOURCE,eligible_training_episode=False,diagnostic_only=True,
        checkpoint_sha256=CHECKPOINT,asset_sha256=src['asset_sha256'],grasp_sha256=src['grasp_sha256'],
        paired_control_run=str(ctrl),paired_control_manifest_sha256=CTRL_MANIFEST[seed],parent_worker_sha256=PARENT_SHA,
        worker_sha256=file_sha(__file__),bound_source_diffs=bindings()[2],cube_observation_z_subtracted_m=CUBE_Z_SHIFT,
        input_alignment='all 56 joint positions and previous applied target minus previous nominal offset (90mm minus 57mm), all 56 velocities minus its backward derivative; first frame zeros',
        right_arm='coordinates only: actual right posture deviations and all right contacts retained; no right actions driven by the BC actor',
        unchanged_observation_fields=['cube XY','cube quaternion','raw finger impulses','phase clock'],
        action='canonical actor output plus current left nominal offset; original caps and final physical projection',
        raw_control='untransformed actor evaluated separately on the same real state, diagnostic only')


def as_vec(x,n):
    v=np.asarray(x,dtype=np.float64)
    if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('Invalid observation/reference vector')
    return v


def canonical_observation(raw,dprev,dprevprev,dt):
    dprev=as_vec(dprev,56);dprevprev=as_vec(dprevprev,56)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid control period')
    rate=(dprev-dprevprev)/dt
    # Do not modify the measured dictionary, in particular the list of contacts
    # later consumed by the guidance and physical-hold measurement.
    obs=dict(raw);obs['q']=(as_vec(raw['q'],56)-dprev).tolist();obs['applied_target']=(as_vec(raw['applied_target'],56)-dprev).tolist()
    obs['joint_velocity']=(as_vec(raw['joint_velocity'],56)-rate).tolist()
    pos=as_vec(raw['cube_position'],3).copy();pos[2]-=CUBE_Z_SHIFT;obs['cube_position']=pos.tolist()
    return obs,rate


def first_observation(env):
    from isaac_compat import tensor,quaternion_from_native
    cube=tensor(env.cube.data.root_state_w)[0]
    return dict(q=tensor(env.robot.data.joint_pos)[0,env.ids].tolist(),joint_velocity=tensor(env.robot.data.joint_vel)[0,env.ids].tolist(),
                applied_target=env.targets[0].tolist(),cube_position=cube[:3].tolist(),cube_quaternion_wxyz=quaternion_from_native(cube[3:7]).tolist(),finger_contact_points=[])


def canonical_actor(original,self,env,phase,step,previous,native,context,out,seed,asset,proof):
    # This invocation is the original frozen actor and clock, on untouched observations.
    action,row=original(self,env,phase,step,previous,native)
    if previous is None:
        if step!=0:raise ValueError('Missing previous measurement')
        physical=first_observation(env);prev=np.zeros(56);older=prev.copy()
        parent.dump(out/'canonical_initial_observation.json',physical)
        parent.dump(out/'canonical_intervention.json',proof);shutil.copyfile(__file__,out/'canonical_worker_source.py')
        self.protocol.update(canonical_input_projection=True,canonical_intervention_sha256=file_sha(out/'canonical_intervention.json'))
    else:
        if previous['step']+1!=step:raise ValueError('Observation lag mismatch')
        physical=previous;prev=as_vec(previous['canonical_destination_reference56'],56)-as_vec(previous['canonical_source_reference56'],56)
        older=as_vec(previous['canonical_prev_delta56'],56)
    canonical,rate=canonical_observation(physical,prev,older,env.control_dt)
    clock=row['phase_elapsed']
    with torch.inference_mode():
        can=self.model(torch.from_numpy(features(canonical,phase,clock))[None]);before=env.targets[:,self.left]
        plain=before+(can-before).clamp(-self.cap,self.cap)
    row.update(canonical_student_raw_left_target=can[0].tolist(),canonical_unretargeted_student_left_target=plain[0].tolist(),
               canonical_prev_delta56=prev.tolist(),canonical_prev_velocity_delta56=rate.tolist(),canonical_phase_elapsed=clock,
               canonical_source_reference56=as_vec(context['oldnow'],56).tolist(),canonical_destination_reference56=as_vec(context['newnow'],56).tolist())
    return action,row


def replay(run,seed,asset):
    from contact_hold_policy import restored
    mf=run/'artifact_manifest.json';entries=json.loads(mf.read_bytes())['sha256']
    def checked(n):
        if file_sha(run/n)!=entries[n]:raise ValueError('Changed canonical evidence: '+n)
        return json.loads((run/n).read_bytes())
    proof=checked('canonical_intervention.json');result=checked('result.json');cfg=checked('config.json')
    if not (proof==receipt(seed,asset) and file_sha(run/'canonical_worker_source.py')==entries['canonical_worker_source.py']==proof['worker_sha256']
        and result['canonical_input_projection'] and result['canonical_intervention_sha256']==entries['canonical_intervention.json'] and result['executed_left_source']==SOURCE
        and result['checkpoint_sha256']==file_sha(run/'evaluated_best.pt')==CHECKPOINT):raise ValueError('Invalid canonical run')
    previous=checked('canonical_initial_observation.json');rows=checked('trajectory.json');proj=checked('student_left_projection.json')
    ck=torch.load(run/'evaluated_best.pt',map_location='cpu',weights_only=True);net=restored(ck).eval();I=ck['metadata']['output_indices'];dt=proj['dt']
    if abs(dt*cfg['control_hz']-1)>1e-9 or not rows:raise ValueError('Invalid control sequence')
    cap=np.minimum(proj['step'],np.array(proj['velocity'])*dt);d=np.zeros(56);dd=d.copy();xs=[];ys=[];expected_phys=[];expected_canon=[]
    for i,r in enumerate(rows):
        if r['step']!=i or r['executed_left_source']!=SOURCE or r['canonical_phase_elapsed']!=r['phase_elapsed']:raise ValueError('Unexpected canonical source or phase')
        # Independent reconstruction: go directly from observed features to the
        # canonical feature vector; do not call the runtime observation helper.
        x=features(previous,r['phase'],r['phase_elapsed']);y=x.copy();rate=(d-dd)/dt
        y[0:56]-=d;y[56:112]-=rate;y[112:168]-=d;y[170]-=CUBE_Z_SHIFT
        if not (np.allclose(d,r['canonical_prev_delta56'],rtol=0,atol=1e-7) and np.allclose(rate,r['canonical_prev_velocity_delta56'],rtol=0,atol=1e-7)):
            raise ValueError('Wrong measured reference lag/velocity correction')
        old=as_vec(r['canonical_source_reference56'],56);new=as_vec(r['canonical_destination_reference56'],56)
        if not(np.allclose(old[I],r['retarget_source_left_reference'],atol=1e-7,rtol=0) and np.allclose(new[I],r['retarget_destination_left_reference'],atol=1e-7,rtol=0)):
            raise ValueError('Actor input and output references differ')
        before=np.asarray(previous['applied_target'])[I];expected=before+np.clip(np.asarray(r['student_raw_left_target'])-before,-cap,cap)
        if not np.allclose(expected,r['unretargeted_student_left_target'],rtol=0,atol=5e-6):raise ValueError('Uncorrected actor/control projection mismatch')
        dd,d=d,new-old;previous=r;xs.append(x);ys.append(y);expected_phys.append(r['student_raw_left_target']);expected_canon.append(r['canonical_student_raw_left_target'])
    with torch.inference_mode():
        predict=lambda a:np.concatenate([net(torch.from_numpy(np.stack(a[k:k+128]))).numpy() for k in range(0,len(a),128)])
        rp=predict(xs);rc=predict(ys)
    ep=float(np.max(np.abs(rp-expected_phys)));ec=float(np.max(np.abs(rc-expected_canon)))
    if ep>2e-4 or ec>2e-4:raise ValueError(f'Dual frozen network replay diverged: physical={ep} canonical={ec}')
    summary=dict(kind=KIND,seed=seed,steps=len(rows),checkpoint_sha256=CHECKPOINT,source_manifest_sha256=file_sha(mf),
                 both_network_inputs_replayed_frames=len(rows),max_raw_physical_replay_error_rad=ep,max_canonical_replay_error_rad=ec,
                 max_abs_network_output_difference_rad=float(np.max(np.abs(rp-rc))),
                 frames_with_changed_raw_output=int(np.count_nonzero(np.max(np.abs(rp-rc),axis=1)>1e-5)),
                 output_limits_verified=True,initial_frame_verified=True,diagnostic_only=True)
    parent.dump(run/'canonical_summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=CTRL_MANIFEST,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve();asset=a.asset.resolve();job=out.parent.parent
    if (job.parent!=RUNS or not re.fullmatch(r'contact_hold_size_r15_[A-Za-z0-9_]*canonical[A-Za-z0-9_]*',job.name) or out.exists()
        or out.parent.name!='from_reset' or out.name!=f'seed_{a.seed}' or asset!=job/'asset90/cube_model.usda'):
        p.error('Expected fresh ...size_r15_...canonical.../from_reset/seed_<seed> and sibling asset90/cube_model.usda')
    if a.preflight:print(json.dumps(receipt(a.seed,asset),indent=2));return
    native,verify,_=bindings()
    if a.native_child:
        # Verify and freeze the receipt before the native simulator installs its scoped controller methods.
        native=bindings(receipt(a.seed,asset))[0];native(out,a.seed,asset);return
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--seed',str(a.seed),'--output',str(out),'--asset',str(asset),'--native-child'],check=True,close_fds=False)
    verify(out,a.seed,asset);replay(out,a.seed,asset)


if __name__=='__main__':main()
