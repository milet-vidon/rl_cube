"""Replay an archived measured trajectory through scalar and batched teachers.

Checks phase/terminal/masked lane behavior, per-step commands and the sparse PD
gain schedule, including post-lift planning and the post-turn hold. No simulation,
data collection, training, or changes to the archived run are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import numpy as np
import torch
from contact_hold_student_eval import StudentEvaluation
from contact_hold_cuda_teacher import BatchedTeacher, FAILURES
from phase_contact_policy import PHASES, FINGERS

def check(run,device,out=None):
    from isaac_compat import quaternion_to_native
    if device=='cpu':torch.set_num_threads(1)
    cfg=json.loads((run/'config.json').read_text());candidate=json.loads((run/'contact_hold_candidate.json').read_text())
    rows=[json.loads(l) for l in (run/'trajectory.jsonl').open()]
    active=torch.tensor([True,False],device=device); n=len(active)
    cpu_gains=[];batch_gains=[]
    def writer(lanes,prop,idx,values):
        assert lanes.tolist()==[0]
        batch_gains.append((step,prop,list(idx),values[0].cpu().numpy()))
    def scalar(prop,*,joint_ids,**kwargs):
        cpu_gains.append((step,prop,list(joint_ids),next(iter(kwargs.values()))[0].numpy()))
    def core(row):
        q=row['cube_quaternion_wxyz']
        return torch.tensor([row['cube_position']+[q[j] for j in (1,2,3,0)]],dtype=torch.float32)
    first=core(rows[0]); old=first.clone();old[:,3:7]=quaternion_to_native(first[:,[6,3,4,5]])
    with TemporaryDirectory() as tmp:
        robot=NS(data=NS(),write_joint_stiffness_to_sim_index=lambda **kw:scalar('stiffnesses',**kw),
                 write_joint_damping_to_sim_index=lambda **kw:scalar('dampings',**kw))
        env=NS(run_dir=Path(tmp),cfg=cfg,ids=list(range(56)),robot=robot,cube=NS(data=NS(root_state_w=old,root_pos_w=old[:,:3])),
               scales=torch.ones(1,56),targets=torch.tensor([rows[0]['applied_target']]),contact_impulse=[0.,0.],metrics={})
        scalar_teacher=StudentEvaluation.teacher_class(NS(candidate=candidate))(env,Path(__file__).with_name('opposition_grasp_v20.json'))
        paths=scalar_teacher.base.phases[:6]
        seq=[torch.from_numpy(x).to(device) for _,x in paths];length=torch.tensor([len(x) for x in seq],device=device)
        table=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True)
        class Plan:
            accepted=active.clone()
            def goal(self,phase,tick):
                sizes=length[phase];return table[phase,torch.minimum(tick,sizes-1)],sizes
        teacher=BatchedTeacher(Plan(),cfg,candidate,gain_writer=writer)
        worst=0.;err_by_phase={};steps={};accepted_step=None
        for step,r in enumerate(rows):
            prev=rows[max(0,step-1)];cs=core(prev);v=torch.tensor([prev['joint_velocity']]);q=torch.tensor([prev['q']]);tar=torch.tensor([prev['applied_target']])
            raw=cs.clone();raw[:,3:7]=quaternion_to_native(cs[:,[6,3,4,5]])
            env.robot.data.joint_pos=q;env.robot.data.joint_vel=v;env.targets=tar
            env.cube.data.root_state_w=raw;env.cube.data.root_pos_w=raw[:,:3]
            env.metrics={k:prev[k] for k in ('height_m','cube_table_contact','linear_speed','angular_speed')}
            env.contact_impulse=prev['contact_impulse'];scalar_teacher.finger_contacts=prev['finger_contact_impulses']
            c=torch.zeros(1,13)
            for i,side in enumerate(('left','right')):
                c[0,5*i:5*i+5]=torch.tensor([prev['finger_contact_impulses'][side].get(f,0.) for f in FINGERS])
            c[0,12]=prev['table_impulse']
            vel=torch.tensor([[prev['linear_speed'],0.,0.,prev['angular_speed'],0.,0.]])
            cast=lambda z:z.to(device).expand(n,-1)
            expected=env.targets+scalar_teacher.action()*env.scales
            cmd,ph,fail,accepted=teacher.action(active,*map(cast,(q,v,cs,tar,c,vel)))
            phase=PHASES[int(ph[0])]
            if phase!=scalar_teacher.name or bool(accepted[0])!=scalar_teacher.accepted or (FAILURES[int(fail[0])] or None)!=scalar_teacher.failure_reason:
                raise AssertionError((step,phase,scalar_teacher.name,FAILURES[int(fail[0])],scalar_teacher.failure_reason,int(teacher.tick[0])))
            err=float((cmd[0].cpu()-expected[0]).abs().max())
            worst=max(worst,err);err_by_phase[phase]=max(err,err_by_phase.get(phase,0.));steps[phase]=steps.get(phase,0)+1
            torch.testing.assert_close(cmd[1],cast(tar)[1],atol=0,rtol=0)
            if accepted[0]:accepted_step=step;break
        assert len(cpu_gains)==len(batch_gains)
        for a,b in zip(cpu_gains,batch_gains):
            assert a[:3]==b[:3]
            np.testing.assert_allclose(a[3],b[3],atol=2e-6)
        root=Path(__file__).resolve().parent
        sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        provenance=('contact_hold_cuda_teacher_check.py','contact_hold_cuda_teacher.py','contact_hold_cuda_teacher_math.py',
                    'contact_hold_cuda_teacher_turn.py','kinematics.py','opposition_feedback.py','tilt_regrasp.py',
                    'contact_hold_teacher.py','settled_wrist_teacher.py','adaptive_wrist_teacher.py','elbow_handover.py',
                    'wrist_direction.py','palm_handover.py','bounded_wrist_recovery.py','opposition_teacher.py')
        report=dict(kind='fixed_measured_trace_scalar_vs_batched_teacher_v1',device=device,torch_version=torch.__version__,
                    trace=str((run/'trajectory.jsonl').resolve()),trajectory_sha256=sha(run/'trajectory.jsonl'),
                    source_sha256={p:sha(root/p) for p in provenance},inputs_sha256={p:sha(run/p) for p in ('config.json','opposition_grasp_v20.json','contact_hold_candidate.json')},
                    steps=steps,accepted_step=accepted_step,worst_target_rad=worst,max_target_error_by_phase=err_by_phase,
                    gain_writes=len(cpu_gains),inactive_lane_target_unchanged=True,phase_terminal_gain_checks_passed=True)
        # 0.0005 rad is 10% of the scalar teacher's smallest finger target slew.
        if worst>.0005:raise AssertionError(f'Command error {worst} rad exceeds replay tolerance')
        print(json.dumps(report,allow_nan=False))
        if out is not None:
            with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    ap.add_argument('--output',type=Path);args=ap.parse_args();check(args.run,args.device,args.output)
