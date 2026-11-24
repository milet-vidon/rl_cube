"""Targeted CPU check: mixed-phase batches are independent one-lane teachers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from phase_contact_policy import FINGERS
from contact_hold_cuda_teacher import BatchedTeacher

ROOT=Path(__file__).resolve().parent


def audit(run):
    torch.set_num_threads(1)
    rows={}
    for line in (run/'trajectory.jsonl').open():
        r=json.loads(line)
        if r['step'] in (460,510,540,570,667,668):rows[r['step']]=r
    template=[rows[j] for j in (668,540,668,460,570,668)]
    cfg=json.loads((run/'config.json').read_text());candidate=json.loads((run/'contact_hold_candidate.json').read_text())
    n=len(template)
    def stack(key):return torch.tensor([r[key] for r in template],dtype=torch.float32)
    core=torch.cat([stack('cube_position'),stack('cube_quaternion_wxyz')[:,[1,2,3,0]]],1)
    for i in (0,2,5):
        r=rows[667]
        core[i,:3]=torch.tensor(r['cube_position']);core[i,3:7]=torch.tensor(r['cube_quaternion_wxyz'])[[1,2,3,0]]
    # Three simultaneous handoffs, with genuinely different offline object poses.
    core[2,:2]+=torch.tensor([.001,-.0008]);core[5,:2]+=torch.tensor([-.0006,.0009])
    angle=Rotation.from_euler('z',1.8,degrees=True)*Rotation.from_quat(core[2,3:].numpy());core[2,3:]=torch.tensor(angle.as_quat())
    nom=stack('nominal_target');nom[5,0]+=.0005
    q=nom.clone();q[1]=stack('q')[1];v=torch.zeros_like(q);target=stack('applied_target')
    active=torch.tensor([1,1,1,1,0,1],dtype=torch.bool)
    vel=torch.zeros(n,6)
    contact=torch.zeros(n,13);contact[:,:5]=.002
    # Unsettled squeeze lane: maintains independent missing-pad correction.
    contact[1,:5]=torch.tensor([0.,.002,0.,.001,0.])
    phase=torch.tensor([5,4,5,0,5,5]);tick=torch.tensor([10,10,10,3,0,10]);streak=torch.tensor([9,0,9,0,0,9])
    def teacher(ids):
        ix=torch.tensor(ids)
        class Plan:
            accepted=active[ix]
            def goal(self,ph,t):
                return nom[ix].clone(),torch.full_like(t,11)
        gains=[]
        def write(lanes,prop,joints,values):
            for lane,val in zip(lanes,values):gains.append((step,ids[int(lane)],prop,joints,val.tolist()))
        t=BatchedTeacher(Plan(),cfg,candidate,write)
        t.phase=phase[ix];t.tick=tick[ix];t.streak=streak[ix]
        return t,gains
    batched,bg=teacher(list(range(n)));ind=[teacher([j]) for j in range(n)]
    worst=worst_path=0.; corrected=set(); seen=[set() for _ in range(n)]
    for step in range(7):
        if step==1:
            assert batched.phase.tolist()==[6,4,6,0,5,6]
            # Compare every post-lift waypoint, not just the next output.
            for j in (0,2,5):
                size=int(batched.turn_lens[j].sum())
                torch.testing.assert_close(batched.turn_lens[j],ind[j][0].turn_lens[0])
                difference=(batched.path[j,:size]-ind[j][0].path[0,:size]).abs().max().item()
                assert difference<5.e-5,(j,difference)
                worst_path=max(worst_path,difference)
        if step==2:
            for j,offset in ((0,0),(2,1),(5,-1)):
                batched.tick[j]=batched.turn_lens[j].sum()+offset
                ind[j][0].tick[:]=ind[j][0].turn_lens.sum()+offset
                orient=Rotation.from_euler('x',-86,degrees=True)*Rotation.from_quat(core[j,3:].numpy())
                core[j,3:]=torch.tensor(orient.as_quat())
            # Distinct missing fingers on different live lanes; one sees table contact.
            contact[0,2]=0; contact[2,0]=0;contact[5,1]=0;contact[2,12]=.01
        if step==3:
            contact[0,2]=.003;contact[2,0]=.001;contact[5,3]=0
        args=(active,q,v,core,target,contact,vel)
        group=batched.action(*args)
        for j,(t,gains) in enumerate(ind):
            ref=t.action(*(x[j:j+1] for x in args))
            for b,s in zip(group[1:],ref[1:]):assert b[j].item()==s.item()
            difference=(group[0][j]-ref[0][0]).abs().max().item()
            assert difference<5.e-5,(step,j,difference)
            worst=max(worst,difference);seen[j].add(group[1][j].item())
            for key in ('tick','phase','streak','wait','hold','lost','failure','accepted','corrected'):
                assert getattr(batched,key)[j].item()==getattr(t,key).item(),(key,j)
            if batched.corrected[j]:corrected.add(j)
    def ordered(events):return sorted(events,key=lambda v:v[:3])
    assert ordered(bg)==ordered([v for _,g in ind for v in g])
    assert corrected=={0,5},corrected
    assert len(bg)>5
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(schema='mixed_phase_multi_lane_isolation_check_v1',test='CPU tensor implementation versus isolated one-lane calls',
                active_lanes=int(active.sum()),inactive_lanes=1,simultaneous_distinct_turn_plans=3,corrected_lanes=sorted(corrected),
                gain_writes=len(bg),phases_by_lane=[sorted(s) for s in seen],max_target_difference_rad=worst,
                max_postlift_waypoint_difference_rad=worst_path,source_sha256={n:sha(ROOT/n) for n in
                ('contact_hold_cuda_teacher_isolation.py','contact_hold_cuda_teacher.py','contact_hold_cuda_teacher_math.py','contact_hold_cuda_teacher_turn.py')},
                trajectory_sha256=sha(run/'trajectory.jsonl'),trace=str((run/'trajectory.jsonl').resolve()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.run);print(json.dumps(result,allow_nan=False))
    if a.output:
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
