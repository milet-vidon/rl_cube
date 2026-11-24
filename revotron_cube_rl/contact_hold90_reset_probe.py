"""Readback around the real, unmodified 90 mm reset; no DAgger rollouts or fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np


def run(a):
    import torch
    from contact_hold90_cuda_domain import open_scene
    from contact_hold90_cuda_sample import verify_prerequisites
    from contact_hold_loop_data import checkpoint
    out=a.output
    p=json.loads(a.plan.read_bytes());all_seeds=p['seeds']
    original_lanes=list(range(len(all_seeds))) if not a.lanes else [int(v) for v in a.lanes.split(',')]
    if not original_lanes or len(set(original_lanes))!=len(original_lanes) or not all(0<=i<len(all_seeds) for i in original_lanes):raise ValueError('Invalid --lanes')
    seeds=[all_seeds[i] for i in original_lanes]
    verify_prerequisites()
    if p.get('role')!='collection' or checkpoint(Path(p['parent']))!=p['parent_sha']:raise ValueError('Unexpected collection plan or checkpoint')
    s=open_scene(Path(p['scene_source']),len(seeds),Path(p['grasp']))
    def cpu(x):return x.detach().cpu().numpy().copy()
    body_names=['Core',*sorted(n for n in s.views if n!='Core')]
    report=dict(plan=str(a.plan.resolve()),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                domain_sha256=hashlib.sha256(Path(__file__).with_name('contact_hold90_cuda_domain.py').read_bytes()).hexdigest(),
                scene_receipt=s.domain_receipt,solver=s.readback,seeds=seeds,original_lanes=original_lanes,bodies=body_names,frames=[])
    offsets=cpu(s.offset).astype(np.float64)
    blobs={'offsets':offsets}; raw_frames=[]
    sources=[(kind,view,np.asarray(lanes),list(view.sensor_paths)) for kind,view,lanes in s.contact_views]
    def record_contacts(step):
        # Second, read-only raw view after the production reducer, for all contacting lanes.
        # Keep compact typed raw rows. Resolve actor IDs only for diagnostic lanes after reset.
        blobs[f'hand_cumulative_{step}']=cpu(s.hand_impulse)
        blobs[f'contacts_cumulative_{step}']=cpu(s.contacts)
        arrays=[]
        for kind,view,lanes,sensors in sources:
            force,_,_,_,counts,starts,other=view.get_raw_contact_data(1.)
            c=cpu(counts.reshape(-1)).astype(np.int64);st=cpu(starts.reshape(-1)).astype(np.int64)
            hit=np.flatnonzero(c); idx=np.repeat(hit,c[hit])
            if len(idx):
                n=c[hit];within=np.arange(len(idx))-np.repeat(np.cumsum(n)-n,n)
                offsets=np.repeat(st[hit],n)+within
                gather=torch.as_tensor(offsets,device=force.device)
                imp=cpu(force.reshape(-1)[gather]);oid=cpu(other.reshape(-1).view(torch.int64)[gather])
                keep=np.isfinite(imp)&(np.abs(imp)>1e-9)
                arrays.append((idx[keep],oid[keep],np.abs(imp[keep])))
            else:arrays.append((np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64),np.empty(0)))
        raw_frames.append((step,arrays))
    def contact_report():
        import warp as wp
        watched=set(map(int,np.flatnonzero(np.isin(original_lanes,[31,526,643,697,1158,1278]))))
        if a.lanes:watched.update(range(len(seeds)))
        if report['frames']:
            watched.update(t['lane'] for t in report['frames'][-1]['top_by_body_error'])
            poses=blobs[f'world_pose_{count}'][0,:,:3]
            watched.update(np.flatnonzero(np.abs(poses-offsets-blobs['target_local_pose'][:,:3]).max(1)>.015).tolist())
        watched=sorted(watched)
        by_lane={i:[] for i in watched}
        ids=set()
        for step,arrs in raw_frames:
            for (idx,oid,imp),(kind,_,lanes,sensors) in zip(arrs,sources):
                keep=np.isin(lanes[idx],watched)
                for j,key,h in zip(idx[keep],oid[keep],imp[keep]):
                    ids.add(int(key));by_lane[int(lanes[j])].append((step,kind,sensors[j],int(key),float(h)))
        keys=np.asarray(sorted(ids),dtype=np.int64)
        paths=s.sv_warp_contact.get_other_actor_paths_from_ids(wp.array(keys.view(np.uint64),dtype=wp.uint64,device='cpu')) if keys.size else []
        idpath=dict(zip(map(int,keys),map(str,paths)))
        from collections import defaultdict
        for i in watched:
            ag=defaultdict(float)
            for step,k,sen,key,h in by_lane[i]:ag[(step,k,sen,idpath[key])]+=h
            by_lane[i]=dict(seed=seeds[i],original_lane=original_lanes[i],pairs=[dict(step=t,sensor_kind=k,sensor=sen,other=oth,impulse_Ns=h) for (t,k,sen,oth),h in sorted(ag.items())],
                     cumulative_hand_impulse_by_step=[blobs[f'hand_cumulative_{t}'][i].tolist() for t,_ in raw_frames],
                     cumulative_self_robot_table_cube_table_by_step=[blobs[f'contacts_cumulative_{t}'][i,10:13].tolist() for t,_ in raw_frames])
        return by_lane
    def snapshot(step):
        # Pose origins (not COMs) of all 27 cube bodies coincide by design.
        raw=cpu(torch.stack([s.views[k].get_transforms().clone() for k in body_names],dim=0))
        actual=raw[:,:,:3].astype(np.float64)
        goal=np.asarray(s.reset_xyz)[:,:3];target=offsets+goal
        error=actual-target[None,:,:]
        ce=error[0];axis=np.abs(ce).max(axis=0)
        abs_body=np.abs(error).max(axis=(0,2));span=np.ptp(actual,axis=0).max(axis=1)
        robot=cpu(s.robot.get_root_transforms())[:,:3].astype(np.float64)
        # Infer the nearest physical tile from measured xy, without allocating n x n distances.
        side=math.ceil(math.sqrt(len(seeds)))
        tilexy=np.rint((actual[0,:,:2]-goal[:,:2])/4+(side-1)/2)
        near=(tilexy[:,0]+side*tilexy[:,1]).astype(np.int64)
        table=cpu(s.contacts[:,12]) if isinstance(s.contacts,torch.Tensor) else np.asarray(s.contacts)[:,12].copy()
        top=[]
        for i in np.argsort(-abs_body)[:min(8,len(seeds))]:
            i=int(i); b,j=np.unravel_index(np.abs(error[:,i,:]).argmax(),(len(body_names),3))
            top.append(dict(lane=i,seed=seeds[i],offset=offsets[i].tolist(),expected_local=goal[i].tolist(),expected_world=target[i].tolist(),
                core_world=actual[0,i].tolist(),core_local=(actual[0,i]-offsets[i]).tolist(),core_delta_m=ce[i].tolist(),
                core_prim=s.views['Core'].prim_paths[i],nearest_grid_lane=int(near[i]),largest_body=body_names[b],axis='xyz'[j],
                body_delta_m=error[b,i].tolist(),max_body_origin_spread_m=float(span[i]),table_impulse=float(table[i])))
        k=dict(step=step,max_core_axis_error_m=axis.tolist(),core_lanes_over_15mm=int((np.abs(ce).max(1)>.015).sum()),
               any_body_lanes_over_15mm=int((abs_body>.015).sum()),max_interbody_origin_spread_m=float(span.max()),
               wrong_nearest_tile=int((near!=np.arange(len(seeds))).sum()),
               max_robot_root_error_m=float(np.max(np.abs(robot-offsets-[0,0,s.cfg['robot_z']]))),
               no_table_impulse=int((table<=0).sum()),top_by_body_error=top)
        report['frames'].append(k)
        blobs[f'world_pose_{step}']=raw;blobs[f'table_{step}']=table;blobs[f'robot_{step}']=robot
        blobs['target_local_pose']=np.asarray(s.reset_xyz)
        print('RESET_PROBE',step,'core_axis_error',axis.tolist(),'core_fail',k['core_lanes_over_15mm'],'max_body_spread',k['max_interbody_origin_spread_m'],'worst',top[0],flush=True)
    # Observe get_transforms directly; never alter physics commands, initial poses, or thresholds.
    advance=s.sim.step;read_contacts=s.read_contacts;count=0
    def observed(*args,**kwargs):
        nonlocal count
        if count==0:snapshot(0)
        value=advance(*args,**kwargs);count+=1
        return value
    def observed_contacts():
        read_contacts()
        record_contacts(count)
        if count in (1,2,4):snapshot(count)
    s.sim.step=observed;s.read_contacts=observed_contacts
    try:
        s.reset(seeds)
        if not (s.contacts[:,12]>0).all():raise ValueError('Missing reset table impulse')
        report['reset_passed']=True
    except BaseException as e:
        report['reset_passed']=False;report['error']=repr(e)
        raise
    finally:
        s.sim.step=advance;s.read_contacts=read_contacts
        try:
            if getattr(s,'reset_xyz',None) is not None and count:snapshot(count)
            report['steps']=count;report['cross_lane_contacts']=int(s.cross)
            np.savez_compressed(out/'frames.npz',**blobs)
            report['contacts_for_diagnostic_lanes']=contact_report()
            print('RESET_HAND_IMPULSES',[(i,np.round(v['cumulative_hand_impulse_by_step'][-1],7).tolist()) for i,v in report['contacts_for_diagnostic_lanes'].items() if v['cumulative_hand_impulse_by_step']],flush=True)
        finally:
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('PROBE_REPORT',out/'report.json', 'pass',report.get('reset_passed'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--lanes',help='Optional comma-separated original plan lane indices; selected seeds form a smaller grid.')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    from isaaclab.app import AppLauncher
    from isaac_compat import close_application
    import torch
    app=AppLauncher(headless=True,device='cuda:0').app
    try:
        with torch.no_grad():run(a)
    except BaseException:traceback.print_exc();raise
    finally:close_application(app)

if __name__=='__main__':main()
