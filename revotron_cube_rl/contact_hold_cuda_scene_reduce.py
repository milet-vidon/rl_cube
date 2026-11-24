"""GPU impulse aggregation; Isaac only exposes the ID-to-USD-path API on the host."""
import re
import numpy as np
import torch
import warp as wp

LANE=re.compile(r'^/World/envs/env_(\d+)(/.*)$')
FINGERS=('thumb','index','middle','ring','little')

@wp.kernel
def _reduce(count:wp.array(dtype=wp.int32),start:wp.array(dtype=wp.int32),lanes:wp.array(dtype=wp.int32),
            force:wp.array(dtype=wp.float32),other:wp.array(dtype=wp.int64),keys:wp.array(dtype=wp.int64),
            meta:wp.array2d(dtype=wp.int32),nk:int,capacity:int,kind:int,missing:wp.array(dtype=wp.int64),
            state:wp.array(dtype=wp.int32), contacts:wp.array2d(dtype=wp.float64),
            features:wp.array2d(dtype=wp.float64),hand:wp.array2d(dtype=wp.float64)):
    i=wp.tid()
    ne=count[i]
    if ne==0:
        return
    idx=start[i]
    if ne<0 or idx<0 or idx>capacity-ne:
        wp.atomic_max(state,1,1)
        return
    wp.atomic_add(state,0,ne)
    lane=lanes[i]
    for k in range(idx,idx+ne):
        impulse=wp.abs(force[k])
        if not wp.isfinite(impulse):
            wp.atomic_max(state,2,1)
            continue
        actor=other[k]
        lo=int(0)
        hi=nk
        while lo<hi:
            j=(lo+hi)//2
            if keys[j]<actor:
                lo=j+1
            else:
                hi=j
        if lo>=nk or keys[lo]!=actor:
            pos=wp.atomic_add(state,3,1)
            if pos<capacity:
                missing[pos]=actor
            continue
        if lane!=meta[lo,0]:
            wp.atomic_max(state,4,1)
            continue
        val=wp.float64(impulse)
        if kind==0:
            if meta[lo,1]==1:
                wp.atomic_add(contacts,lane,10,val*wp.float64(0.5))
            if meta[lo,2]==1:
                wp.atomic_add(contacts,lane,11,val)
        else:
            if meta[lo,2]==1:
                wp.atomic_add(contacts,lane,12,val)
            side=meta[lo,3]
            finger=meta[lo,4]
            if side>=0:
                wp.atomic_add(hand,lane,side,val)
            if finger>=0:
                wp.atomic_add(contacts,lane,finger,val)
                if impulse>1.e-7:
                    wp.atomic_add(features,lane,finger,val)

def _wp(t):
    return wp.from_torch(t.contiguous())

class GpuContacts:
    def __init__(self,scene):
        self.views=[(int(kind=='cube'),view,_wp(torch.tensor(lanes,dtype=torch.int32,device='cuda'))) for kind,view,lanes in scene.contact_views]
        self.cache={}
        self.keys=torch.empty(1,device='cuda',dtype=torch.long)
        self.meta=torch.empty((1,5),device='cuda',dtype=torch.int32)
        self.missing=torch.empty(max(v.max_contact_data_count for _,v,_ in self.views),device='cuda',dtype=torch.long)
        self.state=torch.zeros(5,device='cuda',dtype=torch.int32)
        scene.contacts=torch.zeros((scene.n,13),device='cuda',dtype=torch.float64)
        scene.feature_contacts=torch.zeros((scene.n,10),device='cuda',dtype=torch.float64)
        scene.hand_impulse=torch.zeros((scene.n,2),device='cuda',dtype=torch.float64)
        self.acc=[torch.zeros_like(t) for t in (scene.contacts,scene.feature_contacts,scene.hand_impulse)]
        self.stream=wp.stream_from_torch(torch.cuda.current_stream())
    def clear(self,s):
        for t in (s.contacts,s.feature_contacts,s.hand_impulse):
            t.zero_()
    def validate(self):
        pass  # each raw view is already validated prior to committing its reductions
    def resolve(self,s,missing):
        u=torch.unique(self.missing[:missing]).cpu().numpy()
        paths=s.sv_warp_contact.get_other_actor_paths_from_ids(wp.array(u.view(np.uint64),dtype=wp.uint64,device='cpu'))
        if len(paths)!=len(u):raise ValueError('Contact lookup length mismatch')
        for key,path in zip(u,paths):
            m=LANE.match(path)
            if not m:raise ValueError('Unmapped contact actor: '+str(path))
            env,rel=int(m.group(1)),m.group(2)
            robot=rel.startswith('/Robot/');table=rel.startswith('/Table')
            side=next((j for j,(name,end) in enumerate((('left','L'),('right','R'))) if robot and
                       ('/'+name+'_' in rel or rel.endswith('/Link7_'+end))),-1)
            finger=next((j for j,f in enumerate(FINGERS) if side!=-1 and '/'+('left','right')[side]+'_'+f+'_' in rel),-1)
            self.cache[int(key)]=(env,robot,table,side,5*side+finger if finger!=-1 else -1)
        keys=sorted(self.cache)
        self.keys=torch.from_numpy(np.asarray(keys,dtype=np.int64)).cuda()
        self.meta=torch.from_numpy(np.array([self.cache[k] for k in keys],dtype=np.int32)).cuda()
    def update(self,s):
        for kind,view,lanes in self.views:
            force,_,_,_,count,start,other=view.get_raw_contact_data(1.)
            cap=view.max_contact_data_count
            for attempt in range(2):
                for t in self.acc:t.zero_()
                self.state.zero_()
                wp.launch(_reduce,dim=lanes.shape[0],inputs=[_wp(count.reshape(-1).view(torch.int32)),_wp(start.reshape(-1).view(torch.int32)),lanes,
                          _wp(force.reshape(-1)),_wp(other.reshape(-1).view(torch.int64)),_wp(self.keys),_wp(self.meta),len(self.cache),cap,kind,
                          _wp(self.missing),_wp(self.state),*[_wp(t) for t in self.acc]],stream=self.stream,device='cuda:0')
                total,invalid,nonfinite,missing,cross=self.state.tolist()
                if invalid or total>=cap:raise RuntimeError('Contact buffer exhausted')
                if nonfinite:raise ValueError('Nonfinite raw impulse')
                if cross:s.cross+=1;raise ValueError('Cross-lane contact')
                if not missing:break
                if attempt:raise ValueError('Unresolved contact actor')
                self.resolve(s,missing)
            for target,delta in zip((s.contacts,s.feature_contacts,s.hand_impulse),self.acc):target.add_(delta)
