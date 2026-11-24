"""Extend collection only with a measured, bounded-memory support-and-turn reward.

This collector runs in an isolated child and leaves the pinned legacy sampler
untouched. Validation always uses its original, more general whole-turn score.
"""
import argparse
import math
from pathlib import Path
from types import FunctionType

import torch
import contact_hold90_cuda_sample as original
from contact_hold_loop_data import read,write
from contact_hold_dataset import sha

DEFINITION=('In a 30-frame window with continuous physical support and <=1 cm palm-relative diameter, '
            'the orientations at offsets 12..14 are each >=15 degrees from all three orientations at offsets 0..2; '
            'the object remains supported through offset 29. Collection-only 1-second window, not the validation evaluator.')

class TurnWindow:
    def __init__(self,n,device):
        self.quats=torch.zeros((30,n,4),device=device,dtype=torch.float64)
        self.valid=torch.zeros((30,n),device=device,dtype=torch.bool)
        self.ever=torch.zeros(n,device=device,dtype=torch.bool)
        self.offsets=torch.tensor([0,1,2,12,13,14],device=device)
        self.limit=math.cos(math.radians(15.)/2)
    def update(self,step,acted,hold,quats):
        norm=quats.double().norm(dim=-1)
        self.valid[step%30].copy_((norm-1).abs()<=.001)
        q=quats.double()/norm.clamp_min(1e-8)[:,None]
        self.quats[step%30].copy_(q)
        if step<29:return
        samples=self.quats[(step-29+self.offsets)%30].transpose(0,1)
        dots=torch.bmm(samples[:,:3],samples[:,3:].transpose(1,2)).abs().amax((1,2))
        self.ever |= acted & hold & self.valid.all(0) & (dots<=self.limit+1e-12)

def sample(args):
    if args.role!='collection':return original.sample(args)
    ext={}
    class Meter(original.HoldMeter):
        def __init__(self,n,dev):
            super().__init__(n,dev);ext['turn']=TurnWindow(n,dev)
        def update(self,step,acted,supported,local):
            super().update(step,acted,supported,local)
            ext['turn'].update(step,acted,self.end,ext['quaternion'])
    def palm(scene):
        read_palm=original.local_palm_reader(scene)
        def measure(core):
            ext['quaternion']=core[:,3:7]
            return read_palm(core)
        return measure
    fn=original.sample
    FunctionType(fn.__code__,fn.__globals__|{'HoldMeter':Meter,'local_palm_reader':palm},fn.__name__)(args)
    report_path=args.output/'report.json';r=read(report_path);flags=ext['turn'].ever.tolist()
    if len(flags)!=len(r['per_lane']):raise ValueError('turn/seed length mismatch')
    for i,(lane,result) in enumerate(zip(r['per_lane'],flags)):
        if lane['lane']!=i or lane['seed']!=r['seeds'][i] or (result and not lane['ever_hold']):raise ValueError('invalid measured turn')
        lane['reward_supported_turn']=result
    r.update(collection_turn_definition=DEFINITION,reward_sampler_sha256=sha(Path(__file__)))
    write(report_path,r)

def main():
    # Reuse pinned entry point including AppLauncher, torch.no_grad, error reporting
    # and close_application; only the sample implementation changes in this child.
    base=original.main
    FunctionType(base.__code__,base.__globals__|{'sample':sample},base.__name__)()

if __name__=='__main__':main()
