"""Baseline-only learned phase trajectory; no adaptive state-feedback claim.

Each phase has its own output head over a shared Fourier time basis. All left
targets are inferred from learned weights, not fetched from reference targets.
"""
import numpy as np
import torch
from torch import nn
from phase_contact_policy import PHASES,FEATURES,accepted_prefix,phase_progress,features,sha
SCHEMA='phase_specific_absolute_left28_no_state_correction_v3'

def pair(previous,current,indices,progress):
    from phase_contact_policy import pair as checked_pair
    x,_=checked_pair(previous,current,indices,progress)
    return x,np.asarray(current['applied_target'],dtype=np.float32)[indices]

class PhaseExpertPolicy(nn.Module):
    def __init__(self,mean,std,lower,upper,width=512):
        super().__init__()
        for name,value in (('mean',mean),('std',std),('lower',lower),('upper',upper)):self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer('center',((self.lower+self.upper)/2).repeat(12,1));self.register_buffer('scale',((self.upper-self.lower)/2).repeat(12,1));self.register_buffer('frequency',torch.arange(1,17,dtype=torch.float32)*torch.pi)
        self.net=nn.Sequential(nn.Linear(33,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,12*28))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        phase=x[:,185:197].argmax(1);t=x[:,-1:];basis=torch.cat((t,torch.sin(t*self.frequency),torch.cos(t*self.frequency)),1)
        heads=self.net(basis).reshape(-1,12,28);output=heads[torch.arange(len(x),device=x.device),phase]
        return torch.clamp(self.center[phase]+output*self.scale[phase],self.lower,self.upper)
    def targets(self,x,previous_left):
        return torch.clamp(self(x),torch.maximum(self.lower,previous_left-.04),torch.minimum(self.upper,previous_left+.04))

def restored(saved):
    m=saved['metadata'];s=saved['model'];model=PhaseExpertPolicy(s['mean'],s['std'],s['lower'],s['upper'],m['width']);model.load_state_dict(s);return model
