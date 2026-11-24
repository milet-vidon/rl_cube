"""Phase-anchored absolute targets avoid accumulating small residual biases.

Fixed phase/progress schedule remains required. Measured state provides bounded
neural feedback, never a reference left trajectory at inference.
"""
import numpy as np
import torch
from torch import nn
from phase_contact_policy import PHASES,FEATURES,accepted_prefix,phase_progress,features,sha
SCHEMA='phase_anchored_absolute28_measured_feedback198_v2'

def pair(previous,current,indices,progress):
    from phase_contact_policy import pair as checked_pair
    x,_=checked_pair(previous,current,indices,progress)
    return x,np.asarray(current['applied_target'],dtype=np.float32)[indices]

class AbsolutePolicy(nn.Module):
    def __init__(self,mean,std,lower,upper,width=512):
        super().__init__()
        for name,value in (('mean',mean),('std',std),('lower',lower),('upper',upper)):self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer('center',(self.lower+self.upper)/2);self.register_buffer('scale',(self.upper-self.lower)/2)
        self.base=nn.Sequential(nn.Linear(21,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,28))
        self.feedback=nn.Sequential(nn.Linear(FEATURES,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,28))
        nn.init.zeros_(self.base[-1].weight);nn.init.zeros_(self.base[-1].bias);nn.init.zeros_(self.feedback[-1].weight);nn.init.zeros_(self.feedback[-1].bias)
    def forward(self,x):
        normalized=((x-self.mean)/self.std).clamp(-15,15).clone();normalized[:,112:168]=0.
        phase=x[:,185:];t=x[:,-1:];freq=torch.tensor([1.,2.,4.,8.],device=x.device)*torch.pi
        anchor=torch.cat((phase,torch.sin(t*freq),torch.cos(t*freq)),1)
        absolute=self.center+self.base(anchor)*self.scale+.01*torch.tanh(self.feedback(normalized))
        return torch.clamp(absolute,self.lower,self.upper)
    def targets(self,x,previous_left):
        return torch.clamp(self(x),torch.maximum(self.lower,previous_left-.04),torch.minimum(self.upper,previous_left+.04))

def restored(saved):
    m=saved['metadata'];s=saved['model'];model=AbsolutePolicy(s['mean'],s['std'],s['lower'],s['upper'],m['width']);model.load_state_dict(s);return model
