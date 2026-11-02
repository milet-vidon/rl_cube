"""Privileged-state BC/PPO policy; this is explicitly not a visual actor."""
import torch
from torch import nn
from tensordict import TensorDict
from rsl_compat import ActorCritic
SCHEMA='state261_absolute56_v1'

def observations(env,initial_xyz):
    raw=env.get_observations()['critic']
    t=env.episode_length_buf.float().reshape(-1,1)/900.
    f=torch.tensor([1.,2.,4.,8.,16.]).reshape(1,-1)*torch.pi
    phase=torch.cat([t,torch.sin(t*f),torch.cos(t*f)],-1)
    x=torch.cat([raw,initial_xyz,phase],-1)
    return TensorDict({'policy':x,'critic':x.clone()},batch_size=[1])

class StateActor(nn.Module):
    def __init__(self,mean,std,center,scale,bounded=False):
        super().__init__()
        self.bounded=bounded
        for name,value in [('mean',mean),('std',std),('center',center),('scale',scale)]:self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(261,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,56))
    def forward(self,x):return self.net(((x-self.mean)/self.std).clamp(-10,10))
    def targets(self,a):return self.center+(torch.tanh(a) if self.bounded else a)*self.scale

class StatePolicy(ActorCritic):
    def __init__(self,obs,mean,std,center,scale,bounded=False):
        super().__init__(obs,{'policy':['policy'],'critic':['critic']},56,[64],[256,128],.01,'log')
        self.actor=StateActor(mean,std,center,scale,bounded)
