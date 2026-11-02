"""RSL-RL ActorCritic with trainable RGB CNN and a privileged state critic."""
from collections import OrderedDict
import torch
from torch import nn
from rsl_compat import ActorCritic

class RGBActor(nn.Module):
    def __init__(self,resolution=128,proprio_dim=224):
        super().__init__();self.resolution=resolution;self.proprio_dim=proprio_dim
        self.register_buffer('proprio_scale',torch.ones(proprio_dim))
        self.cnn=nn.Sequential(nn.Conv2d(3,16,5,2),nn.ELU(),nn.Conv2d(16,32,3,2),nn.ELU(),nn.Conv2d(32,32,3,2),nn.ELU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(512+proprio_dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,56))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        proprio=x[:,:self.proprio_dim]*self.proprio_scale;image=x[:,self.proprio_dim:].reshape(-1,self.resolution,self.resolution,3).permute(0,3,1,2)
        return self.head(torch.cat((proprio,self.cnn(image-.5)),-1))

    @torch.no_grad()
    def set_target_error_scaling(self,action_scales):
        """Normalize target-q while preserving the actor's current function.

        Proprioception precedes CNN features in head[0]. Rescaling columns
        168:224 by old_gain/new_gain cancels the changed input gains. Repeating
        this operation with the same action scales leaves weights unchanged.
        """
        if self.proprio_dim!=224:
            raise ValueError('Target-error scaling requires 224-dimensional proprioception')
        action_scales=torch.as_tensor(action_scales,dtype=self.proprio_scale.dtype,
                                      device=self.proprio_scale.device).reshape(-1)
        if action_scales.numel()!=56 or not torch.isfinite(action_scales).all() or not (action_scales>0).all():
            raise ValueError('Expected 56 finite positive action scales')
        new_gain=action_scales.reciprocal()
        old_gain=self.proprio_scale[168:224].clone()
        if not torch.isfinite(new_gain).all() or not torch.isfinite(old_gain).all() or not (old_gain>0).all():
            raise ValueError('Target-error gains must be finite and positive')
        self.head[0].weight[:,168:224].mul_(old_gain/new_gain)
        self.proprio_scale[168:224].copy_(new_gain)

    @torch.no_grad()
    def disable_action_history(self):
        """Remove previous actions from actor input, deliberately changing its function.

        This is an input ablation, so no compensating weight change is applied.
        The persistent scale buffer carries the choice into inference checkpoints.
        """
        if self.proprio_dim!=224:
            raise ValueError('Disabling action history requires 224-dimensional proprioception')
        self.proprio_scale[112:168].zero_()

class VisionActorCritic(ActorCritic):
    def __init__(self,obs,resolution=128):
        # Initialize the standard critic/distribution with a small dummy actor input,
        # then replace only the actor with a CNN. No object state reaches the actor.
        proprio_dim=obs['policy'].shape[-1]-resolution*resolution*3
        dummy=obs.clone();dummy['policy']=obs['policy'][:,:proprio_dim]
        super().__init__(dummy,{'policy':['policy'],'critic':['critic']},56,actor_hidden_dims=[64],critic_hidden_dims=[256,128],init_noise_std=.25,noise_std_type='log')
        self.actor=RGBActor(resolution,proprio_dim)
        print('Actual image actor:',self.actor)

    def load_state_dict(self,state_dict,strict=True):
        # Historical policies predate this identity-default buffer. Add only
        # that field; all other missing/unexpected keys retain strict handling.
        if 'actor.proprio_scale' not in state_dict:
            compatible=OrderedDict(state_dict)
            if hasattr(state_dict,'_metadata'):
                compatible._metadata=state_dict._metadata
            compatible['actor.proprio_scale']=torch.ones_like(self.actor.proprio_scale)
            state_dict=compatible
        return super().load_state_dict(state_dict,strict=strict)
