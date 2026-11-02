"""Contact-supervised execution of supplied joint waypoints; contains no IK or teacher targets."""
import numpy as np
import torch
from isaac_compat import tensor

class SegmentController:
    def __init__(self,env,segments):
        self.env=env;self.segments=segments
        self.index=0;self.tick=0;self.name='ready';self.failure_reason=None;self.wait=0;self.contact_streak=0;self.lost=0

    def action(self):
        phase,begin,goal,steps=self.segments[self.index];self.name=phase
        if self.failure_reason:return torch.zeros_like(self.env.targets)
        self.tick+=1;u=min(self.tick/steps,1.);u=u*u*(3-2*u)
        desired=torch.from_numpy(((1-u)*begin+u*goal).astype(np.float32)).reshape(1,56)
        metrics=getattr(self.env,'metrics',{})
        both=all(x>self.env.cfg['success']['contact_impulse_min'] for x in self.env.contact_impulse)
        self.contact_streak=self.contact_streak+1 if both else 0
        if phase in ('lift','hold'):
            self.lost=0 if both else self.lost+1
            if self.lost>=10:self.failure_reason='lift_contact_lost';return torch.zeros_like(self.env.targets)
        if self.tick>=steps and self.index<len(self.segments)-1:
            arm=[i for i,n in enumerate(self.env.cfg['joint_names']) if n.startswith('Joint')]
            q=tensor(self.env.robot.data.joint_pos)[:,self.env.ids];vel=tensor(self.env.robot.data.joint_vel)[:,self.env.ids]
            settled=float((q-desired)[:,arm].abs().max())<.035 and float(vel[:,arm].abs().max())<.18
            next_phase=self.segments[self.index+1][0]
            ready=(settled if phase!=next_phase and next_phase in ('preshape','lower','lift') else True)
            if phase=='contact' and next_phase=='lift':ready=ready and self.contact_streak>=10
            if ready:self.index+=1;self.tick=0;self.wait=0
            else:
                self.wait+=1;self.tick=steps-1
                if self.wait>120:self.failure_reason='settle_or_contact_timeout'
        return (desired-self.env.targets)/self.env.scales
