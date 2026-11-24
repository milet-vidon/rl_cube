"""Bound progress along an already checked turn path using measured layer motion."""
import math

class TurnPacer:
    def __init__(self, angles, max_lead_deg=.75, stall_ticks=90, extra_ticks=360):
        self.angles=[float(x) for x in angles]
        if not self.angles or not all(math.isfinite(x) for x in self.angles):
            raise ValueError('Expected a finite turn path')
        if any(b<a-1e-3 for a,b in zip(self.angles,self.angles[1:])):
            raise ValueError('Only a monotone positive turn path is supported')
        if not math.isfinite(max_lead_deg) or max_lead_deg<=0 or stall_ticks<1 or extra_ticks<1:
            raise ValueError('Invalid pacing bounds')
        self.max_lead_deg=max_lead_deg;self.stall_ticks=stall_ticks
        self.budget=len(self.angles)+extra_ticks;self.cursor=0;self.last_index=0
        self.ticks=0;self.issued_ticks=0;self.stalled=0;self.held_ticks=0;self.completed=False

    def next_index(self, measured_angle, all_five):
        if not math.isfinite(measured_angle):return None,'nonfinite_layer_feedback'
        if self.cursor==len(self.angles):
            self.completed=True
            return None,None
        self.ticks+=1
        if self.ticks>self.budget:return None,'turn_pacing_step_budget'
        if all_five and self.angles[self.cursor]-measured_angle<=self.max_lead_deg:
            index=self.cursor;self.cursor+=1;self.last_index=index;self.stalled=0
            self.issued_ticks+=1
            return index,None
        self.stalled+=1
        if self.stalled>self.stall_ticks:return None,'turn_contact_or_progress_stalled'
        # Repeat the last checked joint target; never invent a correction pose.
        self.held_ticks+=1;self.issued_ticks+=1
        return self.last_index,None

    def report(self):
        return {'max_lead_deg':self.max_lead_deg,'stall_ticks':self.stall_ticks,
                'maximum_control_ticks':self.budget,'control_ticks':self.issued_ticks,
                'held_ticks':self.held_ticks,'completed':self.completed,
                'scope':'time pacing of existing collision-checked targets; unchanged acceptance gates'}
