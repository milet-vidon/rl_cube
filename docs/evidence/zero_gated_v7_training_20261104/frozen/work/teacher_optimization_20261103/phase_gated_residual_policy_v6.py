"""Public-phase gate over V5's actual-state GRU; all history still advances.

The original squeeze path has 73 targets. FingerArrival is first used at
tick 72; the public source-clock squeeze duration is 85 executed frames.
The adapter reports elapsed phase actions/(duration-1), so first activation
is 72/84; subsequent waiting frames keep the residual active.
No teacher/arrival hidden object or execution-source identity enters this gate.
"""
from copy import deepcopy
import numpy as np
import torch
from phase_contact_policy import PHASES
from phase_residual_policy_v5 import PhaseResidualPolicy,FEATURES,FEATURE_SCHEMA,EpisodeActor,features,project_left_targets

SCHEMA='public_phase_gated_frozen_v4_actual_state_gru_left28_residual_v6'
GATE_SCHEMA='public_phase_no_feedback_before_squeeze_target_72_of_85_clock_v1'
ZERO_PHASES=('ready','preshape','lower','approach')
GATE_CONTRACT={
    'schema':GATE_SCHEMA,'always_zero_phases':list(ZERO_PHASES),
    'squeeze_feedback_first_target_index':72,'squeeze_source_clock_frames':85,
    'squeeze_active_progress_numerator':72,'squeeze_active_progress_denominator':84,
    'squeeze_active_progress_float32':float(np.float32(72/84)),
    'phase_feature_range':[185,197],'progress_feature_index':197,
    'gru_updates_on_every_frame':True,'hidden_arrival_state_is_actor_input':False,
    'original_teacher_source_sha256':'a9fcfbdf084b7a34fbc2c0ef9b6e6134df0f517d4d68f8742138a3300b3a79f1',
    'source_condition':'opposition_teacher.py:156: squeeze and self.tick>=len(path)-1',
    'public_clock_condition':'left_dagger_baseline_runtime_v3.py:schedule_resolver: elapsed/(duration-1)',
}
EXPECTED_LEFT_NAMES=tuple(['Joint'+str(i)+'_L'for i in range(1,8)]+[
    'left_middle_MPR_joint','left_middle_MCP_joint','left_middle_PIP_joint','left_middle_DIP_joint',
    'left_thumb_CMP_joint','left_thumb_CMR_joint','left_thumb_MCP_joint','left_thumb_PIP_joint','left_thumb_DIP_joint',
    'left_index_MPR_joint','left_index_MCP_joint','left_index_PIP_joint','left_index_DIP_joint',
    'left_ring_MPR_joint','left_ring_MCP_joint','left_ring_PIP_joint','left_ring_DIP_joint',
    'left_little_MPR_joint','left_little_MCP_joint','left_little_PIP_joint','left_little_DIP_joint'])
RECOVERY_JOINTS={
    'left_middle_MCP_joint','left_middle_PIP_joint','left_middle_DIP_joint',
    'left_index_MPR_joint','left_index_MCP_joint','left_index_PIP_joint','left_index_DIP_joint',
    'left_ring_MPR_joint','left_ring_MCP_joint','left_ring_PIP_joint','left_ring_DIP_joint'}

def candidate_bounds(left_names):
    if len(left_names)!=28 or len(set(left_names))!=28 or set(left_names)!=set(EXPECTED_LEFT_NAMES):
        raise ValueError('V6 candidate bounds require the actual 28 named left DOFs')
    return [.0005 if name.startswith('Joint')else .006 if name in RECOVERY_JOINTS else .000001 for name in left_names]

class PhaseGatedResidualPolicy(PhaseResidualPolicy):
    def __init__(self,baseline_saved,mean,std,residual_bound,hidden_size=64,gate_contract=None):
        if gate_contract is not None and gate_contract!=GATE_CONTRACT:
            raise ValueError('V6 public phase gate differs from the reviewed source boundary')
        super().__init__(baseline_saved,mean,std,residual_bound,hidden_size)
        active=torch.tensor([phase not in ZERO_PHASES and phase!='squeeze'for phase in PHASES],dtype=torch.bool)
        self.register_buffer('residual_phase_active',active)
        self.register_buffer('squeeze_active_progress',torch.tensor(GATE_CONTRACT['squeeze_active_progress_float32'],dtype=torch.float32))
        self.gate_contract=deepcopy(GATE_CONTRACT)

    def residual_active(self,x):
        phase=x[...,185:197].argmax(-1)
        active=self.residual_phase_active[phase]
        return torch.where(phase==PHASES.index('squeeze'),x[...,197]>=self.squeeze_active_progress,active)

    def effective_bounds(self,x):
        return self.residual_bound*self.residual_active(x).unsqueeze(-1)

    def forward(self,x,previous_left,hidden=None):
        if x.ndim!=3 or x.shape[-1]!=FEATURES or previous_left.shape!=(*x.shape[:2],28):
            raise ValueError('Expected batch/time/features and actual previous left commands')
        if not torch.isfinite(x).all()or not torch.isfinite(previous_left).all():
            raise ValueError('Nonfinite inference state')
        encoded=self.encoder(((x-self.mean)/self.std).clamp(-12.,12.))
        history,next_hidden=self.gru(encoded,hidden)
        residual=torch.tanh(self.head(history))*self.effective_bounds(x)
        base=self.baseline_targets(x,previous_left)
        proposed=torch.clamp(base+residual,self.baseline.lower.float(),self.baseline.upper.float())
        proposed=torch.where(self.residual_active(x).unsqueeze(-1),proposed,base)
        return proposed,next_hidden,residual
