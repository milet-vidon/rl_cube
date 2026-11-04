"""Development hold deadband; physical acceptance thresholds are unchanged."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from adaptive_wrist_teacher import AdaptiveWristTeacher
from tilt_regrasp import TiltTeacher

class SettledWristTeacher(AdaptiveWristTeacher):
    correction_deadband_deg=2.5

    def _hold_error_deg(self):
        from isaac_compat import tensor,quaternion_from_native
        state=tensor(self.env.cube.data.root_state_w)[0]
        actual=Rotation.from_quat(quaternion_from_native(state[3:7]).numpy()[[1,2,3,0]])
        desired=Rotation.from_euler('x',-90.,degrees=True)*Rotation.from_quat(self.reference[[1,2,3,0]])
        angle=float(np.degrees((desired*actual.inv()).magnitude()))
        if not np.isfinite(angle):raise ValueError('Nonfinite measured hold error')
        return angle

    def action(self):
        if self.mode!='tilt' or self.tick<len(self.path):return super().action()
        angle=self._hold_error_deg()
        if angle>self.correction_deadband_deg:
            self._deadband_active=False
            return super().action()
        if not getattr(self,'_deadband_active',False):
            self.planner.diagnostics.append({'kind':'hold_correction_deadband','entry_error_deg':angle,'deadband_deg':self.correction_deadband_deg,'step_in_tilt':self.tick})
            self._deadband_active=True
        if not getattr(self,'hold_gains_set',False):
            ids=[self.env.ids[i] for i,n in enumerate(self.env.cfg['joint_names']) if n.startswith('left_')]
            self.env.robot.write_joint_stiffness_to_sim_index(stiffness=torch.full((1,len(ids)),20.,dtype=torch.float32),joint_ids=ids)
            self.hold_gains_set=True
        # Retain the original finger feedback, retention checks, hold timer and timeout.
        result=TiltTeacher.action(self)
        # Preserve PalmFacingTeacher's measured speed rejection after the base action.
        if self.env.metrics.get('linear_speed',0.)>=.03 or self.env.metrics.get('angular_speed',0.)>=.20:
            self.hold=0;self.accepted=False
        return result
