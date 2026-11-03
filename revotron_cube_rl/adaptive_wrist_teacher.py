"""Development wrist redundancy search, preserving palm pose and original gates."""
import numpy as np
from wrist_direction import WristDirectionPlanner,WristDirectionTeacher
from bounded_wrist_recovery import recover_bounded_pose

class AdaptiveWristPlanner(WristDirectionPlanner):
    def fit_tilt(self,q,side,position,rotation):
        try:return super().fit_tilt(q,side,position,rotation)
        except ValueError as original:
            if side!='left' or self.wrist_goal is None or self.wrist_goal<np.radians(7.):raise
            goal=self.wrist_goal
            try:
                for offset in (-2.,2.,-4.,4.,-6.):
                    candidate=goal+np.radians(offset)
                    if not np.radians(7.)<=candidate<=np.radians(16.):continue
                    self.wrist_goal=candidate
                    try:
                        result=super().fit_tilt(q,side,position,rotation)
                    except ValueError:continue
                    frames=self.k.fk(result);s,e,w=[frames[n][:3,3] for n in ('Link2_L','Link4_L','Link7_L')]
                    direction=frames['left_palm_link'][:3,:3].T@((w-e)/np.linalg.norm(w-e))
                    if (result['Joint6_L']<np.radians(5.) or np.degrees(np.arctan2(direction[1],direction[2]))>-5.
                        or direction[2]<=.3 or e[1]-w[1]<.05 or e[1]-s[1]<.16):continue
                    self.diagnostics.append({'kind':'wrist_redundancy_recovery','original_goal_deg':float(np.degrees(goal)),
                                             'selected_goal_deg':float(np.degrees(candidate)),'original_error':str(original)})
                    print('WRIST_REDUNDANCY_RECOVERY',float(np.degrees(goal)),float(np.degrees(candidate)),flush=True)
                    return result
                return recover_bounded_pose(self,q,position,rotation,original)
            finally:self.wrist_goal=goal

class AdaptiveWristTeacher(WristDirectionTeacher):
    planner_type=AdaptiveWristPlanner
