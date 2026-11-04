"""Opt-in coordinate adapter; progress decisions remain in unchanged TurnPacer."""
from turn_feedback_frame import planning_frame_layer_angle


class PlanningFrameFeedback:
    def __init__(self, opposition_request):
        # Capture the ORIGINAL planning Core, before right approach and hold.
        # Do not replace it by turn-start Core or subtract initial face angle.
        self.initial_core_wxyz = tuple(opposition_request['cube_quaternion'])
        planning_frame_layer_angle(self.initial_core_wxyz, self.initial_core_wxyz, 0.)

    def next_index(self, pacer, previous):
        raw_angle = previous['layer_measurement']['face_angle_deg']
        angle = planning_frame_layer_angle(
            self.initial_core_wxyz, previous['cube_quaternion_wxyz'], raw_angle)
        index, failure = pacer.next_index(
            angle, previous['opposition_audit']['right_contacts']['all_five'])
        return index, failure, {
            'previous_step': previous['step'],
            'current_Core_relative_face_angle_deg': raw_angle,
            'planning_frame_layer_angle_deg': angle,
        }

    def report(self):
        return {
            'mode': 'initial_opposition_Core_frame',
            'initial_Core_wxyz': list(self.initial_core_wxyz),
            'initial_Core_source': 'opposition_request.cube_quaternion',
            'mapping': '-log(C_initial.T @ C_current @ Rz(-face_angle)).z in degrees',
            'initial_layer_offset_subtracted': False,
            'gate_changes': None,
            'limitation': 'median twist reconstruction; raw relative speed, off-axis, contacts and final acceptance remain separate',
        }
