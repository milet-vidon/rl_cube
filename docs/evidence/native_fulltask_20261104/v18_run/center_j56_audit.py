"""Measure the requested wrist coordination independently of task success."""
import numpy as np


def joint_coordination(q, initial_q, names, protocol):
    indices = [names.index(f'Joint{i}_L') for i in (5, 6, 7)]
    current = np.degrees(np.asarray(q)[indices])
    initial = np.degrees(np.asarray(initial_q)[indices])
    delta = current - initial
    reduction = abs(initial[2]) - abs(current[2])
    required_reduction = min(protocol['minimum_j7_reduction_deg'], abs(initial[2]))
    conditions = {
        'j5_outward': delta[0] >= protocol['minimum_j5_outward_delta_deg'],
        'j6_inward': delta[1] >= protocol['minimum_j6_inward_delta_deg'],
        'j7_less_flexed': reduction >= required_reduction,
        'j7_moderate_flexion': abs(current[2]) <= protocol['maximum_j7_abs_deg'],
    }
    return {
        'qualified': bool(all(conditions.values())),
        'failed_conditions': [name for name, good in conditions.items() if not good],
        'initial_j5_j6_j7_deg': initial.tolist(),
        'actual_j5_j6_j7_deg': current.tolist(),
        'delta_j5_j6_j7_deg': delta.tolist(),
        'j7_flexion_reduction_deg': float(reduction),
    }


def next_index_offset(offset, impulse, on_face, settings):
    """Bounded contact feedback; high contact force always takes precedence."""
    if impulse > settings['maximum_impulse_Ns']:
        offset += settings['step_rad']
    elif not on_face or impulse < settings['minimum_impulse_Ns']:
        offset -= settings['step_rad']
    return float(np.clip(offset, settings['lower_offset_rad'], settings['upper_offset_rad']))
