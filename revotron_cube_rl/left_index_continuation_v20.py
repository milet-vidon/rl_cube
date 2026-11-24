"""V20 opt-in: extend only the original index MPR lower feedback bound.

The completed center handover remains the frozen V19 center contract. Source
teacher, physical properties, step size, impulse thresholds and gates are intact.
"""
import math

from center_j56_audit import next_index_offset
from left_index_continuation import SETTINGS as CENTER_SETTINGS
from left_index_continuation import initial_state as center_initial_state

SETTINGS = dict(CENTER_SETTINGS, lower_offset_rad=-.023)
SCHEMA = 'left_index_mpr_continuation_extended_lower_bound_v20'


def initial_state(center, final_row, names, applied):
    original = center_initial_state(center, final_row, names, applied)
    return dict(original, settings=dict(SETTINGS), source_center_settings=dict(CENTER_SETTINGS), schema=SCHEMA)


def validate_state(state):
    if state.get('schema') != SCHEMA or state.get('settings') != SETTINGS:
        raise ValueError('V20 requires its explicit frozen support contract')
    if state.get('source_center_settings') != CENTER_SETTINGS:
        raise ValueError('The original center contract must remain unchanged')
    fields = ('reference_target_rad', 'last_applied_offset_rad', 'next_offset_rad',
              'last_nominal_target_rad', 'last_applied_target_rad')
    if not all(math.isfinite(float(state[name])) for name in fields):
        raise ValueError('Nonfinite index handover')
    old, following = float(state['last_applied_offset_rad']), float(state['next_offset_rad'])
    if not CENTER_SETTINGS['lower_offset_rad'] <= old <= 0 or not CENTER_SETTINGS['lower_offset_rad'] <= following <= 0:
        raise ValueError('The handover must be an actual original center offset')
    if abs(following - old) > CENTER_SETTINGS['step_rad'] + 1e-12:
        raise ValueError('Discontinuous center handover')
    if abs(state['reference_target_rad'] + old - state['last_nominal_target_rad']) > 1e-12:
        raise ValueError('Inconsistent original reference')
    if abs(state['last_nominal_target_rad'] - state['last_applied_target_rad']) > 2e-7:
        raise ValueError('Original center command was not applied')


class ContinuedIndexSupport:
    def __init__(self, state):
        validate_state(state)
        self.settings = dict(SETTINGS)
        self.reference = float(state['reference_target_rad'])
        self.offset = float(state['next_offset_rad'])

    @property
    def target(self):
        return self.reference + self.offset

    def update(self, impulse, on_face):
        if not math.isfinite(impulse) or impulse < 0 or not isinstance(on_face, bool):
            raise ValueError('Invalid measured index contact')
        old = self.offset
        self.offset = next_index_offset(old, impulse, on_face, self.settings)
        return dict(applied_offset_rad=old, measured_impulse_Ns=impulse,
                    on_target_face=on_face, next_offset_rad=self.offset,
                    reference_target_rad=self.reference)


def index_envelope_values(initial, state):
    validate_state(state)
    ref = float(state['reference_target_rad'])
    if not math.isfinite(initial):
        raise ValueError('Invalid measured index baseline')
    low, high = min(initial, ref + SETTINGS['lower_offset_rad']), max(initial, ref)
    count = max(1, math.ceil((high - low) / .001))
    return sorted(set([low + (high - low) * i / count for i in range(count + 1)] +
                      [initial, ref + SETTINGS['lower_offset_rad'], ref,
                       ref + state['next_offset_rad'], ref + CENTER_SETTINGS['lower_offset_rad']]))
