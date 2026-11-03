"""Continue the original centered index feedback without doubling its offset."""
import math

from center_j56_audit import next_index_offset

SETTINGS = dict(joint='left_index_MPR_joint', lower_offset_rad=-.012,
                upper_offset_rad=0., step_rad=.00005,
                minimum_impulse_Ns=.0005, maximum_impulse_Ns=.002)


def initial_state(center, final_row, names, applied):
    settings = dict(center['finger_contact_feedback'])
    if settings != SETTINGS:
        raise ValueError('Continuation requires the unchanged center v3 feedback')
    index = names.index(settings['joint'])
    last = final_row['index_contact_feedback']
    nominal = float(final_row['nominal_target'][index])
    current = float(applied[index])
    old, following = float(last['offset_rad']), float(last['next_offset_rad'])
    if not all(math.isfinite(x) for x in (nominal, current, old, following)):
        raise ValueError('Nonfinite feedback handover')
    if abs(current - nominal) > 2e-7:
        raise ValueError('Applied index target differs from the completed center command')
    if not SETTINGS['lower_offset_rad'] <= old <= 0 or not SETTINGS['lower_offset_rad'] <= following <= 0:
        raise ValueError('Handover offset outside original bounds')
    if abs(following - old) > SETTINGS['step_rad'] + 1e-12:
        raise ValueError('Discontinuous feedback handover')
    return dict(settings=settings, reference_target_rad=nominal-old,
                last_applied_offset_rad=old, next_offset_rad=following,
                last_nominal_target_rad=nominal, last_applied_target_rad=current,
                reference_basis='last nominal center target minus its recorded applied offset')


class ContinuedIndexSupport:
    def __init__(self, state):
        self.settings = dict(state['settings'])
        if self.settings != SETTINGS:
            raise ValueError('Feedback settings changed')
        self.reference = float(state['reference_target_rad'])
        self.offset = float(state['next_offset_rad'])
        if not math.isfinite(self.reference) or not SETTINGS['lower_offset_rad'] <= self.offset <= 0:
            raise ValueError('Invalid continuation state')

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
    ref = float(state['reference_target_rad'])
    if state['settings'] != SETTINGS or not all(math.isfinite(x) for x in (initial, ref)):
        raise ValueError('Invalid index envelope')
    low, high = min(initial, ref-.012), max(initial, ref)
    count = max(1, math.ceil((high-low)/.001))
    return sorted(set([low+(high-low)*i/count for i in range(count+1)] +
                      [initial, ref-.012, ref, ref+state['next_offset_rad']]))
