"""Finite-horizon pickup gates and discounted potential shaping, without Isaac.

``terms['potential']`` is the next step's ``previous_potential``. On a true
terminal transition or a finite-horizon timeout it is zero. Reset must initialize
the potential from the new scene, never reuse the preceding episode's value.
``joint_speed`` is a normalized squared-speed statistic supplied by the caller.
"""
import math

def pickup_reward(cfg, distances, height, impulse, table_impulse, linear_speed, angular_speed,
                  previous_hold, action_rate, joint_speed, drop, previous_potential=0., timeout=False):
    s=cfg['success'];both=all(x>s['contact_impulse_min'] for x in impulse)
    stable=(not drop and both and height>=s['lift_m'] and linear_speed<s['linear_speed_max']
            and angular_speed<s['angular_speed_max'] and table_impulse<1e-5)
    hold=previous_hold+1 if stable else 0
    required_hold=max(1,round(s['hold_seconds']*cfg['control_hz']))
    success=hold>=required_hold
    reach=sum(math.exp(-d/.15) for d in distances)/2
    lift=float(both)*max(0.,min(1.,height/s['lift_m']))
    hold_progress=max(0.,min(1.,hold/required_hold))
    normalized_speed=max(0.,min(1.,float(joint_speed)))
    raw_component={'reach':.2*reach,'bilateral_contact':.1*float(both),
                   'lift':.5*lift,'hold_progress':.2*hold_progress}
    raw_potential=sum(raw_component.values())
    terminal=bool(success or drop or timeout)
    potential=0. if terminal else raw_potential
    shaping=float(cfg.get('discount',.998))*potential-float(previous_potential)
    reward=shaping-.002+5.*float(success)-2.*float(drop)-.005*action_rate-.001*normalized_speed
    terms={'reach':reach,'bilateral_contact':float(both),'lift':lift,'hold':float(stable),
           'hold_progress':hold_progress,'success_once':float(success),'action_rate':action_rate,
           'joint_speed':normalized_speed,'drop':float(drop),'timeout':float(timeout),
           'terminal':float(terminal),'potential':potential,'raw_potential':raw_potential,
           'raw_component':raw_component,'potential_difference':shaping,
           'step_penalty':-.002,'success_bonus':5.*float(success),'drop_penalty':-2.*float(drop),
           'action_rate_penalty':-.005*action_rate,'joint_speed_penalty':-.001*normalized_speed}
    return reward,terms,hold,success,both
