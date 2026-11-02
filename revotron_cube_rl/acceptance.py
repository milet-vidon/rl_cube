"""Task evidence, separate from reward and learning diagnostics."""
ROBOT_TABLE_MAX_NS = 1e-6

def assess_pickup(rows, cfg):
    reasons=[]
    if not rows or not rows[-1]['success']: reasons.append('pickup_gate_failed')
    if sum(r['robot_table_impulse'] for r in rows)>ROBOT_TABLE_MAX_NS: reasons.append('robot_table_contact')
    if any(r['arm_soft_limit_hit'] or r['actual_arm_soft_limit_violation'] for r in rows): reasons.append('arm_soft_limit')
    if any(any(r['hard_limit_hit']) for r in rows): reasons.append('hard_limit')
    if any(r['drop'] for r in rows): reasons.append('drop')
    required=round(cfg['success']['hold_seconds']*cfg['control_hz'])
    held=rows[-required:]
    if len(held)<required or any(not r['bilateral_contact'] or r['cube_table_contact'] or r['height_m']<cfg['success']['lift_m'] for r in held):
        reasons.append('independent_contact_hold_gate')
    return {'accepted':not reasons,'failure_reasons':reasons,'robot_table_threshold_Ns':ROBOT_TABLE_MAX_NS,
        'soft_limit_hit_steps':sum(r['arm_soft_limit_hit'] or r['actual_arm_soft_limit_violation'] for r in rows),
        'hard_limit_hit_steps':sum(any(r['hard_limit_hit']) for r in rows),
        'cube_table_contact_steps':sum(r['cube_table_contact'] for r in rows),
        'max_joint_velocity_rad_s':max((max(map(abs,r['joint_velocity'])) for r in rows),default=0.),
        'max_position_error_rad':max((max(map(abs,r['joint_position_error'])) for r in rows),default=0.),
        'dropped':any(r['drop'] for r in rows),
        'continuous_hold_seconds':max((r['hold_steps'] for r in rows),default=0)/cfg['control_hz']}
