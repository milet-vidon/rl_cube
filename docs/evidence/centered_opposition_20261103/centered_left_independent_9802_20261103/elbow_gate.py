"""Freeze the combined outward-elbow, five-pad, palm-facing physical gate."""
import argparse,json
from pathlib import Path
from elbow_audit import elbow_metrics,MIN_OUTWARD_M,MIN_OUTWARD_DEG
from palm_handover_audit import hand_geometry
from tilt_audit import tilt_error,contact_topology
from kinematics import RobotKinematics
from evidence_paths import local_path,raw_sha
from palm_direction_gate import build_certificate as build_left_certificate

ROOT=Path(__file__).resolve().parent


def combined_hold(rows,names,regrasp=False):
    k=RobotKinematics();streak=longest=0;final=None
    for row in rows:
        m=row['measured_elbow']
        final=elbow_metrics(m['shoulder_root_m'],m['elbow_root_m'],m['wrist_root_m'])
        g=hand_geometry(row,names,k)
        posture=(final['outward_from_shoulder_m']>=MIN_OUTWARD_M
                 and final['upper_arm_outward_deg']>=MIN_OUTWARD_DEG
                 and g['left_palm_normal_world'][1]<=-.6
                 and g['wrist_y_separation_m']>=.03 and g['elbow_y_separation_m']>=.04
                 and g['left_wrist_from_cube_y_m']>=.02 and g['right_wrist_from_cube_y_m']>=.02)
        if regrasp:
            posture &= g['left_facing_right_cos']>=.65 and g['right_facing_left_cos']>=.65 and g['palm_normal_dot']<=-.3
        contact=all(all(contact_topology(row,s).values()) for s in (('left','right') if regrasp else ('left',)))
        good=(row['phase']==('bilateral_hold' if regrasp else 'tilt_hold') and posture and contact
              and row['height_m']>=.08 and not row['cube_table_contact'] and not row['drop']
              and row['linear_speed']<.03 and row['angular_speed']<.20 and tilt_error(row)<=5.)
        streak=streak+1 if good else 0;longest=max(longest,streak)
    return {'accepted':longest>=90,'longest_jointly_qualified_hold_s':longest/30,'final_elbow':final}


def build(protocol,runs):
    protocol=local_path(protocol,ROOT)
    spec=json.loads(protocol.read_text())
    if spec['elbow_min_outward_m']!=MIN_OUTWARD_M or spec['elbow_min_outward_degrees']!=MIN_OUTWARD_DEG:
        raise ValueError('Elbow threshold contract changed')
    result=build_left_certificate(protocol,runs)
    for trial in result['trials']:
        run=local_path(trial['run'],ROOT)
        rows=json.loads((run/f"trajectory_{trial['seed']}.json").read_text())
        cfg=json.loads((run/'config.json').read_text())
        audit=combined_hold(rows,cfg['joint_names'])
        safe=all(sum(r[key] for r in rows)<=1e-6 for key in ['robot_table_impulse','self_contact_impulse'])
        if not audit['accepted'] or not safe:
            trial['accepted']=False;result['failures'].append('combined_elbow_hold_failed:'+str(trial['seed']))
        trial['elbow_combined_audit']=audit
        trial['run']=str(run.relative_to(ROOT))
    if result['independent_total']<5:result['failures'].append('fewer_than_five_independent_seeds')
    result['passed']=not result['failures']
    result['accepted']=sum(t['accepted'] for t in result['trials'])
    result['independent_accepted']=sum(t['accepted'] for t in result['trials'] if t['split']=='independent')
    result.update(left_elbow_outward_validated=result['passed'],right_regrasp_validated=False,
                  gate_source_sha256=raw_sha(__file__),path_format='project-relative runs; workspace-relative assets')
    result['assets_sha256']={str(local_path(p,ROOT).relative_to(ROOT.parent)):d for p,d in result['assets_sha256'].items()}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=build(a.protocol,a.runs);Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['passed','accepted','total','independent_accepted','independent_total','failures']}))
    raise SystemExit(0 if result['passed'] else 1)
