"""Right thumb/four-finger grasp, gated hold, then contact-only face rotation."""
import json,time
import numpy as np,torch
from face_push_runtime import LayerContacts
from pxr import PhysicsSchemaTools
from centered_opposition_runtime import host_plan,append_row,safety_failure,frame
from layer_turn_metrics import measure_layers
from opposed_grasp_audit import qualified,right_contacts
from tilt_audit import contact_topology
from opposed_actuation import configure_right_fingers

class OppositionContacts(LayerContacts):
 def callback(self,headers,data):
  for header in headers:
   paths=[str(PhysicsSchemaTools.intToSdfPath(value))for value in (header.actor0,header.actor1)]
   robot=next((p for p in paths if p.startswith('/World/Robot/right_')or(p.startswith('/World/Robot/Link')and p.endswith('_R'))),None)
   cube=next((p for p in paths if p.startswith('/World/Cube/')),None)
   if robot is None or cube is None:continue
   for point in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]:
    impulse=float(np.linalg.norm(np.asarray(point.impulse)))
    if impulse>1e-7:self.points.append({'robot_body':robot,'cubelet':cube.rsplit('/',1)[-1],'impulse_Ns':impulse,'position_world_m':list(point.position)})

def execute(env,teacher,fingers,audit,rows,frames,run,args,center,stopped=lambda:False):
 names=env.cfg['joint_names'];initial=rows[-1];request={'initial':dict(zip(names,initial['q'])),'joint_names':names,'cube_position':initial['cube_position'],'cube_quaternion':initial['cube_quaternion_wxyz'],'left_contacts':initial['finger_contact_points'],'target_deg':args.push_degrees}
 request['refinement_options']={'ring_inset_m':getattr(args,'ring_inset',0.),'cube_clearance':getattr(args,'cube_clearance_refinement',False)}
 from torso_collision import export_fixed_colliders
 request['torso_colliders']=export_fixed_colliders(env.stage,'/World/Robot')
 print('OPPOSITION_PREFLIGHT_BEGIN',flush=True);planner='opposed_joint_plan.py' if getattr(args,'opposition_planner','baseline')=='joint' else 'opposed_grasp_plan.py';plan=host_plan(planner,request,run,'opposition',420)
 if not plan['passed']:return {'executed':True,'qualified':False,'center':center,'opposition_grasp_passed':False,'failure':'opposition_preflight_rejected','detail':plan}
 print('OPPOSITION_PREFLIGHT_PASSED',plan['checks'],flush=True)
 actuation=configure_right_fingers(env,getattr(args,'right_finger_stiffness',None))
 (run/'opposition_actuation.json').write_text(json.dumps(actuation,indent=2))
 print('RIGHT_FINGER_ACTUATION',actuation['right_finger_stiffness'],flush=True)
 fixed=env.targets[0].numpy().copy();right=[i for i,n in enumerate(names)if n.endswith('_R')or n.startswith('right_')];contacts=OppositionContacts();hold=0;longest=0;turn_hold=0;turn_longest=0;grasp_passed=False;turn_started=False;failure=None;lost=0;reset=env.cube.pose_reset_count;start=time.monotonic();core0=np.array(initial['cube_position'])
 for phase,path in plan['phases']:
  if phase=='opposed_face_turn':
   if hold<90:failure='right_five_finger_hold_gate_not_met';break
   grasp_passed=True;turn_started=True
  for tick,point in enumerate(path):
   fingers.reset_step();audit.reset_step();contacts.points.clear();target=fixed.copy();target[right]=np.array(point)[right];env.step((torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales)
   row=append_row(env,teacher,fingers,audit,rows,run,phase,target);row['right_cubelet_contacts']=list(contacts.points);row['layer_measurement']=measure_layers(row['cubelets'],target_deg=args.push_degrees)
   # append_row journal is generic; preserve complete contact detail in a second durable stream.
   with (run/'opposition_contacts.jsonl').open('a')as log:log.write(json.dumps({'step':row['step'],'right_cubelet_contacts':row['right_cubelet_contacts'],'layer_measurement':row['layer_measurement']},allow_nan=False)+'\n')
   failure=safety_failure(row)
   if stopped():failure='execution_interrupted'
   primary=contact_topology(row,'left');retained=primary['thumb']and sum(primary[f]for f in ('index','middle','ring','little'))>=3;lost=0 if retained else lost+1
   if lost>=10:failure='left_grasp_lost_during_opposition'
   if np.linalg.norm(np.array(row['cube_position'])-core0)>.025:failure='held_cube_position_envelope'
   topology=right_contacts(contacts.points,row['cube_position'],row['cube_quaternion_wxyz'],row['layer_measurement']['face_angle_deg']if turn_started else 0.)
   if topology['unexpected_contacts']:failure='wrong_finger_body_or_layer_contact'
   if contacts.points and phase in ('right_preshape','right_route','right_approach'):failure='premature_cube_contact'
   if row['layer_measurement']['maximum_other_cubelet_rotation_deg']>15:failure='non_target_layers_rotating'
   if env.cube.pose_reset_count!=reset:raise RuntimeError('Unexpected cube pose write during opposed grasp')
   report=qualified(row,names,turned=turn_started);row['opposition_audit']=report
   if phase=='opposition_hold':hold=hold+1 if report['qualified']and not failure else 0;longest=max(longest,hold)
   if phase=='opposed_face_hold':turn_hold=turn_hold+1 if report['qualified']and not failure else 0;turn_longest=max(turn_longest,turn_hold)
   if tick%90==0 or failure:print('OPPOSED',row['step'],phase,'contacts',sum(topology['fingers'].values()),'hold',hold,'angle',round(row['layer_measurement']['face_angle_deg'],2),'failure',failure,flush=True)
   frame(env,frames,args,row,f'RIGHT FINGERS {sum(topology["fingers"].values())}/5 | GRIP HOLD {hold/30:.1f}s | FACE {row["layer_measurement"]["face_angle_deg"]:.1f}/{args.push_degrees:.0f}')
   if failure:break
  if failure:break
 passed=bool(not failure and grasp_passed and turn_longest>=90)
 if failure is None and not passed:failure='opposed_face_turn_hold_gate_not_met'
 return {'executed':True,'qualified':passed,'center':center,'opposition_grasp_passed':grasp_passed,'opposition_hold_s':longest/30,'turn_started':turn_started,'turn_hold_s':turn_longest/30,'target_face':'nZ','target_degrees':args.push_degrees,'right_finger_actuation':actuation,'failure':failure,'elapsed_s':time.monotonic()-start,'final_layers':rows[-1]['layer_measurement'],'cube_pose_writes_during_push':env.cube.pose_reset_count-reset}
