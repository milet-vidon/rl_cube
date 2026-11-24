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
from turn_pacer import TurnPacer
from left_ring_support import RingSupport,ring_impulse,parameters as ring_support_parameters,JOINT as RING_JOINT

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
 task=getattr(env,'full_task_episode',None)
 if task and task.status()['done']:return {'executed':False,'qualified':False,'failure':task.status()['reason']}
 if task and stopped():
  task.abort('execution_interrupted',truncated=True);return {'executed':False,'qualified':False,'failure':'execution_interrupted'}
 result=_execute(env,teacher,fingers,audit,rows,frames,run,args,center,stopped)
 if task and not result.get('qualified',False) and not task.status()['done']:
  task.abort(result.get('failure')or 'original_stage_gate_not_met',detail=result)
 return result

def _execute(env,teacher,fingers,audit,rows,frames,run,args,center,stopped=lambda:False):
 task=getattr(env,'full_task_episode',None)
 if getattr(args,'turn_feedback_planning_frame',False) and not getattr(args,'turn_feedback_pacing',False):raise ValueError('Planning-frame feedback requires turn pacing')
 names=env.cfg['joint_names'];initial=rows[-1];request={'initial':dict(zip(names,initial['q'])),'joint_names':names,'cube_position':initial['cube_position'],'cube_quaternion':initial['cube_quaternion_wxyz'],'left_contacts':initial['finger_contact_points'],'target_deg':args.push_degrees}
 request['refinement_options']={'ring_inset_m':getattr(args,'ring_inset',0.),'cube_clearance':getattr(args,'cube_clearance_refinement',False),'preflight_workers':getattr(args,'preflight_workers',1)}
 from torso_collision import export_fixed_colliders
 request['torso_colliders']=export_fixed_colliders(env.stage,'/World/Robot')
 print('OPPOSITION_PREFLIGHT_BEGIN',flush=True);planner='opposed_joint_plan.py' if getattr(args,'opposition_planner','baseline')=='joint' else 'opposed_grasp_plan.py';plan=host_plan(planner,request,run,'opposition',420)
 if not plan['passed']:return {'executed':True,'qualified':False,'center':center,'opposition_grasp_passed':False,'failure':'opposition_preflight_rejected','detail':plan}
 pacer=None;feedback_adapter=None
 if getattr(args,'turn_feedback_planning_frame',False):
  from turn_pacer_frame import PlanningFrameFeedback
  feedback_adapter=PlanningFrameFeedback(request)
 if getattr(args,'turn_feedback_pacing',False):
  angles=plan.get('turn_path_angles_deg',[]);turn_path=next(path for phase,path in plan['phases'] if phase=='opposed_face_turn')
  if len(angles)!=len(turn_path):return {'executed':True,'qualified':False,'center':center,'opposition_grasp_passed':False,'failure':'turn_pacing_plan_incomplete'}
  try:pacer=TurnPacer(angles)
  except ValueError as error:return {'executed':True,'qualified':False,'center':center,'opposition_grasp_passed':False,'failure':'turn_pacing_plan_invalid','detail':str(error)}
 support=None;support_check=None
 if getattr(args,'left_ring_support',False):
  support_check=host_plan('left_ring_support_preflight.py',{'request':request,'plan':plan,'baseline_target':env.targets[0].tolist()},run,'left_ring_support',1200)
  if not support_check['passed']:return {'executed':True,'qualified':False,'center':center,'opposition_grasp_passed':False,'failure':'left_ring_support_preflight_rejected','detail':support_check}
  support=RingSupport();print('LEFT_RING_SUPPORT_PREFLIGHT_PASSED',support_check['checks'],flush=True)
 print('OPPOSITION_PREFLIGHT_PASSED',plan['checks'],flush=True)
 actuation=configure_right_fingers(env,getattr(args,'right_finger_stiffness',None))
 (run/'opposition_actuation.json').write_text(json.dumps(actuation,indent=2))
 print('RIGHT_FINGER_ACTUATION',actuation['right_finger_stiffness'],flush=True)
 fixed=env.targets[0].numpy().copy();right=[i for i,n in enumerate(names)if n.endswith('_R')or n.startswith('right_')];contacts=OppositionContacts();hold=0;longest=0;turn_hold=0;turn_longest=0;grasp_passed=False;turn_started=False;failure=None;lost=0;reset=env.cube.pose_reset_count;start=time.monotonic();core0=np.array(initial['cube_position'])
 for phase,path in plan['phases']:
  if phase=='opposed_face_turn':
   if hold<90:failure='right_five_finger_hold_gate_not_met';break
   grasp_passed=True;turn_started=True
  paced=phase=='opposed_face_turn'and pacer is not None
  for tick in range(pacer.budget+1 if paced else len(path)):
   if task and stopped():failure='execution_interrupted';task.abort(failure,truncated=True);break
   if paced:
    previous=rows[-1]
    if feedback_adapter is None:index,failure=pacer.next_index(previous['layer_measurement']['face_angle_deg'],previous['opposition_audit']['right_contacts']['all_five'])
    else:index,failure,feedback=feedback_adapter.next_index(pacer,previous)
    if failure or index is None:break
   else:index=tick
   point=path[index]
   fingers.reset_step();audit.reset_step();contacts.points.clear();target=fixed.copy();target[right]=np.array(point)[right]
   if support is not None:target[names.index(RING_JOINT)]+=support.offset
   action=(torch.from_numpy(target).reshape(1,56)-env.targets)/env.scales
   if task:task.step(action,phase=phase,step_id=len(rows))
   else:env.step(action)
   row=append_row(env,teacher,fingers,audit,rows,run,phase,target);row['right_cubelet_contacts']=list(contacts.points);row['layer_measurement']=measure_layers(row['cubelets'],target_deg=args.push_degrees)
   if paced:row['turn_pacing']={'path_index':index,'planned_palm_angle_deg':pacer.angles[index],'held_ticks':pacer.held_ticks}
   if paced and feedback_adapter is not None:row['turn_pacing']['feedback']=feedback
   if support is not None:
    applied_offset=support.offset;impulse=ring_impulse(row);support.update(impulse)
    row['left_ring_support']={'applied_offset_rad':applied_offset,'measured_impulse_Ns':impulse,'next_offset_rad':support.offset}
   # append_row journal is generic; preserve complete contact detail in a second durable stream.
   with (run/'opposition_contacts.jsonl').open('a')as log:log.write(json.dumps({'step':row['step'],'right_cubelet_contacts':row['right_cubelet_contacts'],'layer_measurement':row['layer_measurement'],'turn_pacing':row.get('turn_pacing'),'left_ring_support':row.get('left_ring_support')},allow_nan=False)+'\n')
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
   if task:
    status=task.finish_step(row,failure=failure,audit={'original_audits_completed':True,'stage':'opposed_turn','original_report':report})
    row['task_episode']=status
    if status['done']:failure=status['reason']
   if tick%90==0 or failure:print('OPPOSED',row['step'],phase,'contacts',sum(topology['fingers'].values()),'hold',hold,'angle',round(row['layer_measurement']['face_angle_deg'],2),'failure',failure,flush=True)
   frame(env,frames,args,row,f'RIGHT FINGERS {sum(topology["fingers"].values())}/5 | GRIP HOLD {hold/30:.1f}s | FACE {row["layer_measurement"]["face_angle_deg"]:.1f}/{args.push_degrees:.0f}')
   if failure:break
  if failure:break
 passed=bool(not failure and grasp_passed and turn_longest>=90)
 if failure is None and not passed:failure='opposed_face_turn_hold_gate_not_met'
 return {'executed':True,'qualified':passed,'center':center,'opposition_grasp_passed':grasp_passed,'opposition_hold_s':longest/30,'turn_started':turn_started,'turn_hold_s':turn_longest/30,'target_face':'nZ','target_degrees':args.push_degrees,'right_finger_actuation':actuation,'turn_pacing':(pacer.report()|({'feedback_reference':feedback_adapter.report()}if feedback_adapter is not None else {}))if pacer else None,'left_ring_support':{'parameters':ring_support_parameters(),'preflight':support_check,'final_offset_rad':support.offset}if support else None,'failure':failure,'elapsed_s':time.monotonic()-start,'final_layers':rows[-1]['layer_measurement'],'cube_pose_writes_during_push':env.cube.pose_reset_count-reset}

