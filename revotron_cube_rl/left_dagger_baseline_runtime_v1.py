"""Baseline-only live DAgger runtime, initially restricted to preserve diagnostics.

Imports Isaac only inside construction. Ordinary 27-body cube, original teacher,
original action/gain/safety path. This module never starts a second simulator on
its own; the root-run entry point constructs one after external queue admission.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = 'center_j56_dev_v3_20261103'
RUNTIME_FILES = ('left_dagger_baseline_runtime_v1.py', 'collect_left_dagger_baseline_v1.py',
                 'dagger_collection_core_v1.py', 'collect_left_dagger_draft_v1.py',
                 'left_contact_dagger_schema_v1.json', 'phase_basis_policy_v4.py',
                 'phase_contact_policy.py', 'runtime_platform.py')
ORIGINAL_FILES = ('environment.py', 'articulated_cube_env.py', 'dynamic_cube.py', 'config.json',
                  'isaac_compat.py', 'joint_safety.py', 'finger_contact_audit.py',
                  'palm_contact_audit.py', 'self_contact_audit.py', 'layer_turn_metrics.py',
                  'wrist_direction.py', 'wrist_audit.py', 'elbow_handover.py', 'elbow_audit.py',
                  'elbow_gate.py', 'palm_handover.py', 'palm_handover_audit.py',
                  'tilt_regrasp.py', 'tilt_audit.py', 'tilt_collision_plan.py',
                  'opposition_teacher.py', 'opposition_feedback.py', 'opposition_grasp_v20.json',
                  'proximal_planner.py', 'kinematics.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint_metadata(metadata,preflight,current_hashes):
    if metadata.get('joint_names')!=preflight['names'] or metadata.get('output_indices')!=preflight['left']:
        raise ValueError('Checkpoint joint names/order differ from selected baseline')
    sources=metadata.get('sources',[])
    if len(sources)!=1 or sources[0].get('run')!=BASELINE or sources[0].get('manifest_sha256')!=preflight['metadata']['manifest_sha256']:
        raise ValueError('Checkpoint source manifest differs from selected baseline')
    for name in ('phase_contact_policy.py','phase_basis_policy_v4.py'):
        if metadata.get('source_sha256',{}).get(name)!=current_hashes[name]:
            raise ValueError('Checkpoint inference dependency changed: '+name)
    return True


def checkpoint_preflight(checkpoint,preflight):
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    validate_checkpoint_metadata(saved['metadata'],preflight,{n:sha(ROOT/n) for n in ('phase_contact_policy.py','phase_basis_policy_v4.py')})
    return {'checkpoint_sha256':sha(checkpoint),'schema':saved['metadata']['schema'],
            'source_manifest_sha256':preflight['metadata']['manifest_sha256'],'joint_schema_verified':True,
            'inference_dependencies_verified':['phase_contact_policy.py','phase_basis_policy_v4.py']}


def serializable(value):
    if value is None or isinstance(value, (str, bool, int, float)): return value
    if isinstance(value, dict): return {str(k): serializable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(v) for v in value]
    if hasattr(value, 'detach'): return value.detach().cpu().tolist()
    if hasattr(value, 'numpy'): return value.numpy().tolist()
    if hasattr(value, 'tolist'): return value.tolist()
    if isinstance(value, type): return value.__module__+'.'+value.__qualname__
    try: return [serializable(v) for v in value]
    except TypeError:
        text=str(value)
        if ' object at 0x' in text: raise TypeError('Unstable physics metadata representation')
        return text


def physics_authorship(stage):
    """Cover all actual body/joint/material paths, not assumed scene prefixes."""
    result={}
    dynamic_names={'physics:velocity','physics:angularvelocity'}
    for prim in stage.Traverse():
        attributes={};relationships={}
        for attr in prim.GetAttributes():
            name=attr.GetName();key=name.lower()
            if not ('physics:' in key or key.startswith('physx') or key.startswith('drive:')):continue
            if key in dynamic_names or key.startswith('state:') or ':state:' in key or key.endswith(('targetposition','targetvelocity')):continue
            attributes[name]=serializable(attr.Get())
        for rel in prim.GetRelationships():
            name=rel.GetName();key=name.lower()
            if 'physics' in key or key.startswith('physx') or key.startswith('material:binding'):
                relationships[name]=[str(path) for path in rel.GetTargets()]
        if attributes or relationships:result[str(prim.GetPath())]={'attributes':attributes,'relationships':relationships}
    return result


def source_preflight(source):
    source = Path(source)
    if source.name != BASELINE: raise ValueError('This runtime is restricted to the selected original baseline physics domain')
    from phase_contact_policy import accepted_prefix
    rows,names,left,metadata = accepted_prefix(source)
    experiment=json.loads((source/'experiment.json').read_text())
    for name in ORIGINAL_FILES:
        if sha(ROOT/name) != experiment['source_sha256'][name]:
            raise ValueError('Frozen baseline source changed: '+name)
    # Same certificate checks that guarded the original source experiment.
    from stage_gate import require_staged_gate
    from side_gate import require_side_gate
    from opposition_gate import require_opposition_gate
    from evidence_paths import local_path
    for gate in (require_staged_gate,require_side_gate,require_opposition_gate): gate()
    cert=json.loads((ROOT/'wrist_direction_gate_certificate.json').read_text())
    if not cert['passed'] or cert['independent_accepted']<5: raise ValueError('Missing accepted wrist baseline')
    for name,digest in cert['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Wrist certificate source changed: '+name)
    if sha(ROOT/'wrist_direction_protocol_v1.json')!=cert['protocol_sha256']: raise ValueError('Wrist protocol changed')
    for name,digest in cert['assets_sha256'].items():
        if sha(local_path(name,ROOT))!=digest: raise ValueError('Wrist certificate asset changed: '+name)
    for trial in cert['trials']:
        run=local_path(trial['run'],ROOT)
        for name,key in (('config.json','config_sha256'),('runtime_config.json','runtime_config_sha256'),
                         ('asset_contract.json','asset_contract_sha256'),(f"trajectory_{trial['seed']}.json",'trajectory_sha256')):
            if sha(run/name)!=trial[key]: raise ValueError('Wrist certificate evidence changed: '+str(run/name))
    if experiment['friction']!='.9,.8' or experiment['hand_friction'] is not None: raise ValueError('Unexpected source material')
    return {'rows':rows,'names':names,'left':left,'metadata':metadata,'experiment':experiment,
            'phase_frame_counts':dict(Counter(r['phase'] for r in rows)),
            'source_files':{n:sha(ROOT/n) for n in set(ORIGINAL_FILES+RUNTIME_FILES)},
            'original_wrist_certificate_sha256':sha(ROOT/'wrist_direction_gate_certificate.json')}


def termination_status(done, metrics, episode_steps, maximum_steps):
    """Explicit conservative v1 contract: no sampling after any original done.

    The original four-return API supplies one done and no truncated flag. We
    record timeout-derived truncation without pretending it was returned.
    """
    success=bool(metrics.get('success',False));drop=bool(metrics.get('drop',False));timeout=episode_steps>=maximum_steps
    causes=[name for name,value in (('generic_pickup_success',success),('drop',drop),('timeout',timeout)) if value]
    return {'raw_done':bool(done),'raw_truncated':None,'api_return_arity':4,
            'derived_timeout_truncated':timeout,'derived_drop_terminated':drop,
            'generic_pickup_success':success,'done_causes':causes,
            'unknown_done_cause':bool(done and not causes),
            'done_matches_original_causes':bool(done)==bool(success or drop or timeout),
            'sampling_terminal':bool(done or success or drop or timeout),
            'contract':'preserve_diagnostic_v1_stops_on_every_original_done_not_a_training_episode'}


class BaselineRuntime:
    decoder_id='original_environment_targets_plus_scales_times_native_action_float32_v1'

    def __init__(self,args,preflight):
        self.args=args;self.source=Path(args.source);self.output=Path(args.output);self.preflight=preflight
        self.app=self.env=None;self.rows=[];self.last_row=None;self.failure=None;self.gate_report=None
        self.last_termination=None;self.phase_seen=Counter();self.current_schedule={'phase':'ready','phase_progress':0.}
        self.started=time.monotonic();self.closed=False;self.plan_hashes=set();self.source_sha256=preflight['source_files']
        self.joint_names=preflight['names'];self.left_indices=preflight['left'];self.source_rows=preflight['rows']
        self.maximum_reference_target_error=0.;self.maximum_reference_q_error=0.;self.maximum_reference_cube_error=0.
        self.journal=(self.output/'trajectory.jsonl').open('w')
        self.plan_dir=self.output/'plans';self.plan_dir.mkdir()
        snapshot=self.output/'source_snapshot';snapshot.mkdir()
        for name in sorted(self.source_sha256): shutil.copy2(ROOT/name,snapshot/name)
        self.write('runtime_protocol.json',{'scope':'privileged_state_preserve_teacher_diagnostic_pickup_to_wrist_only',
            'source_manifest':preflight['metadata'],'source_sha256':self.source_sha256,'seed':args.seed,
            'checkpoint_validation':preflight['checkpoint_validation'],
            'eligible_training_episode':False,'teacher_probability':args.teacher_probability,
            'physics_domain':'original_baseline_27_ordinary_rigid_bodies_no_damping_or_articulation_wrappers',
            'public_phase_progress':'elapsed actual actions in current teacher phase / baseline phase duration; clamped at one during additional waiting',
            'phase_counts':preflight['phase_frame_counts'],'termination':'stop on every env done, drop, timeout, original teacher failure, or final original grasp gate',
            'gate_threshold_changes':False,'center_oracle_implemented':False})
        try:
            self._initialize()
        except BaseException as exc:
            import traceback
            self.write('error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            self.close();raise

    def write(self,name,value):
        (self.output/name).write_text(json.dumps(serializable(value),indent=2,allow_nan=False))

    def _initialize(self):
        import torch
        from isaaclab.app import AppLauncher
        from runtime_platform import platform_kit_args
        self.app=AppLauncher(headless=True,enable_cameras=False,kit_args=platform_kit_args('')).app
        from articulated_cube_env import DynamicPickupEnv
        from wrist_direction import WristDirectionTeacher
        from finger_contact_audit import FingerContactAudit
        from palm_contact_audit import DetailedSelfContactAudit
        from isaac_compat import tensor,quaternion_from_native
        self.tensor=tensor;self.quaternion_from_native=quaternion_from_native
        self.env=DynamicPickupEnv(self.output,friction=(.9,.8),hand_friction=None,seed=self.args.seed,
            vision=False,preview=False,auto_reset=False,arm_damping=8.)
        env=self.env;env.max_episode_length=int(self.preflight['experiment']['max_steps']);env.cfg['success']['hold_seconds']=3.
        env.rng=np.random.default_rng(self.args.seed);env.reset();env.metrics={};env.pair_impulses={}
        if env.cfg != json.loads((self.source/'config.json').read_text()): raise ValueError('Baseline runtime config differs from source')
        if env.contract != json.loads((self.source/'asset_contract.json').read_text()): raise ValueError('Baseline runtime asset contract differs from source')
        if env.cfg['joint_names']!=self.joint_names or len(env.cube.objects)!=27: raise ValueError('Wrong baseline joint/cube representation')
        self.write('config.json',env.cfg);self.write('asset_contract.json',env.contract)
        self.teacher=WristDirectionTeacher(env,ROOT/'opposition_grasp_v20.json')
        self.fingers=FingerContactAudit();self.audit=DetailedSelfContactAudit()
        self.initial_reset_count=env.cube.pose_reset_count
        self.contact_impulses={'left':{f:0. for f in ('thumb','index','middle','ring','little')},
                               'right':{f:0. for f in ('thumb','index','middle','ring','little')}}
        self.contact_points=[]
        self.backend_names=list(env.robot.root_view.shared_metatype.dof_names)
        if len(set(self.backend_names))!=len(self.backend_names): raise ValueError('Ambiguous backend DOF names')
        self.backend_indices=[self.backend_names.index(n) for n in self.joint_names]
        self.initial_gains=self.control_settings()
        self.physics_identity=self._physics_identity()
        self.write('physics_identity.json',self.physics_identity)
        self.write('initial_state.json',self.capture())
        self.archive_changed_plans({})

    def _backend(self,getter):
        value=getattr(self.env.robot.root_view,getter)()
        value=np.asarray(serializable(value))
        if value.ndim!=2 or value.shape[0]!=1: raise ValueError('Unexpected backend gain dimensions')
        return value[0,self.backend_indices].tolist()

    def control_settings(self):
        return {'joint_stiffness':self._backend('get_dof_stiffnesses'),
                'joint_damping':self._backend('get_dof_dampings'),
                'joint_armature':self._backend('get_dof_armatures')}

    def _physics_identity(self):
        from pxr import Usd
        from isaac_compat import runtime_versions
        env=self.env;robot_path=Path(env.robot.cfg.spawn.usd_path).resolve();stage=Usd.Stage.Open(str(robot_path))
        assets={str(Path(layer.realPath).resolve()):sha(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        assets[str(env.dynamic_source.resolve())]=sha(env.dynamic_source)
        usd_settings=physics_authorship(env.stage)
        simcfg=self.env.sim.cfg
        physics_cfg=getattr(simcfg,'physics',getattr(simcfg,'physx',None))
        if physics_cfg is None: raise ValueError('Missing actual simulation solver configuration')
        solver=serializable(physics_cfg.to_dict())
        return {'environment_class':'articulated_cube_env.DynamicPickupEnv',
                'config_sha256':sha(self.output/'config.json'),'assets_sha256':assets,
                'source_sha256':{n:self.source_sha256[n] for n in ORIGINAL_FILES},
                'cube_constraint_topology':{'representation':'27 ordinary rigid object handles with original source joints',
                    'd6_damping_augmentation':False,'native_articulation_facade':False,'model':env.cube.model},
                'damping':{'actual_robot':self.initial_gains['joint_damping'],'cube':'original source; no added damping'},
                'armature':{'actual_robot':self.initial_gains['joint_armature'],'cube':'original source'},
                'friction':{'cube_material':[.9,.8],'hand_override':None,'table_material':[.8,.7]},
                'solver_options':{'physics_cfg':solver,'device':env.device,'physics_dt':env.dt,'control_dt':env.control_dt,'num_envs':env.num_envs,'runtime_versions':runtime_versions()},
                'initial_joint_stiffness':self.initial_gains['joint_stiffness'],
                'authored_physics_attributes_and_relationships':usd_settings,
                'physics_authorship_scope':'all stage paths including Core/Joints/Material/Table and material bindings; excludes runtime velocity/state/target values'}

    def capture(self):
        env=self.env;t=self.tensor;root=t(env.cube.data.root_state_w)[0]
        return {'q':t(env.robot.data.joint_pos)[0,env.ids].tolist(),
                'joint_velocity':t(env.robot.data.joint_vel)[0,env.ids].tolist(),
                'applied_target':env.targets[0].tolist(),'cube_position':root[:3].tolist(),
                'cube_quaternion_wxyz':self.quaternion_from_native(root[3:7]).tolist(),
                'cube_linear_velocity':root[7:10].tolist(),'cube_angular_velocity':root[10:13].tolist(),
                'finger_contact_impulses':deepcopy(self.contact_impulses),'finger_contact_points':deepcopy(self.contact_points),
                **self.current_schedule,'previous_executed_source':'reset' if not self.rows else 'teacher',
                'oracle_env_inputs':{'metrics':deepcopy(env.metrics),'contact_impulse':env.contact_impulse.tolist(),
                                    'episode_length':int(env.episode_length_buf[0]),'max_episode_length':env.max_episode_length,
                                    'pose_reset_count':env.cube.pose_reset_count},
                'control_settings':self.control_settings()}

    def decode_target(self,native_action):
        # Identical operation order/dtype to frozen environment.py:195.
        return (self.env.targets+self.env.scales*native_action.cpu())[0].tolist()

    def schedule_resolver(self,observation,before,after):
        phase=after['name']
        duration=self.preflight['phase_frame_counts'].get(phase)
        if duration is None or phase in ('left_center_transport','center_hold'): raise ValueError('Unsupported active phase')
        step=self.phase_seen[phase];self.current_schedule={'phase':phase,'phase_progress':min(1.,step/max(1,duration-1))}
        return dict(self.current_schedule)

    def validate_control_change(self,before,after,teacher_before,teacher_after):
        if teacher_before.get('hold_gains_set',False) or not teacher_after.get('hold_gains_set',False): return False
        if teacher_after.get('name')!='tilt_hold': return False
        if before['joint_damping']!=after['joint_damping'] or before['joint_armature']!=after['joint_armature']:return False
        expected=list(before['joint_stiffness'])
        for i,name in enumerate(self.joint_names):
            if name.startswith('left_'):expected[i]=20.
        return after['joint_stiffness']==expected

    def _gate(self,row=None):
        row=row or {};safe=(row.get('self_contact_impulse',0.)<=1e-6 and row.get('robot_table_impulse',0.)<=1e-6)
        limits=not(row.get('arm_soft_limit_hit',False) or row.get('actual_arm_soft_limit_violation',False) or any(row.get('hard_limit_hit',[])))
        intact=row.get('layer_measurement',{}).get('maximum_relative_rotation_deg',0.)<=30.
        return {'robot_safety':safe,'joint_limits':limits,'cube_integrity':intact,
                'no_pose_write':self.env.cube.pose_reset_count==self.initial_reset_count,
                'stage_gate_qualified':False,'original_audit_source_sha256':{n:self.source_sha256[n] for n in
                    ('environment.py','joint_safety.py','wrist_audit.py','elbow_gate.py','palm_handover_audit.py','tilt_audit.py','layer_turn_metrics.py')}}

    def pre_gates(self,observation):
        if self.failure or (self.last_termination and self.last_termination['sampling_terminal']):raise RuntimeError('Runtime already terminal')
        self.fingers.reset_step();self.audit.reset_step()
        # Original teacher owns its measured phase transitions; this adapter
        # does not force a transition by a reference frame deadline.
        return self._gate(self.last_row),False

    def execute_packet(self,packet):
        import torch
        env=self.env;action=packet['native_teacher_normalized_action'].detach().clone()
        if action.shape!=(1,56) or action.dtype!=torch.float32:raise ValueError('Expected original native float32 teacher action')
        if packet['executed_source']!='teacher':raise ValueError('Live learner sampling remains locked pending termination-contract review')
        returned=env.step(action)
        if len(returned)!=4:raise ValueError('Original baseline step return arity changed')
        done=bool(returned[2].item())
        self.last_termination=termination_status(done,env.metrics,int(env.episode_length_buf[0]),env.max_episode_length)
        self._record_row(action,packet)

    def _record_row(self,action,packet):
        from palm_handover_audit import measured_hand_frames,tilt_error
        from elbow_audit import measured_elbow
        from wrist_audit import wrist_geometry
        from layer_turn_metrics import measure_layers
        env=self.env;t=self.tensor;step=len(self.rows)
        row=dict(step=step,phase=self.teacher.name,**env.metrics);row.update(self.fingers.metrics());row.update(self.audit.metrics())
        row.update(q=t(env.robot.data.joint_pos)[0,env.ids].tolist(),
                   cube_quaternion_wxyz=self.quaternion_from_native(t(env.cube.data.root_state_w)[0,3:7]).tolist(),
                   tilt_degrees=-90,right_contact_axis=0,right_thumb_sign=-1,
                   measured_hand_frames=measured_hand_frames(env),measured_elbow=measured_elbow(env),
                   tilt_reference_wxyz=self.teacher.reference.tolist() if self.teacher.reference is not None else None,
                   nominal_target=self.teacher.nominal_target[0].tolist())
        row['measured_wrist']=wrist_geometry(row,self.joint_names);row['tilt_error_deg']=tilt_error(row)
        row['cubelets']=env.cube.states();row['layer_measurement']=measure_layers(row['cubelets']);row['pose_reset_count']=env.cube.pose_reset_count
        row['termination']=deepcopy(self.last_termination);row['executed_source']=packet['executed_source'];row['native_action']=action[0].tolist()
        row['policy_phase_progress']=self.current_schedule['phase_progress']
        if step<len(self.source_rows) and self.args.seed==self.preflight['metadata']['seed']:
            reference=self.source_rows[step]
            target_error=float(np.max(abs(np.asarray(row['applied_target'])-reference['applied_target'])))
            q_error=float(np.max(abs(np.asarray(row['q'])-reference['q'])))
            cube_error=float(np.max(abs(np.asarray(row['cube_position'])-reference['cube_position'])))
            row['diagnostic_reference_difference']={'phase_matches':row['phase']==reference['phase'],
                'target_max_rad':target_error,'q_max_rad':q_error,'cube_position_max_m':cube_error}
            self.maximum_reference_target_error=max(self.maximum_reference_target_error,target_error)
            self.maximum_reference_q_error=max(self.maximum_reference_q_error,q_error)
            self.maximum_reference_cube_error=max(self.maximum_reference_cube_error,cube_error)
        self.rows.append(row);self.last_row=row;self.phase_seen[row['phase']]+=1
        self.contact_impulses=deepcopy(row['finger_contact_impulses']);self.contact_points=deepcopy(row['finger_contact_points'])
        self.teacher.finger_contacts=self.contact_impulses;self.teacher.contact_points=self.contact_points
        self.journal.write(json.dumps(row,allow_nan=False)+'\n');self.journal.flush()

    def post_gates(self,observation):
        from palm_handover_audit import assess_handover
        from wrist_audit import assess_wrist
        row=self.last_row;gate=self._gate(row)
        if not gate['robot_safety'] or not gate['joint_limits']:self.failure='robot_safety_gate'
        if row['drop']:self.failure='cube_drop'
        if self.teacher.failure_reason:self.failure=self.teacher.failure_reason
        if not gate['cube_integrity']:self.failure='cube_layers_deformed_before_turn'
        if not gate['no_pose_write']:self.failure='unexpected_cube_pose_write'
        if self.last_termination['sampling_terminal'] and not self.failure:
            self.failure='original_environment_done:'+','.join(self.last_termination['done_causes'] or ['unknown'])
        qualified=False
        if self.teacher.accepted:
            measured=assess_handover(self.rows,self.joint_names,regrasp=False,full_episode=True)
            wrist=assess_wrist([r if r['layer_measurement']['assembled_for_grasp'] else r|{'phase':'cube_not_assembled'} for r in self.rows],self.joint_names)
            hold=longest=0
            for r in self.rows:
                hold=hold+1 if r['phase']=='tilt_hold' and r['layer_measurement']['assembled_for_grasp'] else 0;longest=max(longest,hold)
            qualified=bool(not self.failure and measured['accepted'] and wrist['accepted'] and longest>=90)
            self.gate_report={'teacher_accepted':bool(self.teacher.accepted),'measured':measured,'wrist':wrist,'assembled_hold_s':longest/30.,'qualified':qualified}
            if not qualified and not self.failure:self.failure='dynamic_grasp_acceptance_gate_not_met'
        gate['stage_gate_qualified']=qualified
        if row['step']%120==0 or self.failure or qualified:
            print('DAGGER_BASELINE_PRESERVE',row['step'],row['phase'],'height',row['height_m'],'done',self.last_termination['raw_done'],'failure',self.failure,flush=True)
        return gate,qualified,self.failure

    def archive_changed_plans(self,snapshot):
        from dagger_collection_core_v1 import canonical
        for key,value in (('pickup',self.teacher.base.phases),('tilt',self.teacher.path)):
            if value is None:continue
            digest=snapshot.get('base',{}).get('phases_sha256') if key=='pickup' else snapshot.get('path_sha256')
            if digest in self.plan_hashes:continue
            data=canonical(value);actual_digest=hashlib.sha256(data.encode()).hexdigest()
            if digest is not None and digest!=actual_digest:raise ValueError('Plan changed between expert snapshot and archive')
            digest=actual_digest
            if digest not in self.plan_hashes:
                (self.plan_dir/(key+'_'+digest+'.json')).write_text(data);self.plan_hashes.add(digest)

    def finalize(self,collection):
        if not collection.get('original_wrist_gate_qualified') and not self.failure:
            self.failure='collection_step_budget_before_original_wrist_gate'
        result={**collection,'kind':'preserve_original_teacher_diagnostic_not_training_episode',
                'seed':self.args.seed,'eligible_training_episode':False,'failure':self.failure,
                'original_grasp_gate':self.gate_report,'last_environment_termination':self.last_termination,
                'max_reference_target_error_rad':self.maximum_reference_target_error,
                'max_reference_q_error_rad':self.maximum_reference_q_error,
                'max_reference_cube_position_error_m':self.maximum_reference_cube_error,
                'cube_pose_writes_during_control':self.env.cube.pose_reset_count-self.initial_reset_count,
                'maximum_lift_m':max((r['height_m'] for r in self.rows),default=0.),
                'elapsed_s':time.monotonic()-self.started,'runtime_bridge_validated':False,
                'center_oracle_implemented':False,'pure_student_episode_claim':False}
        self.write('result.json',result);self.write('trajectory.json',self.rows)
        self.journal.flush()
        self.write('artifact_manifest.json',{'sha256':{p.name:sha(p) for p in self.output.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}})
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        if hasattr(self,'journal'):self.journal.close()
        if self.env is not None:self.env.close()
        if self.app is not None:
            from isaac_compat import close_application
            close_application(self.app)


def create_runtime(args):
    if float(args.teacher_probability)!=1.:raise ValueError('First baseline runtime is preserve diagnostic only; live mixture requires reviewed termination contract')
    preflight=source_preflight(args.source);preflight['checkpoint_validation']=checkpoint_preflight(args.checkpoint,preflight)
    return BaselineRuntime(args,preflight)
