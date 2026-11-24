"""Isolated worker for the dashboard's explicit, GPU-idle-only physical evaluation.

Run by the local job manager, with the real training and Isaac lock FDs inherited.
The production validation sampler selects the actions, simulates the scene and
decides termination. Observers in THIS child only copy its measurements to JSONL.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import inference

SEEDS = list(range(20_000_000, 20_000_004))
LANE = 0


def _sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _write(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('x') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.rename(path)


def require_reservations():
    from contact_hold_continuous import LOCKS
    for var, name in [('CUBE_DASH_GPU_FD','contact_hold_gpu_training.lock'), ('CUBE_DASH_ISAAC_FD','isaac_gui_evaluation.lock')]:
        try:
            fd=int(os.environ[var])
            if fd < 3 or not os.path.samestat(os.fstat(fd),(LOCKS / name).stat()):
                raise ValueError(name)
            # Inherited descriptors reference the same open file description as the scheduler.
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (KeyError, OSError, ValueError) as err:
            raise RuntimeError('请通过面板在训练 GPU 空闲时启动物理评估。') from err


def make_plan(model_id):
    from contact_hold90_cuda_sample import FILES
    models = inference._model_cards()
    if model_id not in models:
        raise ValueError('未找到此已完成模型')
    card, ck = models[model_id]
    run = card['run'] if model_id != inference.ARCHIVE_MODEL else 'contact_hold90_long_v2'
    settings = inference._read_json(inference.RUNS/run/'settings.json')
    root_sha = settings.get('warm_start_sha256') if settings.get('engine')==inference.TUNABLE_ENGINE else settings.get('source_weights_sha256')
    if root_sha != inference.R15_SHA or _sha(ck) != card['sha256']:
        raise ValueError('冻结模型校验失败')
    scene = inference.RUNS/inference.ARCHIVE/'from_reset'/'seed_21611'
    grasp = inference.RUNS/inference.ARCHIVE/'calibration'/'grasp90.json'
    code={name:settings['code'][name] for name in FILES}
    if (Path(settings['scene_source']).resolve() != scene or Path(settings['grasp']).resolve()!=grasp
            or _sha(grasp)!=settings['grasp_sha256'] or _sha(scene/'dynamic_training_scene.usda')!=settings['scene_sha256']
            or any(_sha(inference.ROOT/n)!=h for n,h in code.items())):
        raise ValueError('场景、抓取标定或采样源码与训练设置不一致')
    return card, dict(role='validate',seeds=list(SEEDS),parent=str(ck.parent),parent_sha=card['sha256'],
                scene_source=str(scene),grasp=str(grasp),grasp_sha256=settings['grasp_sha256'],
                asset_sha256=settings['cube_sha256'],scene_manifest_sha256=_sha(scene/'artifact_manifest.json'),code=code)


def _rotated_point(body_pose, offset, env_origin):
    """Transform a link-local offset with a physical XYZW body pose into lane-local world meters."""
    import numpy as np
    t = np.asarray(body_pose,dtype=float)
    q = t[3:]
    n = np.linalg.norm(q)
    if t.shape!=(7,) or not (.998 < n < 1.002): raise ValueError('无效的刚体位姿')
    v = np.asarray(offset,dtype=float)
    xyz = q[:3]/n
    c = np.cross(xyz,v)
    return (t[:3]-env_origin+v+2.*(q[3]/n*c+np.cross(xyz,c))).tolist()


class Recorder:
    def __init__(self, stream):
        self.stream=stream
        self.count=0
        self.phase_counts=Counter()
        self.phase_names=()
        self.link_frames={}
        self.origin=None
        self.scene=None
        self.input_phase=None
        self.raw_left=None
        self.executed=None

    def bind(self, scene):
        from assets import fixed_frame
        from phase_contact_policy import PHASES
        self.scene=scene
        self.phase_names=PHASES
        self.table_z=float(scene.cfg['table_z'])
        self.origin=scene.offset[LANE].detach().cpu().numpy().copy()
        self.left=[i for i,n in enumerate(scene.names) if n.startswith('left_') or n.endswith('_L')]
        if len(self.left)!=28:raise ValueError('不是 28 维左侧执行器')
        names=list(scene.robot.shared_metatype.link_names)
        for side in ('left','right'):
            body, offset, _ = fixed_frame(scene.stage,'/World/envs/env_0000/Robot',side+'_palm_link')
            self.link_frames[side+'_palm']=(names.index(body),offset)
            suffix='L' if side=='left' else 'R'
            for part, index in [('shoulder',2), ('elbow',4), ('wrist',7)]:
                self.link_frames[side+'_'+part]=(names.index(f'Link{index}_{suffix}'),[0,0,0])

    def forward(self,_module,args,out):
        # PyTorch forward hook returns None: the production output is untouched.
        x=args[0]
        self.input_phase=x[LANE,-len(self.phase_names)-1:-1].detach().clone()
        self.raw_left=out[LANE].detach().clone()

    def capture(self,step,acted,supported):
        if not bool(acted[LANE]):return
        import numpy as np
        import torch
        s=self.scene
        if step!=self.count or self.raw_left is None or self.executed is None:raise ValueError('缺少动作或帧次序不连续')
        ph=self.input_phase.cpu().tolist()
        if len(ph)!=len(self.phase_names) or ph.count(1.)!=1 or ph.count(0.)!=len(ph)-1:raise ValueError('阶段输入异常')
        phase=self.phase_names[ph.index(1.)]
        # These are readbacks after the original step_physics call. No pose writes/extra physics steps.
        core=s.core()[LANE].double().cpu().numpy()
        q=core[3:7]
        body=s.robot.get_link_transforms()[LANE].double().cpu().numpy()
        rotations=torch.stack([v.get_transforms()[LANE,3:7] for v in s.views.values()]).double().cpu().numpy()
        def unit_rows(a):
            norms=np.linalg.norm(a,axis=-1,keepdims=True)
            if not ((norms>.998)&(norms<1.002)).all():raise ValueError('无效的魔方旋转')
            return a/norms
        rel=float(np.min(np.abs(unit_rows(rotations)@unit_rows(q))))
        impulses=s.hand_impulse[LANE].cpu().tolist()
        contact=s.contacts[LANE].cpu().tolist()
        frame={'step':step,'phase':phase,'cube':core[:3].tolist(),'quat':[q[3],*q[:3]],
               'height_m':core[2]-self.table_z-.045, 'left_impulse':impulses[0],'right_impulse':impulses[1],
               'table_contact':bool(contact[12]>=1e-5),'supported':bool(supported[LANE]),
               'face_angle_deg':None,'layer_deviation_deg':math.degrees(2*math.acos(max(0.,min(1.,rel)))),
               'finger_contacts':[], 'finger_contact_positions_recorded':False, 'finger_impulses_Ns':contact[:10],
               'raw_left_rad':self.raw_left.cpu().tolist(), 'applied_left_rad':self.executed.cpu().tolist()}
        for label,(idx,off) in self.link_frames.items():
            frame[label]=_rotated_point(body[idx],off,self.origin)
        self.stream.write(json.dumps(frame,allow_nan=False,separators=(',',':'))+'\n')
        self.count+=1;self.phase_counts[phase]+=1
        if self.count%30==1:self.stream.flush()
        self.raw_left=self.input_phase=self.executed=None


def observed_validation(sampler, args, recorder):
    """Instrument only the child validation call. The originals still own every state transition."""
    import contact_hold90_cuda_domain as domain
    import contact_hold_policy as policy
    from contextlib import ExitStack
    native_scene=domain.open_scene;native_restore=policy.restored;native_meter=sampler.HoldMeter
    class SceneTap:
        def __init__(self,s):self._scene=s
        def __getattr__(self,name):return getattr(self._scene,name)
        def step_physics(self,cmd):
            result=self._scene.step_physics(cmd)
            recorder.executed=cmd[LANE,recorder.left].detach().clone()
            return result
    class MeterTap(native_meter):
        def update(self,step,acted,supported,local):
            result=super().update(step,acted,supported,local)
            recorder.capture(step,acted,supported)
            return result
    def open_tap(*a,**kw):
        s=native_scene(*a,**kw)
        recorder.bind(s)
        return SceneTap(s)
    with ExitStack() as changes:
        def restore_tap(*a,**kw):
            model=native_restore(*a,**kw)
            hook=model.register_forward_hook(recorder.forward)
            changes.callback(hook.remove)
            return model
        changes.enter_context(patch.object(domain,'open_scene',open_tap))
        changes.enter_context(patch.object(policy,'restored',restore_tap))
        changes.enter_context(patch.object(sampler,'HoldMeter',MeterTap))
        sampler.sample(args)


def result_card(model, report, plan, count, phase_counts, table_z, frames_sha):
    lanes=report['per_lane']
    if (report['role']!='validate' or report['seeds']!=SEEDS or report['parent_sha']!=plan['parent_sha']
        or report['cube_edge_m']!=.09 or report['asset_sha256']!=plan['asset_sha256'] or report['grasp_sha256']!=plan['grasp_sha256']
        or report['physics_device']!='cuda:0' or report['cross_lane_contacts']!=0 or len(lanes)!=len(SEEDS)
        or [r['seed'] for r in lanes]!=SEEDS or any(r['teacher_executed_left_steps'] or r['student_executed_left_steps']!=r['steps'] for r in lanes)):
        raise ValueError('不是所选学生模型的独立物理闭环验证')
    r=lanes[LANE]
    if r['steps']!=count or dict(phase_counts)!=r['phase_steps']:
        raise ValueError('录制帧数或阶段与验证报告不符')
    eid=f"physical-{model['id']}-{SEEDS[LANE]}"
    ep=dict(id=eid,label=f"新物理闭环 · Seed {SEEDS[LANE]}",seed=SEEDS[LANE],step_count=count,hz=30,edge_mm=90,
            origin_model_id=model['id'],controller='student_left_from_reset_scripted_right_and_phases')
    summary=dict(peak_height_m=r['max_height_m'],best_hold_s=r['best_controlled_hold_s'],end_reason=r['end_reason'],
                 whole_turn=r['ever_whole_turn'],face_turn=None,final_table_contact=r['final_table_contact'])
    return dict(kind='dashboard_physical_validation_v1',episode=ep,model_id=model['id'],step_count=count,seed=SEEDS[LANE],lane=LANE,
                panel_seeds=SEEDS,per_lane=lanes,summary=summary,table_z_m=table_z,checkpoint_sha256=plan['parent_sha'],
                trajectory_sha256=frames_sha,face_twist_commanded=False,physics_stepped=True,trace_role='post_action_physical_measurements')


def run(job, model_id, steps):
    import torch
    import contact_hold90_cuda_sample as sampler
    if type(steps)!=int or not 2<=steps<=3500:raise ValueError('最大步数应在 2..3500 范围内')
    require_reservations()
    job=Path(job)
    if job.resolve()!=job or not job.is_dir():raise ValueError('需指定管理器建立的独立绝对目录')
    paths=[job/n for n in ['plan.json','frames.jsonl','worker_result.json','raw_validation','worker_result.tmp','plan.tmp']]
    if any(v.exists() or v.is_symlink() for v in paths):raise ValueError('拒绝覆盖现有评估结果')
    plan_path, frames, final, raw=paths[:4]
    model,plan=make_plan(model_id)
    req_path=job/'request.json'
    if req_path.is_symlink() or req_path.stat().st_size>16384:raise ValueError('非法任务请求')
    req=json.loads(req_path.read_bytes())
    if (req.get('id')!=job.name or req.get('model')!=model_id or req.get('sha256')!=plan['parent_sha']
            or req.get('seed')!=SEEDS[LANE]):
        raise ValueError('任务、冻结模型、种子或目录不一致')
    _write(plan_path,plan)
    # Launch Kit only after reservations and the fixed model/domain have both been verified.
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    from isaac_compat import close_application
    try:
        args=SimpleNamespace(plan=plan_path,output=raw,role='validate',steps=steps)
        with frames.open('x') as stream, torch.no_grad():
            capture=Recorder(stream)
            observed_validation(sampler,args,capture)
        rpt=json.loads((raw/'report.json').read_text())
        if rpt['plan_sha256']!=_sha(plan_path):raise ValueError('物理报告与请求不一致')
        result=result_card(model,rpt,plan,capture.count,capture.phase_counts,capture.table_z,_sha(frames))
        result['plan_sha256']=_sha(plan_path);result['validation_report_sha256']=_sha(raw/'report.json')
        result['worker_source_sha256']=_sha(Path(__file__))
        _write(final,result)
        print('ROLLOUT_COMPLETE',capture.count,'steps',result['summary']['end_reason'],flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:close_application(app)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3500)
    a=p.parse_args();run(a.output,a.model,a.steps)


if __name__=='__main__':main()
