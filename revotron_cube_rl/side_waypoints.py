"""BC of joint waypoint maps. At execution the network supplies all joint goals;
contact/position gates supervise phase progress. This is not end-to-end PPO.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent

def features(x):
    a,b=x.unbind(-1)
    return torch.stack([torch.ones_like(a),a,b,a*a,a*b,b*b,a**3,a*a*b,a*b*b,b**3],-1)

class WaypointNet(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean));self.register_buffer('scale',torch.as_tensor(scale))
        self.skip=nn.Linear(10,mean.numel(),bias=False)
        self.residual=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,mean.numel()))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,x):
        y=self.skip(features(x))+self.residual(x)
        return self.mean+self.scale*y.reshape(-1,*self.mean.shape)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','train']);p.add_argument('--name',required=True);p.add_argument('--data');p.add_argument('--epochs',type=int,default=2000);a=p.parse_args()
    run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False);started=time.monotonic();torch.manual_seed(1945)
    if a.mode=='generate':
        from side_grasp import SidePlanner
        planner=SidePlanner();xs=[];ys=[];splits=[];metadata=None
        train_axis=np.linspace(-.017,.017,7);val_axis=(train_axis[:-1]+train_axis[1:])/2
        for split,axis in [('training',train_axis),('geometry_validation',val_axis)]:
            for dx in axis:
                for dy in axis:
                    xyz=np.array([.48+dx,dy,.9285]);segments=planner.plan(xyz,.096,.025)
                    xs.append([dx/.017,dy/.017]);ys.append(np.stack([s[2] for s in segments]));splits.append(split)
                    if metadata is None:metadata={'names':[s[0] for s in segments],'steps':[s[3] for s in segments],'home':segments[0][1].tolist()}
                    if len(xs)%10==0:print('PLANS',len(xs),split,flush=True)
        np.savez_compressed(run/'plans.npz',x=np.array(xs,dtype=np.float32),y=np.array(ys,dtype=np.float32),split=np.array(splits))
        (run/'metadata.json').write_text(json.dumps(metadata|{'parameters':{'width':.096,'palm_offset':.025,'wrist_deg':35.,'tilt_deg':15.,'finger_mode':'curled'},'source_sha256':{f:sha(ROOT/f) for f in ['side_grasp.py','side_control.py','kinematics.py','config.json']},'label_kind':'offline IK teacher plans; not claims of physical success at each grid point','elapsed_s':time.monotonic()-started},indent=2))
        return
    from side_gate import require_side_gate
    gate=require_side_gate()
    data=Path(a.data);metadata=json.loads((data.parent/'metadata.json').read_text())
    for f,h in metadata['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Stale geometric labels: '+f)
    d=np.load(data);train=d['split']=='training';valid=~train
    x=torch.tensor(d['x'][train]);y=torch.tensor(d['y'][train]);vx=torch.tensor(d['x'][valid]);vy=torch.tensor(d['y'][valid])
    mean=y.mean(0);scale=y.std(0).clamp_min(.02);scale[(y.max(0).values-y.min(0).values)<1e-6]=0.
    net=WaypointNet(mean,scale);labels=((y-mean)/scale.clamp_min(.02)).flatten(1)
    with torch.no_grad():net.skip.weight.copy_(torch.linalg.lstsq(features(x),labels).solution.T)
    opt=torch.optim.Adam(net.parameters(),lr=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6)
    history=[];best=float('inf')
    def save_model(epoch):
        torch.save({'model':net.state_dict(),'mean':mean,'scale':scale,'metadata':metadata,'algorithm':'BC waypoint regressor + contact/arrival phase supervision','training_labels_sha256':sha(data),'side_teacher_gate_sha256':sha(ROOT/'side_teacher_gate.json'),'config_sha256':sha(ROOT/'config.json'),'inference_code_sha256':{f:sha(ROOT/f) for f in ['side_waypoints.py','side_control.py','side_policy_evaluate.py','self_contact_audit.py']},'selected_epoch':epoch,'success_claim':False},run/'checkpoint.pt')
    with torch.no_grad():initial=(net(vx)-vy).abs();best=float(initial.square().mean())
    history.append({'epoch':-1,'geometry_validation_rmse_rad':best**.5,'geometry_validation_max_joint_error_rad':float(initial.max()),'kind':'fitted cubic initialization'})
    print('BC_INITIAL',history[-1],flush=True);save_model(-1)
    for epoch in range(a.epochs):
        loss=((net(x)-y)/.001).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sched.step()
        if epoch%100==0 or epoch==a.epochs-1:
            with torch.no_grad():err=(net(vx)-vy).abs();val=float(err.square().mean());max_error=float(err.max())
            entry={'epoch':epoch,'train_loss':float(loss.detach()),'validation_rmse_rad':val**.5,'validation_max_joint_error_rad':max_error};history.append(entry);print('BC',entry,flush=True)
            if val<best:
                best=val
                save_model(epoch)
            (run/'result.json').write_text(json.dumps({'history':history,'elapsed_s':time.monotonic()-started,'physical_success_claim':False,'training_plans':int(train.sum()),'geometry_validation_plans':int(valid.sum())},indent=2))
    (run/'training_config.json').write_text(json.dumps(vars(a)|{'seed':1945,'training_device':'cpu','architecture':'cubic skip + 2x64 tanh residual neural waypoint map','actor_input':'initial cube XY; all joint goals predicted; contact and proprioception used by phase supervisor'},indent=2))
    (run/'config.json').write_bytes((ROOT/'config.json').read_bytes())
    for f in ['side_waypoints.py','side_control.py','side_policy_evaluate.py','self_contact_audit.py']:(run/f).write_bytes((ROOT/f).read_bytes())

if __name__=='__main__':main()
