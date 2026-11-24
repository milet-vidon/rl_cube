"""Index-led approach and bounded contact-driven sweep for the held nZ layer."""
import math

import torch
import torch.nn.functional as F

from contact_hold_cuda_teacher_math import log_rot, mv, rotvec
from natural_face90_v5_controller import _align, _dls, face_point
from natural_face90_v5_metrics import rotate


def _point(radial, angle, depth):
    return face_point(torch.full_like(angle, .030), radial, angle, depth)


def pad_near_marker(radial, local_tip, face):
    angle = face.double()
    marker = _point(radial, angle, torch.zeros_like(angle))
    delta = local_tip - marker
    return (delta[:, 2].abs() <= .006) & (delta[:, :2].norm(dim=1) <= .012)


def _direction_samples(device):
    # Equal-area Fibonacci directions in palm coordinates, computed once.
    i = torch.arange(256, dtype=torch.float64, device=device) + .5
    z = 1 - 2*i/len(i)
    radius = (1-z.square()).sqrt()
    angle = i*(math.pi*(3-math.sqrt(5)))
    return torch.stack((radius*angle.cos(), radius*angle.sin(), z), 1)


def leading_axis(vectors, radii, pad_axis, previous, samples):
    """Pick a face-facing hand direction exposing the index ahead of other digits.

    vectors point from passive hand landmarks to the calibrated pad, all in
    palm coordinates. Rounded landmarks are routing proxies, never collision tests.
    """
    candidates = torch.cat((samples.expand(len(vectors),-1,-1), previous[:,None], pad_axis[:,None]), 1)
    margins = (vectors @ candidates.transpose(1,2) - radii[None,:,None]).amin(dim=1)
    quality = margins.clamp_max(.012) + .005*(candidates*previous[:,None]).sum(-1)
    quality = quality.masked_fill((candidates*pad_axis[:,None]).sum(-1) < .45, -torch.inf)
    return candidates[torch.arange(len(vectors),device=vectors.device),quality.argmax(1)]


class TurnGuide:
    def __init__(self, controller):
        self.c = controller
        self.arm = controller.rp.ix
        self.reference = controller.right_ref.clone()
        self.orientation = controller.route_rotation.clone()
        self.samples = _direction_samples(self.reference.device)
        names = list(controller.e.robot.shared_metatype.link_names)
        landmarks = [('Link7_R', (0.,0.,0.), .044)]
        for finger in ('thumb','index','middle','ring','little'):
            for joint in ('MCP','PIP'):
                landmarks.append((f'right_{finger}_{joint}_link', (0.,0.,0.), .014))
            if finger != 'index':
                # Centres of the authored distal rubber meshes in DIP-body coordinates.
                center = (0.,.0245,0.) if finger == 'thumb' else (.0025,0.,.021)
                landmarks.append((f'right_{finger}_DIP_link', center, .017))
        self.links = [names.index(name) for name,_,_ in landmarks]
        self.index_link = names.index('right_index_DIP_link')
        self.offsets = self.reference.new_tensor([offset for _,offset,_ in landmarks])
        self.radii = self.reference.new_tensor([r for _,_,r in landmarks])
        n = len(self.reference)
        self.vectors = self.reference.new_zeros(n,len(landmarks),3)
        self.axis = self.reference.new_zeros(n,3)
        self.sweep_deg = self.reference.new_zeros(n)
        self.clear_z = self.sweep_deg.clone()
        self.start_face = self.sweep_deg.clone()
        self.pad_gap_m = self.sweep_deg.clone()
        self.index_lead_m = self.sweep_deg.clone()
        self.goal_local = self.reference.new_zeros(n,3)
        self.recovering = torch.zeros(n, dtype=torch.bool, device=self.reference.device)
        self.aligned = self.recovering.clone()
        self.drive_contact = self.recovering.clone()
        self.streak = self.sweep_deg.long()
        self.window = self.streak.clone()
        self.lost = self.streak.clone()

    def _plan(self, fresh, target, palm, inward, tip):
        if not fresh.any():
            return
        bodies = self.c.e.robot.get_link_transforms().double()
        passive = bodies[:, self.links]
        points = passive[:,:,:3] + rotate(passive[:,:,3:], self.offsets)
        inv = palm.transpose(1,2)
        vectors = (tip[:,None] - points) @ palm
        protrusion = F.normalize(mv(inv, tip-bodies[:,self.index_link,:3]),dim=1)
        chosen = leading_axis(vectors,self.radii,protrusion,mv(inv,inward),self.samples)
        self.axis = torch.where(fresh[:,None],chosen,self.axis)
        self.vectors = torch.where(fresh[:,None,None],vectors,self.vectors)
        self.orientation = torch.where(fresh[:,None,None],palm,self.orientation)
        self.reference = torch.where(fresh[:,None],target[:,self.arm].double(),self.reference)

    def _path(self, fresh, active, radial, local_tip, orientation_ok, obs):
        face = obs['face_deg'].double()
        touch = obs['right_contact'].bool()
        index = obs['index_contact'].bool()
        reached = obs['contact_goal_reached'].bool()
        good = (obs['support'].bool() & obs['static_contact'].bool() & obs['structure_ok'].bool()
                & obs['allowed_layer_state'].bool() & ~obs['wrong_contact'].bool())
        marker = _point(radial,face,torch.zeros_like(face))
        near = pad_near_marker(radial,local_tip,face)
        lateral = (local_tip[:,:2]-marker[:,:2]).norm(dim=1)
        self.pad_gap_m = marker[:,2]-local_tip[:,2]
        self.aligned &= ~fresh
        self.recovering &= ~fresh
        self.window = torch.where(fresh,0,self.window)
        self.streak = torch.where(fresh,0,self.streak)
        self.lost = torch.where(fresh,0,self.lost)
        self.sweep_deg = torch.where(fresh,face,self.sweep_deg)
        # Barrier/knuckle contact far in front of the pad used to continue for
        # thousands of steps. Back out from the actual touching configuration.
        obstructed = touch & ~(index & near)
        stalled = (self.window>=120) & (face-self.start_face < .5)
        enter_recovery = active & ~reached & ~self.recovering & (obstructed | stalled)
        self.clear_z = torch.where(enter_recovery,torch.minimum(local_tip[:,2]-.014,
                                   torch.full_like(face,-.075)),self.clear_z)
        released = (self.recovering & ~touch & (local_tip[:,2]<=self.clear_z+.003))
        self.recovering = torch.where(active,(self.recovering & ~released | enter_recovery) & ~reached,
                                      self.recovering)
        self.aligned &= ~(self.recovering | (active & ~good))
        ready = active & ~self.recovering & orientation_ok & good & (lateral<=.004)
        self.aligned |= ready
        correct = active & self.aligned & ~self.recovering & good & touch & index & near & ~reached
        self.streak = torch.where(correct,self.streak+1,0)
        self.drive_contact = correct & (self.streak>=3)
        self.start_face = torch.where(self.drive_contact & (self.window==0),face,self.start_face)
        self.window = torch.where(self.drive_contact,
                                  torch.where(self.window>=120,torch.ones_like(self.window),self.window+1),0)
        # Restart the progress window after verified positive object movement.
        window_done = self.drive_contact & (self.window==1)
        self.start_face = torch.where(window_done,face,self.start_face)
        self.lost = torch.where(correct,0,(self.lost+1).clamp_max(6))
        previous = torch.maximum(self.sweep_deg,face)
        bound = torch.minimum(face+16.,torch.full_like(face,18.))
        advance = torch.minimum(previous+.20,bound)
        # Hold the last tangential target across short measurement gaps. After
        # sustained loss, reacquire the material marker without advancing.
        hold_angle = (self.lost<6) & near & good & ~self.recovering & ~reached
        next_angle = torch.where(self.drive_contact,advance,torch.where(hold_angle,self.sweep_deg,face))
        self.sweep_deg = torch.where(active,next_angle,self.sweep_deg)
        standoff = torch.minimum(torch.full_like(face,-.030),self.index_lead_m-.014)
        depth = torch.where(self.aligned & good,torch.full_like(face,.0005),standoff)
        goal = _point(radial,self.sweep_deg,depth)
        pullback = torch.cat((local_tip[:,:2], self.clear_z[:,None]),1)
        goal = torch.where(self.recovering[:,None],pullback,goal)
        goal = torch.where(reached[:,None],_point(radial,face,torch.full_like(face,-.055)),goal)
        self.goal_local = goal
        return goal

    def command(self,target,q,handoff,fresh,learned,latent,core,tip,local_tip,obs):
        c, arm = self.c, self.arm
        q = q.double()
        _, palm, _, _ = c.rp.fk(q[:,arm])
        inward = rotate(core[:,3:],q.new_tensor([0.,0.,1.]))
        self._plan(fresh,target,palm,inward,tip)
        actual_axis = mv(palm.transpose(1,2),inward)
        self.index_lead_m = ((self.vectors*actual_axis[:,None]).sum(-1)-self.radii).amin(1)
        oriented = ((self.axis*actual_axis).sum(-1)>.985) & (self.index_lead_m>=.002)
        goal = self._path(fresh,learned,c.radial,local_tip,oriented,obs)
        # Rim clearance before rotating. Hold wrist orientation while unloading
        # an obstruction, then align the hand from an unloaded configuration.
        desired = rotvec(_align(mv(self.orientation,self.axis),inward)) @ self.orientation
        desired = torch.where((self.recovering & obs['right_contact'].bool())[:,None,None],palm,desired)
        prev = self.reference
        ref_q = q.clone(); ref_q[:,arm]=prev
        ref_tip,_,origins,axes = c.tip.fk(ref_q[:,c.tip.ix])
        _,ref_palm,_,_ = c.rp.fk(prev)
        error = core[:,:3]+rotate(core[:,3:],goal)-tip
        delta = _dls(ref_tip,ref_palm,origins,axes,ref_tip+error,
                     log_rot(desired @ ref_palm.transpose(1,2)),.012)
        delta = c.clearance.steer(prev,delta,ref_tip,origins,axes)
        cap = torch.minimum(c.e.step,c.e.vel/30).expand_as(target)[:,arm].double()
        lim = c.e.limits.expand(len(q),-1,-1)[:,arm].double()
        scale = (cap/delta.abs().clamp_min(1e-12)).amin(1,keepdim=True).clamp_max(1.)
        nominal = (prev+scale*delta).clamp(q[:,arm]-.040,q[:,arm]+.040).clamp(lim[...,0],lim[...,1])
        nominal = c.clearance.project(prev,nominal)
        self.reference = torch.where(learned[:,None],nominal,prev)
        residual = .008*latent.double().tanh()
        residual = torch.where(self.recovering[:,None],0.,residual)
        wanted = (nominal+residual).clamp(lim[...,0],lim[...,1])
        old = target[:,arm].double()
        result = handoff.clone()
        result[:,arm] = c.clearance.project(old,old+(wanted-old).clamp(-cap,cap)).to(result.dtype)
        return torch.where(learned[:,None],result,target)
