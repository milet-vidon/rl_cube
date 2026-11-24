"""Per-finger cube contact evidence, without changing collision filtering."""
import numpy as np
import omni.physx
from pxr import PhysicsSchemaTools

FINGERS=('thumb','index','middle','ring','little')

class FingerContactAudit:
    def __init__(self):
        self.reset_step()
        self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.callback)
    def reset_step(self):
        self.impulses={s:{f:0. for f in FINGERS} for s in ['left','right']};self.contacts=[];self.self_contacts=[]
    def callback(self,headers,data):
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0,h.actor1)]
            if all(p.startswith('/World/Robot/') for p in paths):
                for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                    impulse=float(np.linalg.norm(np.asarray(d.impulse)))
                    if impulse>1e-7:self.self_contacts.append({'paths':paths,'impulse_Ns':impulse,'separation_m':float(d.separation),'position_world_m':list(d.position)})
            if not any(p.startswith('/World/Cube') for p in paths):continue
            for side in self.impulses:
                for finger in FINGERS:
                    if not any(f'/World/Robot/{side}_{finger}_' in p for p in paths):continue
                    for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                        impulse=float(np.linalg.norm(np.asarray(d.impulse)))
                        self.impulses[side][finger]+=impulse
                        if impulse>1e-7:self.contacts.append({'side':side,'finger':finger,'impulse_Ns':impulse,'position_world_m':list(d.position),'normal_world':list(d.normal),'separation_m':float(d.separation)})
    def metrics(self):
        return {'finger_contact_impulses':{s:dict(x) for s,x in self.impulses.items()},'finger_contact_points':list(self.contacts),'robot_self_contact_points':list(self.self_contacts)}

from opposition_acceptance import assess_opposition
