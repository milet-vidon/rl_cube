"""Detailed self-contact diagnostics without changing collision filtering."""
import numpy as np
from pxr import PhysicsSchemaTools
from self_contact_audit import SelfContactAudit


class DetailedSelfContactAudit(SelfContactAudit):
    def __init__(self):
        self.points=[]
        super().__init__()

    def reset_step(self):
        super().reset_step()
        self.points.clear()

    def callback(self, headers, data):
        super().callback(headers, data)
        for h in headers:
            actors=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0,h.actor1)]
            if not all(x.startswith('/World/Robot/') for x in actors):
                continue
            colliders=[]
            for key in ('collider0','collider1'):
                value=getattr(h,key,None)
                colliders.append(str(PhysicsSchemaTools.intToSdfPath(value)) if value is not None else None)
            for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]:
                impulse=float(np.linalg.norm(np.asarray(d.impulse)))
                if impulse>1e-8:
                    self.points.append({'actors':actors,'colliders':colliders,
                        'position_world_m':np.asarray(d.position).tolist(),'impulse_Ns':impulse})

    def metrics(self):
        return super().metrics() | {'self_contact_points':list(self.points)}
