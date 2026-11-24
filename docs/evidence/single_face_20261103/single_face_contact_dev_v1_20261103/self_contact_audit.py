"""Record self contacts without changing collision filtering or physics."""
import numpy as np
from pxr import PhysicsSchemaTools
import omni.physx

class SelfContactAudit:
    def __init__(self):
        self.pairs={};self.subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.callback)
    def reset_step(self):self.pairs.clear()
    def callback(self,headers,data):
        for h in headers:
            paths=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0,h.actor1)]
            if all(x.startswith('/World/Robot/') for x in paths):
                impulse=sum(float(np.linalg.norm(np.asarray(d.impulse))) for d in data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data])
                if impulse>1e-8:
                    key=' | '.join(paths);self.pairs[key]=self.pairs.get(key,0.)+impulse
    def metrics(self):
        body=sum(v for k,v in self.pairs.items() if '/World/Robot/world' in k)
        return {'self_contact_pairs':dict(self.pairs),'arm_body_impulse':body,'self_contact_impulse':sum(self.pairs.values())}
