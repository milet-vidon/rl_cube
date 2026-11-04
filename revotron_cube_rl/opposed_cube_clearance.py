"""Collision residuals for right-grasp refinement; final path gates still decide."""
import numpy as np
import coal
from scipy.spatial.transform import Rotation

class CubeClearance:
    def __init__(self, scene):
        self.scene = scene
        self.bodies = []
        self.turn_rotations = [scene.C @ Rotation.from_euler('z', -a, degrees=True).as_matrix() @ scene.C.T
                               for a in np.linspace(0.,getattr(scene,'turn_degrees',0.),4)[1:]] if getattr(scene,'turn_degrees',0.) else []
        for name, size, center in (
            ('stationary_layers', (.057,.057,.038), (.0,.0,.0095)),
            ('target_nZ_layer', (.057,.057,.019), (.0,.0,-.019)),
        ):
            transform = np.eye(4)
            transform[:3,:3] = scene.C
            transform[:3,3] = scene.center + scene.C @ np.array(center)
            self.bodies.append((name, coal.Box(*size), transform, np.zeros(3), np.linalg.norm(np.array(size)/2)))

    @staticmethod
    def minimum_distance(link, body):
        fingertip = any(link.startswith('right_' + f + '_DIP') for f in ('thumb','index','middle','ring','little'))
        # Original final gate allows -2 mm only for target-layer DIP contact.
        # Optimize to a 0.5 mm tighter clearance, without changing that gate.
        return -.0015 if fingertip and body == 'target_nZ_layer' else .0025

    def residual(self, shapes):
        errors = [600 * max(self.minimum_distance(a[0],b[0])-self.scene.distance(a,b),0.)
                  for a in shapes for b in self.bodies]
        # Fingers move rigidly with the palm in the planned turn. Include their
        # swept geometry against stationary layers; full IK/path checks follow.
        for rotation in self.turn_rotations:
            for shape in shapes:
                if not shape[0].startswith('right_'): continue
                name, mesh, transform, center, radius = shape
                moved = transform.copy()
                moved[:3,:3] = rotation @ transform[:3,:3]
                moved[:3,3] = self.scene.center + rotation @ (transform[:3,3]-self.scene.center)
                candidate = (name,mesh,moved,center,radius)
                errors.append(600 * max(.0025-self.scene.distance(candidate,self.bodies[0]),0.))
        return errors
