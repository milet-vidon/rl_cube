"""Scoped factory extension; bind only right-hand contact material before reset."""
import articulated_cube_env as dynamic
from right_contact_material import validate_friction,apply_right_material

class RightContactEnv(dynamic.DynamicPickupEnv):
    def __init__(self,run_dir,right_hand_friction,**kwargs):
        coefficients=validate_friction(right_hand_friction)
        if kwargs.get('hand_friction') is not None:raise ValueError('Right-only and bilateral hand materials are mutually exclusive')
        original=dynamic.DynamicCubeHandle
        def build(stage,*args,**options):
            handle=original(stage,*args,**options)
            handle.model.update(apply_right_material(stage,coefficients))
            return handle
        dynamic.DynamicCubeHandle=build
        try:super().__init__(run_dir,**kwargs)
        finally:dynamic.DynamicCubeHandle=original
