"""Explicit Isaac Lab 2/3 boundaries; project observations remain wxyz."""
from importlib.metadata import version

LAB3 = int(version('isaaclab').split('.')[0]) >= 3


def tensor(value):
    """Lab 3 ProxyArray -> zero-copy Torch view; Lab 2 tensors pass through."""
    return value.torch if hasattr(value, 'torch') else value


def quaternion_to_native(wxyz):
    """Convert project / Gf scalar-first quaternions for the selected Lab API."""
    if not LAB3:
        return wxyz
    if isinstance(wxyz, (tuple, list)):
        return type(wxyz)((wxyz[1], wxyz[2], wxyz[3], wxyz[0]))
    return wxyz[..., [1, 2, 3, 0]]


def quaternion_from_native(quat):
    if not LAB3:
        return quat
    if isinstance(quat, (tuple, list)):
        return type(quat)((quat[3], quat[0], quat[1], quat[2]))
    return quat[..., [3, 0, 1, 2]]


def physx_config(**kwargs):
    if LAB3:
        from isaaclab_physx.physics import PhysxCfg
    else:
        from isaaclab.sim import PhysxCfg
    return PhysxCfg(**kwargs)


def simulation_config(*, physx, **kwargs):
    from isaaclab.sim import SimulationCfg
    return SimulationCfg(**kwargs, **{'physics' if LAB3 else 'physx': physx})


def runtime_versions():
    return {name: version(name) for name in ('isaacsim', 'isaaclab', 'torch', 'rsl-rl-lib')}


def prepare_single_world(sim):
    """Declare this already-authored single scene for Lab 3 render/physics backends."""
    if not LAB3:
        return
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner import make_clone_plan, replicate
    from pxr import UsdPhysics
    cfgs = tuple(AssetBaseCfg(prim_path=str(prim.GetPath()))
                 for prim in sim.stage.GetPrimAtPath('/World').GetChildren()
                 if not prim.IsA(UsdPhysics.Scene))
    plan = make_clone_plan(cfgs, ((),), 1, shared_assets=tuple(range(len(cfgs))))
    sim.set_clone_plan(plan)
    replicate(plan)


def normalize_windows_asset_paths(stage):
    """Author Linux-resolvable texture overrides; preserve the source USD files."""
    from pathlib import Path
    from pxr import Sdf
    changes = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or '\\' not in value.path:
                continue
            source = next((spec for spec in attr.GetPropertyStack() if spec.HasDefaultValue()), None)
            if source is None:
                continue
            resolved = Sdf.ComputeAssetPathRelativeToLayer(source.layer, value.path.replace('\\', '/'))
            if not Path(resolved).is_file():
                raise FileNotFoundError(f'Cannot resolve {attr.GetPath()}: {value.path} -> {resolved}')
            attr.Set(Sdf.AssetPath(resolved))
            changes.append({'attribute': str(attr.GetPath()), 'source': value.path, 'resolved': resolved})
    return changes


def close_application(app):
    """Lab 3 close() exits the process; propagate an active exception as failure."""
    import sys
    kwargs = {'wait_for_replicator': False}
    if LAB3:
        kwargs['exit_code'] = 1 if sys.exc_info()[0] else 0
    app.close(**kwargs)
