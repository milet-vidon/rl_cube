"""Scoped numerical convergence probe; retain timestep, velocity solver and gates."""
from contextlib import contextmanager

@contextmanager
def position_iterations(count, module=None):
    if count is None:
        yield
        return
    if count not in (32, 64):
        raise ValueError('Position iterations must be 32 or 64')
    if module is None:
        import environment as module
    original = module.physx_config
    def configure(**kwargs):
        return original(**(kwargs | {'min_position_iteration_count': count,
                                    'max_position_iteration_count': 64}))
    module.physx_config = configure
    try:
        yield
    finally:
        module.physx_config = original

def verify_solver(env, expected):
    scene = env.stage.GetPrimAtPath('/physicsScene')
    names = ('solverType', 'minPositionIterationCount', 'maxPositionIterationCount',
             'minVelocityIterationCount', 'enableExternalForcesEveryIteration',
             'enableStabilization')
    actual = {n: scene.GetAttribute('physxScene:' + n).Get() for n in names}
    if actual['minPositionIterationCount'] != expected or actual['maxPositionIterationCount'] != 64:
        raise ValueError('Requested position iteration configuration not applied')
    if actual['solverType'] != 'TGS' or actual['minVelocityIterationCount'] != 4:
        raise ValueError('Unexpected solver or velocity iteration change')
    if env.cfg['physics_hz'] != 240 or env.cfg['control_hz'] != 30:
        raise ValueError('Unexpected simulation or control frequency')
    return {'scene': actual, 'physics_hz': 240, 'control_hz': 30,
            'verified': True, 'scope': 'numerical convergence development probe; no success assertion'}
