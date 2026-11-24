"""Load the single operator world through the normal USD-to-PhysX attach path."""
import inspect
import textwrap

from contact_hold90_cuda_domain import _specialize
from natural_face90_recipe_contacts import (
    OLD_ALLOCATION, capacity_plan, check_capacity_source, private_opener, verify_capacity,
)


def single_world_type(base):
    # The shared scene source is pinned by the R27 archive. Keep its ordinary
    # prepare_single_world/reset pair. The multi-world callback between them
    # excludes /World/envs from parsing, including the only authored robot.
    constructor = textwrap.dedent(inspect.getsource(base.__init__))
    start = constructor.index("    # Same native env-ID replication API")
    end = constructor.index("    self.paths={}", start)
    old_count_check = "if self.robot.count!=n:raise RuntimeError(f'Expected {n} robot articulations, got {self.robot.count}')"
    init = _specialize(base.__init__, [
        (constructor[start:end], ''),
        (OLD_ALLOCATION, "max_contact_data_count=demo_capacity(n)['records_per_view']"),
        (old_count_check, 'if getattr(self.robot, "count", 0) != 1:'
         ' raise RuntimeError("PhysX did not load exactly one robot at /World/envs/env_0000/Robot/root_joint")'),
    ], {'demo_capacity': capacity_plan})
    return type('SingleWorldDemoScene', (base,), {'__init__': init})


def open_demo_scene(source, grasp):
    from contact_hold90_cuda_domain import open_scene
    from contact_hold_cuda_scene import Scene

    capacity = check_capacity_source(1)
    scene = private_opener(open_scene, single_world_type(Scene))(source, 1, grasp)
    scene.domain_receipt['raw_contact_capacity'] = verify_capacity(scene, capacity)
    scene.domain_receipt['scene_loading'] = 'single_authored_world'
    print(f'DEMO_PHYSX_READY robots={scene.robot.count} cube_bodies={len(scene.views)}', flush=True)
    return scene
