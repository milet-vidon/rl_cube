"""Recipe-only raw contact capacity; never changes collision or success gates.

32 lanes exhausted the legacy view's 32*512 records on 2026-11-18. Allocate
eight times that capacity before creating sensors. Like the existing 90 mm
domain, specialize verified functions with PRIVATE globals; the shared Scene,
its reducer, the R27 assets and every historical trainer remain unchanged.
"""
import inspect
import json

RECORDS_PER_LANE = 4096
OLD_ALLOCATION = 'max_contact_data_count=n*512'
SCENE_IMPORT = 'from contact_hold_cuda_scene import Scene'


def capacity_plan(lanes):
    if type(lanes) is not int or not 1 <= lanes <= 32:
        raise ValueError('Recipe contact capacity requires 1..32 lanes')
    return dict(schema='recipe_raw_contact_capacity_v1',lanes=lanes,
                records_per_lane=RECORDS_PER_LANE,records_per_view=lanes*RECORDS_PER_LANE)


def check_capacity_source(lanes):
    """Read-only preflight; does not import Kit or initialize a simulator."""
    from contact_hold90_cuda_domain import ROOT, _source, open_scene
    _source('contact_hold_cuda_scene.py')
    if ((ROOT/'contact_hold_cuda_scene.py').read_text().count(OLD_ALLOCATION)!=1 or
        inspect.getsource(open_scene).count(SCENE_IMPORT)!=1):
        raise ValueError('Contact allocation specialization no longer matches source')
    return capacity_plan(lanes)


def private_scene_type(base):
    from contact_hold90_cuda_domain import _specialize
    init=_specialize(base.__init__,[(OLD_ALLOCATION,
        "max_contact_data_count=recipe_capacity(n)['records_per_view']")],
        {'recipe_capacity':capacity_plan})
    return type('RecipeContactScene',(base,),{'__init__':init})


def private_opener(original,scene_type):
    from contact_hold90_cuda_domain import _specialize
    return _specialize(original,[(SCENE_IMPORT,'Scene = recipe_scene_type')],
                       {'recipe_scene_type':scene_type})


def verify_capacity(scene,plan):
    result=[]
    for kind,view,lanes in scene.contact_views:
        if kind not in ('robot','cube') or len(lanes)!=view.sensor_count:
            raise ValueError('Unexpected contact sensor coverage')
        if int(view.max_contact_data_count)!=plan['records_per_view']:
            raise ValueError('Raw contact capacity readback mismatch: '+kind)
        result.append(dict(kind=kind,sensors=view.sensor_count,
                           records=int(view.max_contact_data_count)))
    if sorted(x['kind'] for x in result)!=['cube','robot']:
        raise ValueError('Missing/duplicate raw contact view')
    return dict(plan,views=result)


def open_scene(source,n,grasp):
    plan=check_capacity_source(n)
    from contact_hold90_cuda_domain import open_scene as original
    from contact_hold_cuda_scene import Scene
    scene=private_opener(original,private_scene_type(Scene))(source,n,grasp)
    receipt=verify_capacity(scene,plan)
    scene.domain_receipt['raw_contact_capacity']=receipt
    print('RECIPE_RAW_CONTACT_CAPACITY',json.dumps(receipt),flush=True)
    return scene


def step_physics(scene,command,out):
    """Keep the original fail-closed reducer; enrich capacity errors only."""
    try:
        scene.step_physics(command)
    except RuntimeError as error:
        if not str(error).startswith('Contact buffer exhausted'):
            raise
        details=[]
        for kind,view,_ in scene.contact_views:
            info=dict(kind=kind,capacity=int(view.max_contact_data_count))
            try:
                raw=view.get_raw_contact_data(1.)
                counts=raw[4].reshape(-1).long();starts=raw[5].reshape(-1).long()
                info.update(reported_total=int(counts.sum()),max_sensor_contacts=int(counts.max()),
                    negative_count=bool((counts<0).any()),
                    invalid_span=bool(((counts>0)&((starts<0)|(starts+counts>info['capacity']))).any()))
            except Exception as diagnostic_error:
                info['diagnostic_error']=str(diagnostic_error)[:240]
            details.append(info)
        message=json.dumps(dict(error='Contact buffer exhausted',views=details),ensure_ascii=False)
        # Never accept truncated impulses, drop a sensor, reset a closed lane,
        # or use this incomplete batch for PPO. Preserve the original failure.
        try:
            with (out/'contact_capacity_failure.json').open('x') as f:f.write(message+'\n')
        except OSError:
            pass
        raise RuntimeError('Contact buffer exhausted; '+message) from error
