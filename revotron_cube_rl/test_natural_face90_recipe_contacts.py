"""Capacity allocation, isolation and rejection checks; no PhysX is started."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
import torch
import natural_face90_recipe_contacts as c


class FixtureScene:
    def __init__(self,source,n):
        self.call=dict(source=source,max_contact_data_count=n*512)
    def unchanged_method(self):return 'unchanged'


def fixture_opener(source,n,grasp):
    from contact_hold_cuda_scene import Scene
    return Scene(source,n)


class Capacity(unittest.TestCase):
    def test_supported_batch_sizes_and_bad_values(self):
        for n in (2,4,8,16,32):
            self.assertEqual(c.check_capacity_source(n)['records_per_view'],n*4096)
        for n in (0,33,-1,True,8.0):
            with self.assertRaises(ValueError):c.capacity_plan(n)

    def test_private_scene_does_not_patch_original_class_or_globals(self):
        original=FixtureScene.__init__;before=dict(fixture_opener.__globals__)
        cls=c.private_scene_type(FixtureScene)
        modified=c.private_opener(fixture_opener,cls)
        self.assertEqual(modified('fixture',32,None).call['max_contact_data_count'],131072)
        self.assertEqual(FixtureScene('fixture',32).call['max_contact_data_count'],16384)
        self.assertIs(FixtureScene.__init__,original)
        self.assertEqual(fixture_opener.__globals__,before)
        self.assertIs(cls.unchanged_method,FixtureScene.unchanged_method)

    def test_real_constructor_specialization_changes_only_allocation(self):
        # Read/compile the actual constructor without executing Kit imports.
        path=Path(__file__).with_name('contact_hold_cuda_scene.py')
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Scene')
        init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
        scope={};exec(compile(ast.Module(body=[init],type_ignores=[]),str(path),'exec'),scope)
        base=type('SourceOnlyScene',(),{'__init__':scope['__init__']})
        specialized=c.private_scene_type(base)
        # The only new global used by the private constructor is the allocation
        # function; the pinned base constructor still contains its old capacity.
        self.assertIs(specialized.__init__.__globals__['recipe_capacity'],c.capacity_plan)
        self.assertNotIn('recipe_capacity',base.__init__.__globals__)
        self.assertEqual(inspect.getsource(base.__init__).count(c.OLD_ALLOCATION),1)
        from contact_hold90_cuda_domain import _source
        _source('contact_hold_cuda_scene.py')

    def test_capacity_readback_rejects_clamped_or_missing_sensor_view(self):
        def view(cap):return NS(sensor_count=2,max_contact_data_count=cap)
        scene=NS(contact_views=[('robot',view(131072),[0,1]),('cube',view(131072),[0,1])])
        receipt=c.verify_capacity(scene,c.capacity_plan(32));self.assertEqual(len(receipt['views']),2)
        scene.contact_views[0][1].max_contact_data_count=16384
        with self.assertRaisesRegex(ValueError,'readback'):c.verify_capacity(scene,c.capacity_plan(32))
        scene.contact_views=[]
        with self.assertRaisesRegex(ValueError,'Missing'):c.verify_capacity(scene,c.capacity_plan(32))

    def test_step_delegates_once_and_preserves_command_identity(self):
        command=object();calls=[]
        scene=NS(step_physics=lambda x:calls.append(x))
        c.step_physics(scene,command,Path('/nonexistent'))
        self.assertEqual(calls,[command])

    def test_exhaustion_remains_fatal_with_counts_and_invalid_span(self):
        def fail(_):raise RuntimeError('Contact buffer exhausted')
        raw=(None,None,None,None,torch.tensor([6,4]),torch.tensor([0,6]))
        view=NS(max_contact_data_count=8,get_raw_contact_data=lambda dt:raw)
        scene=NS(step_physics=fail,contact_views=[('robot',view,[0,1])])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'Contact buffer exhausted') as ctx:
                c.step_physics(scene,None,Path(temp))
            r=json.loads((Path(temp)/'contact_capacity_failure.json').read_text())
            self.assertEqual(r['views'][0]['reported_total'],10)
            self.assertTrue(r['views'][0]['invalid_span'])
            self.assertIsInstance(ctx.exception.__cause__,RuntimeError)

    def test_other_errors_do_not_get_reclassified_or_ignored(self):
        error=RuntimeError('CUDA out of memory')
        def fail(_):raise error
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError) as ctx:c.step_physics(NS(step_physics=fail),None,Path(temp))
            self.assertIs(ctx.exception,error);self.assertEqual(list(Path(temp).iterdir()),[])

    def test_new_capacity_source_is_part_of_checkpoint_fingerprint(self):
        from natural_face90_recipe_long import fingerprint,worker_software
        self.assertIn('natural_face90_recipe_contacts.py',worker_software(fingerprint()))


if __name__=='__main__':unittest.main()
