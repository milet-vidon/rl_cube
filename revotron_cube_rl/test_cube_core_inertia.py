import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cube_core_inertia import source_core_parameters,source_core_inertia
XML='<mujoco><default><default class="core"><geom size="0.01 0 0"/></default></default><worldbody><body name="core"><geom class="core" mass="0.00253704"/></body></worldbody></mujoco>'
class SourceCoreInertiaTests(unittest.TestCase):
    def parameters(self,xml=XML,mass=.002537037):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cube.xml';p.write_text(xml);return source_core_parameters(p,mass)
    def test_source_sphere_and_mass_rounding(self):
        r=self.parameters();self.assertAlmostEqual(r['diagonal_inertia_kg_m2'][0],1.0148148e-7,places=15)
        self.assertEqual(r['runtime_core_mass_kg'],.002537037)
    def test_mismatched_mass_rejected(self):
        with self.assertRaises(ValueError):self.parameters(mass=.003)
    def test_non_sphere_rejected(self):
        with self.assertRaises(ValueError):self.parameters(XML.replace('size="','type="box" size="'))
    def test_nonpositive_radius_rejected(self):
        with self.assertRaises(ValueError):self.parameters(XML.replace('0.01 0 0','-0.01 0 0'))
    def test_factory_restored_on_exception(self):
        original=lambda *a:SimpleNamespace(model={});module=SimpleNamespace(DynamicCubeHandle=original)
        with patch('cube_core_inertia.apply_source_core_inertia',side_effect=ValueError('probe')):
            with self.assertRaisesRegex(ValueError,'probe'):
                with source_core_inertia(True,module):module.DynamicCubeHandle(None,'cube.usda')
        self.assertIs(module.DynamicCubeHandle,original)
    def test_disabled_does_not_replace_factory(self):
        original=object();module=SimpleNamespace(DynamicCubeHandle=original)
        with source_core_inertia(False,module):self.assertIs(module.DynamicCubeHandle,original)
if __name__=='__main__':unittest.main()
