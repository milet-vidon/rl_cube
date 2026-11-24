import unittest
from types import SimpleNamespace
import numpy as np
import coal
from opposed_distance_cache_v1 import ExactJobDistanceCache

class ExactCacheTests(unittest.TestCase):
    def fixture(self):
        shapes=[coal.Box(1.,1.,1.),coal.Box(2.,2.,2.)]
        bodies=[(str(i),shape,np.eye(4),np.zeros(3),float(i+1)) for i,shape in enumerate(shapes)]
        scene=SimpleNamespace(request=coal.DistanceRequest(),moving=[bodies[0]],fixed=[bodies[1]],last_pair=None,calls=0)
        def distance(a,b):
            scene.calls+=1
            d=float(a[2][0,3]-b[2][0,3])
            if d<.004:scene.last_pair=(a[0],b[0],d)
            return d
        scene.distance=distance
        cache=ExactJobDistanceCache(scene);cache.begin_job()
        return scene,cache,bodies

    def test_exact_hit_preserves_float_bits_and_near_pair_side_effect(self):
        s,c,(a,b)=self.fixture();first=c.distance(a,b);s.last_pair=('stale',)
        second=c.distance(a,b)
        self.assertEqual(np.float64(first).tobytes(),np.float64(second).tobytes())
        self.assertEqual(s.last_pair,('0','1',first));self.assertEqual(s.calls,1)

    def test_far_hit_retains_preceding_pair_without_invented_update(self):
        s,c,(a,b)=self.fixture();a[2][0,3]=.1;c.distance(a,b);s.last_pair=('keep',)
        c.distance(a,b);self.assertEqual(s.last_pair,('keep',));self.assertEqual(s.calls,1)

    def test_each_transform_center_radius_name_and_order_participates(self):
        s,c,(a,b)=self.fixture();c.distance(a,b)
        alternatives=[]
        t=a[2].copy();t[1,3]=.001;alternatives.append((a[0],a[1],t,a[3],a[4]))
        p=a[3].copy();p[0]=.001;alternatives.append((a[0],a[1],a[2],p,a[4]))
        alternatives.append((a[0],a[1],a[2],a[3],a[4]+.001))
        alternatives.append(('other',a[1],a[2],a[3],a[4]))
        for altered in alternatives:c.distance(altered,b)
        c.distance(b,a);self.assertEqual(s.calls,6)

    def test_new_identical_box_is_not_reused_by_shape_or_native_address(self):
        s,c,(a,b)=self.fixture();c.distance(a,b)
        new=(a[0],coal.Box(1.,1.,1.),a[2],a[3],a[4])
        c.distance(new,b);c.distance(new,b)
        self.assertEqual(s.calls,3);self.assertEqual(c.report()['bypasses'],2)
        self.assertIs(c.geometries[id(a[1])],a[1])

    def test_nondefault_dtype_and_layout_bypass(self):
        s,c,(a,b)=self.fixture()
        for t in (a[2].astype(np.float32),np.asfortranarray(a[2])):
            altered=(a[0],a[1],t,a[3],a[4]);c.distance(altered,b);c.distance(altered,b)
        self.assertEqual(s.calls,4);self.assertEqual(c.report()['bypasses'],4)

    def test_job_clear_and_mutated_registered_geometry_fail_closed(self):
        s,c,(a,b)=self.fixture();c.distance(a,b);c.begin_job();c.distance(a,b)
        self.assertEqual(s.calls,2)
        a[1].halfSide=np.array([.7,.8,.9])
        with self.assertRaises(ValueError):c.begin_job()

    def test_bounded_capacity_never_skips_original_result(self):
        s,c,(a,b)=self.fixture();c.maximum_entries=1
        c.distance(a,b);a[2][0,3]=.1;c.distance(a,b);c.distance(a,b)
        self.assertEqual(s.calls,3);self.assertEqual(c.report()['peak_entries'],1)

if __name__=='__main__':unittest.main()
