"""Bounded job-local memoization for immutable offline CollisionScene geometry.

Every caller predicate still executes. Only exact distance computations repeat.
Use only with freshly constructed, read-only scene geometry and DistanceRequest;
this is not a physics/contact cache and must never wrap a live simulator.
"""
import numpy as np


class ExactJobDistanceCache:
    def __init__(self, scene, *, maximum_entries=100000):
        import coal
        if type(scene.request) is not coal.DistanceRequest:
            raise ValueError('Only original coal DistanceRequest is supported')
        if not isinstance(maximum_entries,int) or not 1 <= maximum_entries <= 100000:
            raise ValueError('Explicit bounded per-job cache required')
        self.scene=scene;self.original=scene.distance;self.maximum_entries=maximum_entries
        self.geometries={};self.reference_geometry={}
        # Strong references prevent Python/native object identity reuse. Dynamic
        # layer Boxes created later are deliberately absent and bypass caching.
        for unused,geometry,*rest in scene.moving+scene.fixed:
            if type(geometry) not in (coal.Convex,coal.Box):continue
            ident=id(geometry);self.geometries[ident]=geometry
            self.reference_geometry[ident]=self._geometry_bytes(geometry)
        self.cache={};self.jobs=0;self.calls=0;self.hits=0;self.misses=0;self.bypasses=0;self.peak_entries=0

    @staticmethod
    def _geometry_bytes(geometry):
        # Convex support vertices and static Box extents are immutable in the
        # reviewed planner. Verify at each job; no geometry setters occur inside
        # a job. The registry keeps the same object and its native topology.
        if hasattr(geometry,'halfSide'):return np.asarray(geometry.halfSide).tobytes()
        return np.asarray(geometry.points()).tobytes()

    def begin_job(self):
        self.cache.clear();self.jobs+=1
        for ident,geometry in self.geometries.items():
            if self._geometry_bytes(geometry)!=self.reference_geometry[ident]:
                raise ValueError('Registered planner geometry changed; exact cache invalid')

    def _key(self,body):
        name,geometry,transform,center,radius=body
        ident=id(geometry)
        if self.geometries.get(ident) is not geometry:return None
        # Restrict layout/type rather than conflating differently evaluated
        # inputs. All current planner mesh transforms satisfy these conditions.
        if (type(transform) is not np.ndarray or type(center) is not np.ndarray or
            transform.dtype!=np.float64 or center.dtype!=np.float64 or
            transform.shape!=(4,4) or center.shape!=(3,) or
            not transform.flags.c_contiguous or not center.flags.c_contiguous or
            type(radius) not in (float,np.float64)):
            return None
        return (name,ident,transform.tobytes(),center.tobytes(),type(radius),float(radius).hex())

    def distance(self,a,b):
        self.calls+=1
        ka,kb=self._key(a),self._key(b)
        if ka is None or kb is None:
            self.bypasses+=1;return self.original(a,b)
        key=(ka,kb)
        try:
            value=self.cache[key]
        except KeyError:
            self.misses+=1;value=self.original(a,b)
            if len(self.cache)<self.maximum_entries:
                self.cache[key]=value;self.peak_entries=max(self.peak_entries,len(self.cache))
            return value
        self.hits+=1
        # Preserve the sole side effect of original CollisionScene.distance.
        # Broad-phase and >=.004 returns leave the preceding pair unchanged.
        if value<.004:self.scene.last_pair=(a[0],b[0],value)
        return value

    def report(self):
        return {'schema':'exact_job_local_distance_cache_v1','jobs':self.jobs,'calls':self.calls,
            'hits':self.hits,'misses':self.misses,'bypasses':self.bypasses,
            'original_distance_evaluations':self.misses+self.bypasses,
            'peak_entries':self.peak_entries,'maximum_entries':self.maximum_entries,
            'registered_immutable_geometries':len(self.geometries),
            'new_or_unknown_geometry_bypassed':True,'all_predicates_executed':True}
