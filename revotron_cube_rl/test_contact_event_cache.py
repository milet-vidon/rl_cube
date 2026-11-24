"""Exact offline replay of unchanged callback ASTs in private namespaces.

This proves Python consumer equivalence for synthetic payloads, not native
PhysX event lifetime/order or runtime factory integration.
"""
from pathlib import Path
import ast,gc,hashlib,json,struct,time,unittest,weakref
from types import SimpleNamespace
import numpy as np
from contact_event_cache import ContactMultiplexer,EventCache,CachedData,_active

class NativeVector:
    conversions=0
    def __init__(self,values,dtype=np.float32,fortran=False):self.values=np.array(values,dtype=dtype,order='F' if fortran else 'C')
    def __array__(self,dtype=None,copy=None):
        type(self).conversions+=1
        return np.array(self.values,dtype=dtype,copy=True,order='K')
    def __iter__(self):return iter(self.values.tolist())

class NativePoint:
    def __init__(self,impulse,dtype=np.float32):self._impulse=impulse;self.dtype=dtype;self.separation=-.000001
    @property
    def impulse(self):return NativeVector(self._impulse,self.dtype)
    @property
    def position(self):return NativeVector([.5,.02,1.25],self.dtype)
    @property
    def normal(self):return NativeVector([0.,1.,0.],self.dtype)

PATHS={1:'/World/Robot/left_index_DIP_link',2:'/World/Cube/nZ_pX',3:'/World/Robot/right_middle_DIP_link',4:'/World/Robot/left_ring_DIP_link',5:'/World/Table',6:'/World/Ground',7:'/World/Robot/world',8:'/World/Robot/colliderA',9:'/World/Robot/colliderB'}
class PathToken:
    def __init__(self,value):self.value=value
    def __str__(self):return self.value
def resolver(value):return PathToken(PATHS[value])

def callbacks(numpy_namespace,schema_namespace,root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    namespace={'np':numpy_namespace,'PhysicsSchemaTools':schema_namespace,'FINGERS':('thumb','index','middle','ring','little')};hashes={}
    selected=[('environment.py','CubePickupEnv',('_on_contact',)),('finger_contact_audit.py','FingerContactAudit',('reset_step','callback','metrics')),('self_contact_audit.py','SelfContactAudit',('reset_step','callback','metrics')),('palm_contact_audit.py','DetailedSelfContactAudit',('reset_step','callback','metrics')),('face_push_runtime.py','LayerContacts',('callback',)),('opposed_grasp_runtime.py','OppositionContacts',('callback',))]
    for file,name,methods in selected:
        tree=ast.parse((root/file).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        body=[n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in methods]
        for method in body:hashes[file+':'+method.name]=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()
        bases=node.bases if name in ('DetailedSelfContactAudit','OppositionContacts') else []
        # Method AST nodes are reused without rewriting their operations.
        clone=ast.ClassDef(name=name,bases=bases,keywords=[],body=body,decorator_list=[]);module=ast.fix_missing_locations(ast.Module(body=[clone],type_ignores=[]));exec(compile(module,str(root/file),'exec'),namespace)
    env=namespace['CubePickupEnv']();env._contact_paths={};env.contact_impulse=np.zeros(2);env.table_impulse=0.;env.robot_table_impulse=0.;env.contact_events=0;env.pair_impulses={}
    finger=namespace['FingerContactAudit']();finger.reset_step()
    detailed=namespace['DetailedSelfContactAudit']();detailed.pairs={};detailed.points=[];detailed.reset_step()
    opposition=namespace['OppositionContacts']();opposition.points=[]
    def snapshot():return {'environment':{'contact_impulse':env.contact_impulse.tolist(),'table_impulse':env.table_impulse,'robot_table_impulse':env.robot_table_impulse,'contact_events':env.contact_events,'pair_impulses':dict(env.pair_impulses),'paths':dict(env._contact_paths)},'finger':finger.metrics(),'self':detailed.metrics(),'opposition':list(opposition.points)}
    def reset():env.contact_impulse[:]=0.;env.table_impulse=env.robot_table_impulse=0.;finger.reset_step();detailed.reset_step();opposition.points.clear()
    return [env._on_contact,finger.callback,detailed.callback,opposition.callback],snapshot,reset,hashes

def fixture(points_per_header=8):
    pairs=[(1,2),(3,2),(1,4),(7,5),(6,5)];headers=[];data=[]
    values=[0.,np.nextafter(1e-8,0),1e-8,np.nextafter(1e-8,1),np.nextafter(1e-7,0),1e-7,np.nextafter(1e-7,1),.001]
    for a,b in pairs:
        headers.append(SimpleNamespace(actor0=a,actor1=b,collider0=8,collider1=9,contact_data_offset=len(data),num_contact_data=points_per_header))
        for i in range(points_per_header):data.append(NativePoint([values[i%len(values)],(-1.)**i*values[i%len(values)]/3.,0.],np.float32 if i%2 else np.float64))
    return headers,data

def assert_exact(test,a,b):
    test.assertIs(type(a),type(b))
    if isinstance(a,float):test.assertEqual(struct.pack('!d',a),struct.pack('!d',b))
    elif isinstance(a,dict):
        test.assertEqual(list(a),list(b))
        for key in a:assert_exact(test,a[key],b[key])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_exact(test,x,y)
    else:test.assertEqual(a,b)

class CacheTests(unittest.TestCase):
    def test_frozen_callback_outputs_exact(self):
        headers,data=fixture();base,base_snapshot,_,hashes=callbacks(np,SimpleNamespace(intToSdfPath=resolver));mux=ContactMultiplexer(resolver);cached,snapshot,_,same=callbacks(mux.numpy_proxy,mux.schema_proxy);tokens=[mux.subscribe(c) for c in cached]
        self.assertEqual(hashes,same);self.assertIs(mux.numpy_proxy.linalg.norm,np.linalg.norm)
        for callback in base:callback(headers,data)
        mux.dispatch(headers,data);assert_exact(self,base_snapshot(),snapshot());self.assertGreater(mux.last_stats['asarray_hits'],0);self.assertGreater(mux.last_stats['path_hits'],0)
    def test_default_copy_dtype_layout_and_kwargs_bypass(self):
        cache=EventCache();value=cache.register(NativeVector([[1.,2.,3.],[4.,5.,6.]],np.float32,True));token=_active.set(cache);mux=ContactMultiplexer(resolver)
        try:
            a=mux.numpy_proxy.asarray(value);b=mux.numpy_proxy.asarray(value);c=mux.numpy_proxy.array(value)
            for result in (a,b,c):self.assertEqual(result.dtype,np.dtype('float32'));self.assertTrue(result.flags.f_contiguous)
            self.assertFalse(np.shares_memory(a,b));a[0,0]=99;self.assertEqual(b[0,0],1.)
            self.assertEqual(cache.stats['array_misses'],1);self.assertEqual(cache.stats['asarray_misses'],1)
            before=dict(cache.stats);d=mux.numpy_proxy.asarray(value,dtype=np.float64);self.assertEqual(d.dtype,np.dtype('float64'));self.assertEqual(cache.stats,before)
            raw=np.arange(3.);cache.register(raw);self.assertIs(mux.numpy_proxy.asarray(raw),raw)
        finally:_active.reset(token);cache.clear()
    def test_vector_identity_lifetime_and_slice_order(self):
        cache=EventCache();data=CachedData([NativePoint([1,2,3]),NativePoint([4,5,6])],cache);a=data[0].impulse;self.assertIs(data[0].impulse,a);weak=weakref.ref(a);del a
        self.assertIs(data[:][0],data[0]);self.assertIs(data[-1],data[1]);self.assertIsNotNone(weak());data.clear();cache.clear();gc.collect();self.assertIsNone(weak())
    def test_token_drop_and_explicit_unsubscribe(self):
        mux=ContactMultiplexer(resolver);seen=[];a=mux.subscribe(lambda h,d:seen.append('a'));b=mux.subscribe(lambda h,d:seen.append('b'));del a;gc.collect();mux.dispatch([],[]);self.assertEqual(seen,['b']);b.unsubscribe();mux.dispatch([],[]);self.assertEqual(seen,['b']);self.assertEqual(mux._subscriptions,{})
    def test_exception_clears_context_without_global_changes(self):
        mux=ContactMultiplexer(resolver);original=(np.array,np.asarray);headers,data=fixture()
        def raises(h,d):mux.numpy_proxy.asarray(d[0].impulse);raise RuntimeError('test')
        token=mux.subscribe(raises)
        with self.assertRaisesRegex(RuntimeError,'test'):mux.dispatch(headers,data)
        self.assertIsNone(_active.get());self.assertEqual((np.array,np.asarray),original);self.assertEqual(mux.last_stats['asarray_misses'],1);token.unsubscribe()
    def test_new_event_does_not_reuse_stale_values(self):
        mux=ContactMultiplexer(resolver);values=[];token=mux.subscribe(lambda h,d:values.append(mux.numpy_proxy.asarray(d[0].impulse).tolist()))
        point=NativePoint([1.,2.,3.]);mux.dispatch([], [point]);point._impulse=[4.,5.,6.];mux.dispatch([], [point]);self.assertEqual(values,[[1.,2.,3.],[4.,5.,6.]])
    def test_nested_dispatch_restores_outer_cache(self):
        outer=ContactMultiplexer(resolver);inner=ContactMultiplexer(resolver);seen=[];a=inner.subscribe(lambda h,d:seen.append(_active.get()))
        def callback(h,d):before=_active.get();inner.dispatch([],[]);self.assertIs(_active.get(),before);seen.append(before)
        b=outer.subscribe(callback);outer.dispatch([],[]);self.assertIsNot(seen[0],seen[1]);self.assertIsNone(_active.get())

if __name__=='__main__':unittest.main()
