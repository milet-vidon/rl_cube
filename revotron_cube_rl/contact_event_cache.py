"""Unintegrated, pure-Python contact-cache prototype.

No global NumPy or PhysX monkeypatch is performed. Consumers must explicitly
receive `numpy_proxy` and `schema_proxy` in their PRIVATE callback namespace.
Runtime factory/zero-arg-super/native subscription integration is deliberately
not implemented pending same-event PhysX shadow equivalence tests.
"""
import contextvars,weakref
import numpy as np

_active=contextvars.ContextVar('contact_event_cache_frame',default=None)

class EventCache:
    def __init__(self):
        self.vectors={};self.arrays={};self.paths={}
        self.stats={'array_misses':0,'array_hits':0,'asarray_misses':0,'asarray_hits':0,'path_misses':0,'path_hits':0,'registered_vectors':0}
    def register(self,value):
        # Retain strong references until the dispatch ends: identity cannot be
        # recycled while used as a key. ndarray inputs already convert cheaply
        # and retain their normal aliasing semantics through the bypass path.
        if not isinstance(value,np.ndarray):
            if id(value) not in self.vectors:self.stats['registered_vectors']+=1
            self.vectors[id(value)]=value
        return value
    def convert(self,operation,args,kwargs):
        original=getattr(np,operation)
        if len(args)!=1 or kwargs or self.vectors.get(id(args[0])) is not args[0]:return original(*args,**kwargs)
        key=(operation,id(args[0]))
        if key not in self.arrays:
            # Keep np.array and np.asarray caches separate: vector protocols
            # may respond differently to copy/dtype requests.
            value=original(args[0])
            if type(value) is not np.ndarray or not (value.flags.c_contiguous or value.flags.f_contiguous):return value
            stored=value.copy(order='K');stored.flags.writeable=value.flags.writeable
            self.arrays[key]=stored;self.stats[operation+'_misses']+=1
        else:self.stats[operation+'_hits']+=1
        # Original callbacks are read-only. Independent arrays also prevent one
        # consumer from mutating a cached value subsequently seen by another.
        output=self.arrays[key].copy(order='K');output.flags.writeable=self.arrays[key].flags.writeable
        return output
    def path(self,resolver,args,kwargs):
        if len(args)!=1 or kwargs:return resolver(*args,**kwargs)
        key=args[0]
        if key not in self.paths:self.paths[key]=resolver(*args);self.stats['path_misses']+=1
        else:self.stats['path_hits']+=1
        return self.paths[key]
    def clear(self):self.vectors.clear();self.arrays.clear();self.paths.clear()

class NumpyProxy:
    """Explicit namespace dependency, never installed on the NumPy module."""
    def __getattr__(self,name):return getattr(np,name)
    def array(self,*args,**kwargs):
        cache=_active.get();return np.array(*args,**kwargs) if cache is None else cache.convert('array',args,kwargs)
    def asarray(self,*args,**kwargs):
        cache=_active.get();return np.asarray(*args,**kwargs) if cache is None else cache.convert('asarray',args,kwargs)

class SchemaProxy:
    def __init__(self,resolver):self.resolver=resolver
    def intToSdfPath(self,*args,**kwargs):
        cache=_active.get();return self.resolver(*args,**kwargs) if cache is None else cache.path(self.resolver,args,kwargs)

class CachedPoint:
    def __init__(self,raw,cache):self._raw=raw;self._cache=cache;self._fields={}
    def __getattr__(self,name):
        if name not in ('impulse','position','normal'):return getattr(self._raw,name)
        if name not in self._fields:self._fields[name]=self._cache.register(getattr(self._raw,name))
        return self._fields[name]

class CachedData:
    def __init__(self,raw,cache):self.raw=raw;self.cache=cache;self.points={}
    def __len__(self):return len(self.raw)
    def __getitem__(self,key):
        if isinstance(key,slice):return [self[i] for i in range(*key.indices(len(self)))]
        index=key if key>=0 else len(self)+key
        if not 0<=index<len(self):raise IndexError(index)
        if index not in self.points:self.points[index]=CachedPoint(self.raw[index],self.cache)
        return self.points[index]
    def __iter__(self):
        for i in range(len(self)):yield self[i]
    def clear(self):
        for point in self.points.values():point._fields.clear();point._raw=None
        self.points.clear();self.raw=None

class Subscription:
    def __init__(self,owner,key,callback):self._owner=weakref.ref(owner);self.key=key;self.callback=callback;self.active=True
    def unsubscribe(self):
        if not self.active:return
        self.active=False;self.callback=None;owner=self._owner()
        if owner is not None:owner._subscriptions.pop(self.key,None)
    def __del__(self):self.unsubscribe()

class ContactMultiplexer:
    """Explicit dispatch only; no native subscriptions/factories are touched."""
    def __init__(self,path_resolver):
        self.numpy_proxy=NumpyProxy();self.schema_proxy=SchemaProxy(path_resolver)
        self._subscriptions={};self._next=0;self.last_stats=None;self.closed=False
    def subscribe(self,callback):
        if self.closed:raise RuntimeError('Multiplexer is closed')
        key=self._next;self._next+=1;token=Subscription(self,key,callback)
        owner=weakref.ref(self)
        def expired(_):
            obj=owner()
            if obj is not None:obj._subscriptions.pop(key,None)
        self._subscriptions[key]=weakref.ref(token,expired);return token
    def dispatch(self,headers,data):
        if self.closed:raise RuntimeError('Multiplexer is closed')
        cache=EventCache();wrapped=CachedData(data,cache);context=_active.set(cache)
        try:
            # Registration order is explicit. Native order/exception behavior
            # still needs observation before this prototype can be integrated.
            for reference in list(self._subscriptions.values()):
                token=reference()
                if token is not None and token.active:token.callback(headers,wrapped)
        finally:
            self.last_stats=dict(cache.stats);_active.reset(context);wrapped.clear();cache.clear()
    def close(self):
        for reference in list(self._subscriptions.values()):
            token=reference()
            if token is not None:token.unsubscribe()
        self._subscriptions.clear();self.closed=True
