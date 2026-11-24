"""Synthetic same-payload benchmark; not a PhysX or whole-training speed claim."""
from pathlib import Path
import argparse,json,time,hashlib,unittest
from types import SimpleNamespace
import numpy as np
from contact_event_cache import ContactMultiplexer
from test_contact_event_cache import callbacks,fixture,resolver,NativeVector,assert_exact

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--events',type=int,default=20);p.add_argument('--points-per-header',type=int,default=1024);a=p.parse_args()
    if a.output.exists():raise ValueError('Unique benchmark output required')
    headers,data=fixture(a.points_per_header);base,base_snapshot,base_reset,hashes=callbacks(np,SimpleNamespace(intToSdfPath=resolver));mux=ContactMultiplexer(resolver);cached,snapshot,reset,cached_hashes=callbacks(mux.numpy_proxy,mux.schema_proxy);tokens=[mux.subscribe(c) for c in cached];assert hashes==cached_hashes
    NativeVector.conversions=0;start=time.perf_counter()
    for _ in range(a.events):
        base_reset()
        for callback in base:callback(headers,data)
    baseline_s=time.perf_counter()-start;baseline_conversions=NativeVector.conversions
    NativeVector.conversions=0;start=time.perf_counter();stats={}
    for _ in range(a.events):
        reset();mux.dispatch(headers,data)
        for k,v in mux.last_stats.items():stats[k]=stats.get(k,0)+v
    cached_s=time.perf_counter()-start;cached_conversions=NativeVector.conversions
    assert_exact(unittest.TestCase(),base_snapshot(),snapshot())
    root=Path(__file__).resolve().parent
    result={'scope':'synthetic long-event replay of unchanged callback ASTs in private namespaces; no native subscription or global NumPy patch','events':a.events,'headers_per_event':len(headers),'points_per_header':a.points_per_header,'callback_count':len(base),'baseline_s':baseline_s,'cache_s':cached_s,'synthetic_speedup':baseline_s/cached_s,'baseline_native_vector_conversions':baseline_conversions,'cache_native_vector_conversions':cached_conversions,'cache_stats':stats,'all_output_types_order_and_float_bits_exact':True,'norm_function_unchanged':mux.numpy_proxy.linalg.norm is np.linalg.norm,'callback_AST_sha256':hashes,'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('contact_event_cache.py','test_contact_event_cache.py','benchmark_contact_event_cache.py')},'limits':['Native PhysX event payload layout/lifetime/order/exception behavior untested','No runtime factory or original frozen callback module modified','Synthetic vector property/NumPy conversion costs can differ from native Float3 costs','Single synthetic benchmark does not predict overall simulation speedup']}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('callback_AST_sha256','source_sha256')},indent=2),flush=True)

if __name__=='__main__':main()
