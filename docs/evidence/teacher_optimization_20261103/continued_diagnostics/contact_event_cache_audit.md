# Contact event cache prototype audit — 2026-11-03

Decision: **do not integrate this prototype**. It preserves the tested consumer outputs, but both synthetic vectors and real native `carb.Float3` replay are slower. No original runtime/audit source, NumPy module, native subscription factory, or controller was modified. No Isaac/Kit app was started by these tests.

## Interface and scope

Independent sources in `revotron_cube_rl/`:

- `contact_event_cache.py`: `ContactMultiplexer(path_resolver)`, `.subscribe(callback)` returning a weakly tracked subscription token, `.dispatch(headers, data)`, `.close()`. Consumers must explicitly receive `mux.numpy_proxy` and `mux.schema_proxy` in a private callback namespace. This is deliberately not a production subscription adapter.
- `test_contact_event_cache.py`: extracts and compiles original callback method ASTs unchanged in private namespaces, including the inherited `DetailedSelfContactAudit.callback -> SelfContactAudit.callback` path. Seven tests pass.
- `benchmark_contact_event_cache.py`: pure Python synthetic vector payload replay.
- `benchmark_contact_cache_float3.py`: genuine native vector payload replay, but synthetic header/contact wrappers. Runs without SimulationApp, AppLauncher, or `omni.kit.app` imports.

One per-event context caches raw vector identity, independent default NumPy conversions, and actor/collider path resolution. `np.array` and `np.asarray` have separate caches. Each return is an independent copy with the original dtype/layout/writeability; ndarray input and non-default args/kwargs bypass caching. Original `np.linalg.norm`, thresholds, point order, scalar accumulation order, and all contact points are retained. Nested dispatch restores its parent context; exception cleanup clears references. Dropping or unsubscribing tokens removes subscriptions.

Test coverage: original callback metrics and list/dict ordering checked with exact types and floating-point bytes; separate operation caches; dtype/Fortran layout; independent copies; kwargs bypass; ndarray alias semantics; vector identity/lifetime; data slice order; weak token lifetime; explicit unsubscribe; exception cleanup; no global NumPy mutation; no stale values across events; nested contexts.

## Native type verification

Installed Isaac Sim 6.2 package PhysX binding:

`extscache/omni.physx-110.4.4+110.4.0.lx64.r.cp312.u7f4/omni/physx/bindings/_physx.pyi`

`ContactData.impulse`, `.position`, and `.normal` all return `carb._carb.Float3`. Binding SHA-256: `d5a6c56a0bf929917d1876fadb4127d4232b58d8b65e7aaefc856a4f6b0ca1f0`.

Standalone import works by adding the installed `kit/kernel/py` to this process's `sys.path` and installed `kit` to its `LD_LIBRARY_PATH`. No simulator application is constructed. Native Float3 conversion yields an owned float64 ndarray of shape (3,), strides (8,), for both default `np.asarray` and `np.array`.

## Measurements

All replay cases: 20 events per trial, 5 headers per event, 1024 points per header; four top-level consumers, with inherited self-contact audit also executed. Native replay warms up each mode and alternates baseline/cache order over three trials. Exact consumer output agreement holds after every trial.

| Payload | Baseline | Cache | Cache/baseline time |
|---|---:|---:|---:|
| Python synthetic vector | 0.425 s | 0.758 s | 1.78x |
| Native Float3, stable property object (median) | 1.049 s | 1.271 s | 1.211x |
| Native Float3, fresh object per property access (median) | 1.134 s | 1.328 s | 1.171x |

Native microbenchmark, 200,000 calls per operation: `np.asarray(Float3)` 3.151 microseconds; `np.array(Float3)` 3.191 microseconds; independent ndarray copy 0.117 microseconds. This supports the profile's costly native conversion observation, but the current proxy, context, dictionary and per-call copy overhead more than consumes the saved conversions in these consumers.

Python fixture conversion count fell 202,240 -> 140,800 (30.4%). In each native event the default-array cache has 3,072 misses and zero hits, while default-asarray has 3,968 misses and 3,072 hits. Keeping these separate preserves operation semantics but limits reuse. Conversion count reduction alone is not a valid speedup claim.

Remote evidence under `work/teacher_optimization_20261103/`:

- `contact_event_cache_synthetic_v1.json`
- `contact_event_cache_float3_v1.json`

Evidence includes source/callback AST hashes and native binding hash. Original prototype sources remain unchanged after those measurements. All uploaded new sources were normalized to LF before execution/hash generation.

## Limits and next decision

Neither test uses a real `ContactDataVector`, native `ContactEventHeader`, actual PhysX path resolver, or actual callback subscription delivery. Native point-property object identity, lifetime, mutation constraints, callback exception propagation and subscription ordering still require shadow verification. Stable/fresh vector modes bound two hypotheses; they do not establish which native event policy is used. Private AST cloning is only an offline test harness; runtime binding and inherited method lookup are unresolved and must not be casually monkeypatched.

Do not deploy this implementation or spend a simulator episode solely to demonstrate its synthetic slowdown. A future candidate should first show an offline advantage while retaining the same tests; possible useful directions are a typed/native preprocessing cache with an explicit interface or simpler path-only caching. Any shared canonical impulse array spanning `np.array` and `np.asarray` needs a narrowly proven Float3 contract; it cannot be generalized to arbitrary vector protocols. Caching derived norm results or changing accumulation order is outside the current authorization and equivalence contract.

If a faster candidate is developed, the next single-Isaac shadow should deliver each actual event to original consumers and isolated shadow consumers, compare all per-step state/metrics/output records bit-for-bit, capture native type and identity/lifetime observations, and measure added overhead without using shadow outputs for control. Only after that should an opt-in control run assess end-to-end wall time and unchanged gates. Existing sequential planning parallelization already has exact validation evidence and is the supported optimization; no GPU/num-env switch follows from this cache experiment.
