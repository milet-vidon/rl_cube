"""Solve independent reset IK paths on CPU workers; online teacher feedback stays lane-local."""
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

def solve(job):
    from opposition_teacher import OppositionPlanner
    j,xyz,yaw=job
    p=OppositionPlanner(Path(__file__).resolve().parent/'opposition_grasp_v20.json')
    try:return j,xyz,yaw,p.plan(xyz,yaw),p.diagnostics,None
    except ValueError as exc:
        if not any(t in str(exc).lower() for t in ('envelope','ik','spline','time path','preshape','approach','lift','lower','infeasible')):raise
        return j,xyz,yaw,None,[],str(exc)

def reset_paths(core):
    from isaac_compat import quaternion_from_native
    from scipy.spatial.transform import Rotation
    jobs=[]
    for j,state in enumerate(core):
        q=quaternion_from_native(state[3:7]).numpy()
        yaw=Rotation.from_quat(q[[1,2,3,0]]).as_euler('xyz')[2]
        jobs.append((j,state[:3].numpy().copy(),yaw))
    if len(jobs)<32:
        yield from map(solve,jobs);return
    # Never fork the active Kit/CUDA process. The clean CPU forkserver shares loaded libraries.
    mp.set_forkserver_preload(['opposition_teacher'])
    with ProcessPoolExecutor(max_workers=min(8,len(os.sched_getaffinity(0))),mp_context=mp.get_context('forkserver')) as pool:
        yield from pool.map(solve,jobs,chunksize=4)

def bind_path(xyz,yaw,path,diagnostics):
    def prepared(planner,actual_xyz,actual_yaw):
        if not np.array_equal(actual_xyz,xyz) or abs(actual_yaw-yaw)>1e-12:raise RuntimeError('Wrong reset pose for prepared IK')
        planner.diagnostics[:]=diagnostics
        return path
    return prepared
