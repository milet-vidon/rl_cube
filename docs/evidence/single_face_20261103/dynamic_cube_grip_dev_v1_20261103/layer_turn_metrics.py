"""Cube-local face-turn measurements; whole-object motion cannot count as a turn."""
import numpy as np
from scipy.spatial.transform import Rotation


def face_axis(face):
    if face not in ('pX','nX','pY','nY','pZ','nZ'):raise ValueError(face)
    axis=np.zeros(3);axis['XYZ'.index(face[1])]=1 if face[0]=='p' else -1
    return axis


def rotation(body):
    return Rotation.from_quat(np.asarray(body['quaternion_wxyz'])[[1,2,3,0]])


def measure_layers(states,face='nZ',target_deg=90.):
    core=rotation(states['Core']);axis=face_axis(face);angles={};off_axis={};total={}
    for name,state in states.items():
        if name=='Core':continue
        rel=core.inv()*rotation(state);q=rel.as_quat()
        # q and -q represent the same physical rotation.
        if q[3]<0:q=-q
        angle=float(np.degrees(2*np.arctan2(np.dot(q[:3],axis),q[3])))
        twist=Rotation.from_rotvec(axis*np.radians(angle))
        angles[name]=angle;off_axis[name]=float(np.degrees((twist.inv()*rel).magnitude()));total[name]=float(np.degrees(rel.magnitude()))
    selected=[n for n in angles if face in n.split('_')]
    if len(selected)!=9:raise ValueError(f'Face {face} needs 9 cubelets, got {selected}')
    errors=[abs((angles[n]-target_deg+180)%360-180)for n in selected]
    rest=[total[n]for n in angles if n not in selected]
    maximum=max(errors);rest_max=max(rest);off_max=max(off_axis[n]for n in selected)
    separation=max(float(np.linalg.norm(np.asarray(v['position_m'])-states['Core']['position_m']))for k,v in states.items()if k!='Core')
    return {'face':face,'target_deg':target_deg,'selected_cubelets':selected,'signed_angles_deg':angles,
            'maximum_relative_rotation_deg':max(total.values()),'face_angle_deg':float(np.median([angles[n]for n in selected])),
            'maximum_face_error_deg':maximum,'maximum_face_off_axis_deg':off_max,'maximum_other_cubelet_rotation_deg':rest_max,
            'maximum_joint_origin_separation_m':separation,'geometry_qualified':maximum<=5 and off_max<=5 and rest_max<=5 and separation<=.002,
            'assembled_for_grasp':max(total.values())<=5 and separation<=.002}
