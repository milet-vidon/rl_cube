from pathlib import Path
import xml.etree.ElementTree as ET
import json
ROOT=Path(__file__).resolve().parent
p=ROOT/'source/revotron_system/urdf/revotron_bimanual_revo3.urdf'
tree=ET.parse(p);r=tree.getroot()
fixed=['waist_pitch_joint','neck_yaw_joint','head_pitch_joint']
original=[j.get('name') for j in r.findall('joint') if j.get('type')!='fixed']
assert len(original)==59
for name in fixed:
    j=r.find(f"joint[@name='{name}']"); assert j is not None
    j.set('type','fixed')
    for tag in ['axis','limit','dynamics']:
        n=j.find(tag)
        if n is not None:j.remove(n)
target=p.with_name('revotron_bimanual_revo3_56dof.urdf')
tree.write(target,encoding='utf-8',xml_declaration=True)
names=[]
for side,prefix in [('L','left_'),('R','right_')]:
    names += [f'Joint{i}_{side}' for i in range(1,8)]
    names += [j.get('name') for j in r.findall('joint') if j.get('type')=='revolute' and j.get('name').startswith(prefix)]
assert len(set(names))==56
missing=[str((p.parent/m.get('filename')).resolve()) for m in r.findall('.//mesh') if not (p.parent/m.get('filename')).exists()]
assert not missing,missing
data={'original_active_joints':59,'variant_active_joints':56,'fixed_at_zero':fixed,'action_order':'Proposed explicit order. Resolve imported simulation joint indices by names.','joint_names':names,'arm_left':[0,7],'hand_left':[7,28],'arm_right':[28,35],'hand_right':[35,56],'chest_camera_added':False,'missing_meshes':missing}
(ROOT/'joint_mapping_56.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print('56 DOF variant and mesh audit ready')
