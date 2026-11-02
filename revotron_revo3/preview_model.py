"""Actual URDF visual geometry preview and GLB export; no generative image content."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
URDF=ROOT/'source/revotron_system/urdf/revotron_bimanual_revo3.urdf'
r=ET.parse(URDF).getroot()
def origin(node):
    T=np.eye(4)
    if node is not None:
        T[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T
links={n.get('name'):n for n in r.findall('link')}
joints=r.findall('joint'); children={j.find('child').get('link') for j in joints}
transforms={n:np.eye(4) for n in links if n not in children}
while len(transforms)<len(links):
    old=len(transforms)
    for j in joints:
        parent=j.find('parent').get('link'); child=j.find('child').get('link')
        if parent in transforms: transforms[child]=transforms[parent]@origin(j.find('origin'))
    assert len(transforms)>old,'Disconnected URDF'
materials={m.get('name'):np.fromstring(m.find('color').get('rgba'),sep=' ') for m in r.findall('material') if m.find('color') is not None}
scene=trimesh.Scene(); triangles=[]; colors=[]; counts={}; missing=[]
for name,link in links.items():
    for i,v in enumerate(link.findall('visual')):
        g=v.find('geometry'); me=g.find('mesh')
        if me is None: continue
        p=(URDF.parent/me.get('filename')).resolve()
        if not p.exists(): missing.append(str(p)); continue
        if p.suffix.lower()=='.dae':
            # Collada screen mesh: triangles and position source, no optional pycollada dependency.
            d=ET.parse(p); ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
            mm=d.find('.//c:mesh',ns); vert=mm.find('c:vertices/c:input',ns).get('source')[1:]
            verts=np.fromstring(mm.find(f"c:source[@id='{vert}']/c:float_array",ns).text,sep=' ').reshape(-1,3)
            face=[]
            for ts in mm.findall('c:triangles',ns):
                inputs=ts.findall('c:input',ns); stride=max(int(x.get('offset','0')) for x in inputs)+1
                offset=int(next(x for x in inputs if x.get('semantic')=='VERTEX').get('offset','0'))
                face.extend(np.fromstring(ts.find('c:p',ns).text,sep=' ',dtype=int).reshape(-1,stride)[:,offset].reshape(-1,3))
            m=trimesh.Trimesh(verts,np.array(face),process=False)
        else: m=trimesh.load(p,force='mesh',process=False)
        scale=np.fromstring(me.get('scale','1 1 1'),sep=' ')
        m.vertices*=scale
        m.apply_transform(transforms[name]@origin(v.find('origin')))
        material=v.find('material'); color=np.array([.65,.68,.72,1.])
        if material is not None:
            color=materials.get(material.get('name'),color)
            if material.find('color') is not None: color=np.fromstring(material.find('color').get('rgba'),sep=' ')
        m.visual.face_colors=np.tile((color*255).astype(np.uint8),(len(m.faces),1))
        scene.add_geometry(m,node_name=f'{name}_{i}',geom_name=f'{name}_{i}')
        triangles.append(m.triangles); colors.append(np.tile(color[:3],(len(m.faces),1)))
        counts[name]=counts.get(name,0)+len(m.faces)
assert not missing,missing
scene.export(ROOT/'revotron_bimanual_revo3.glb')
tri=np.concatenate(triangles); col=np.concatenate(colors)
np.savez_compressed(ROOT/'preview_geometry.npz',triangles=tri,colors=col)
active=[j.get('name') for j in joints if j.get('type')!='fixed']
audit={'links':len(links),'active_joints':len(active),'joint_names':active,'visual_triangles':len(tri),'bounds':scene.bounds.tolist(),'missing_meshes':missing,'pose':'All joints at URDF zero; visualization only, not a grasp pose'}
(ROOT/'model_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
