"""CPU raster preview from actual URDF mesh triangles."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
d=np.load(ROOT/'preview_geometry.npz'); tri=d['triangles']; colors=d['colors']
normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
bounds=np.array([tri.min(axis=(0,1)),tri.max(axis=(0,1))]); center=bounds.mean(axis=0)
def render(view,size=(900,1200)):
    eye=np.array(view,dtype=float); eye/=np.linalg.norm(eye)
    right=np.cross([0,0,1],eye); right/=np.linalg.norm(right); up=np.cross(eye,right)
    visible=normal@eye>0.0001
    t=tri[visible]-center; c=colors[visible]; n=normal[visible]
    x=t@right; y=t@up; z=t@eye
    scale=min((size[0]-90)/np.ptp(x),(size[1]-130)/np.ptp(y))
    xx=(x-(x.max()+x.min())/2)*scale+size[0]/2
    yy=-(y-(y.max()+y.min())/2)*scale+size[1]/2+20
    points=np.stack([xx,yy],axis=-1).astype(np.int32)
    light=np.array([1,-.7,1.8]); light/=np.linalg.norm(light)
    shades=.35+.65*np.maximum(n@light,0)
    rgb=np.clip(c*shades[:,None]*255,0,255).astype(np.uint8)
    img=np.full((size[1],size[0],3),242,dtype=np.uint8)
    for i in np.argsort(z.mean(axis=1)):
        cv2.fillConvexPoly(img,points[i],tuple(map(int,rgb[i])),lineType=cv2.LINE_AA)
    return Image.fromarray(img)
front=render([1,0,.14]); angled=render([1,-1,.35])
out=Image.new('RGB',(1800,1320),'#f2f2f2');out.paste(front,(0,90));out.paste(angled,(900,90))
draw=ImageDraw.Draw(out); font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34); small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
draw.text((40,20),'REVOTRON + REVO3 | Official bimanual model',fill='#182839',font=font)
draw.text((40,67),'Front view',fill='#526173',font=small);draw.text((940,67),'Three-quarter view',fill='#526173',font=small)
draw.text((40,1280),'Actual URDF visual meshes, zero-joint pose. 14 arm + 42 hand joints; 3 additional torso/head joints.',fill='#526173',font=small)
out.save(ROOT/'revotron_preview.png')
print('Rendered',len(tri),'triangles',flush=True)
