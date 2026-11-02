"""Schematic replay of measured PhysX poses; not a camera render."""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
a=json.loads((ROOT/'validated_cpu_result.json').read_text())
b=json.loads((ROOT/'no_contact_result.json').read_text())
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
colors={'pX':(180,35,38),'nX':(255,120,35),'pY':(35,90,200),'nY':(40,175,90),'pZ':(235,235,235),'nZ':(245,210,50)}
eye=np.array([1.3,-1.7,1.4]); eye/=np.linalg.norm(eye)
right=np.cross([0,0,1],eye); right/=np.linalg.norm(right)
up=np.cross(eye,right)
def rot(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def draw_cube(d,frame,cx):
    polys=[]
    for name,p in frame['positions'].items():
        p=np.array(p); R=rot(frame['quaternions'][name])
        for k in range(3):
            axes=[i for i in range(3) if i!=k]
            for sign in [-1,1]:
                normal=np.zeros(3); normal[k]=sign
                if np.dot(R@normal,eye)<0: continue
                pts=[]
                for u,v in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                    t=normal*0.00935; t[axes]=[u*0.00935,v*0.00935]
                    pts.append(p+R@t)
                pts=np.array(pts)
                key=('p' if sign==1 else 'n')+'XYZ'[k]
                c=colors.get(key,(30,33,40)) if key in name.split('_') else (30,33,40)
                xy=[(cx+np.dot(t,right)*3500,265-np.dot(t,up)*3500) for t in pts]
                polys.append((np.mean(pts@eye),xy,c))
    for _,pts,c in sorted(polys,key=lambda x:x[0]): d.polygon(pts,fill=c,outline=(10,15,22),width=2)
frames=[]
for fa,fb in zip(a['frames'],b['frames']):
    im=Image.new('RGB',(1000,520),(19,26,38)); d=ImageDraw.Draw(im)
    d.text((24,18),'Isaac Sim 5.1 | measured physics trajectory replay',font=font,fill='white')
    d.text((25,64),'CONTACT ENABLED',font=font,fill=(80,220,180))
    d.text((525,64),'SAME MOTION, CONTACT DISABLED',font=font,fill=(255,180,90))
    d.line((500,60,500,465),fill=(70,80,95))
    draw_cube(d,fa,250); draw_cube(d,fb,750)
    d.text((25,440),f"Turn +{fa['phase']} | simulation t = {fa['time']:.1f} s",font=font,fill='white')
    d.text((25,479),'Fixed core; external fixtures omitted. Schematic cubies drawn from logged positions and rotations.',font=small,fill=(180,190,205))
    frames.append(im)
frames[-1].save(ROOT/'final_state.png')
frames[0].save(ROOT/'contact_vs_no_contact.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=100,loop=0)
print('Preview saved')
