from pathlib import Path
import cv2,json,hashlib,shutil
repo=Path('/home/brainco/rl_cube/github_project');runs=repo/'revotron_cube_rl/runs';new=runs/'center_j56_dev_v3_20261103';old=runs/'centered_opposition_center_dev_v2_20261103';out=Path('/home/brainco/Documents/Codex/2026-11-03/j56_center/outputs');out.mkdir(parents=True,exist_ok=True)
source=cv2.VideoCapture(str(new/'demo.mp4'));previous=cv2.VideoCapture(str(old/'demo.mp4'));fps=source.get(cv2.CAP_PROP_FPS);assert fps==10 and previous.get(cv2.CAP_PROP_FPS)==10
start_s=53.5;start=int(start_s*fps);assert source.get(cv2.CAP_PROP_FRAME_COUNT)>start
old_count=int(previous.get(cv2.CAP_PROP_FRAME_COUNT));previous_last=None
source.set(cv2.CAP_PROP_POS_FRAMES,start);previous.set(cv2.CAP_PROP_POS_FRAMES,start);width=int(source.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(source.get(cv2.CAP_PROP_FRAME_HEIGHT));clip=out/'j56_center_demo.mp4';comparison=out/'centering_comparison.mp4';writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'avc1'),fps,(width,height));pair_writer=cv2.VideoWriter(str(comparison),cv2.VideoWriter_fourcc(*'avc1'),fps,(2*width,height));assert writer.isOpened()and pair_writer.isOpened();count=0
while True:
 ok,frame=source.read();old_ok,old_frame=previous.read()
 if not ok:break
 writer.write(frame)
 if old_ok:previous_last=old_frame.copy()
 if previous_last is not None:
  first=previous_last.copy();second=frame.copy();first[:28]=0;second[:28]=0
  cv2.putText(first,'Previous final pose (still)'if start+count>=old_count-15 else 'Previous centering',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(second,'J5 outward + J6 inward',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
  pair_writer.write(cv2.hconcat([first,second]))
 count+=1
writer.release();pair_writer.release();source.release();previous.release();assert count>0
for path in (clip,comparison):
 cap=cv2.VideoCapture(str(path));assert cap.read()[0],path;cap.release()
shutil.copy2(new/'final.png',out/'final.png')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'source_run':new.name,'split':'development','seed':9501,'source_video_sha256':sha(new/'demo.mp4'),'previous_run':old.name,'previous_video_sha256':sha(old/'demo.mp4'),'clip_start_s':start_s,'fps':fps,'frames':count,'duration_s':count/fps,'playback_speed':1.,'source_tail_still_frames':15,'comparison_old_end_extended_with_labeled_still_frame':True,'sha256':{p.name:sha(p)for p in (clip,comparison,out/'final.png')}};(out/'media_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
