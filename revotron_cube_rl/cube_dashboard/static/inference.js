/* No external rendering or network libraries. All geometry is derived from the selected recording. */
const NS = 'http://www.w3.org/2000/svg';
const phases = {ready:'准备',preshape:'张手预抓',lower:'下移',approach:'接近',squeeze:'夹紧',primary_lift:'抬升',elbow_outward:'手肘外展',tilt_90:'整颗倾转',wrist_reconfigure:'调整手腕',tilt_hold:'倾转后稳持',left_center_transport:'移向中心',center_hold:'中心稳持'};
const phaseName = s => phases[s] || s || '—';
const finite = n => typeof n === 'number' && Number.isFinite(n);
const vector = (v,n) => Array.isArray(v) && v.length===n && v.every(finite);
const fixed = (n,k=1) => finite(n) ? (Number(n.toFixed(k)) || 0).toFixed(k) : '—';
const truth = (b,yes,no) => b===true ? yes : b===false ? no : '未记录';
const icon = {
 play:'<path d="m8 5 10 7-10 7z" fill="currentColor" stroke="none"/>',
 pause:'<path d="M8 6v12M16 6v12" stroke-width="4"/>',
 prev:'<path d="M6 5v14m12-14-9 7 9 7z"/>', next:'<path d="M18 5v14M6 5l9 7-9 7z"/>',
 cube:'<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.6 8-4.6M12 12v9"/>',
 fit:'<path d="M8 4H4v4m12-4h4v4M4 16v4h4m12-4v4h-4"/>',
 arrow:'<path d="M4 12h15m-6-6 6 6-6 6"/>'
};
const svgIcon=(k)=>'<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'+icon[k]+'</svg>';
const template = `
<div class="ix">
  <div class="ix-toolbar"><div class="ix-field"><label data-l="episode">回放轨迹</label><select data-k="episode" disabled></select></div><div class="ix-field"><label data-l="model">推理模型</label><select data-k="model" disabled></select></div><div class="ix-context"><i></i> 本地工作台</div><button type="button" class="ix-refresh" data-k="refresh">刷新列表</button></div>
  <div data-k="notice" class="ix-notice" role="status">读取轨迹目录…</div>
  <div data-k="body" hidden>
    <div class="ix-overview">
      <section class="ix-card ix-spatial">
        <div class="ix-heading"><div><div class="ix-eyebrow">PHYSICS REPLAY</div><h2>轨迹回放</h2></div><span class="ix-live"><i></i> 已记录</span></div>
        <div class="ix-stage"><canvas data-k="scene" tabindex="0" aria-label="物理轨迹的几何示意，立方体代表魔方；拖动可旋转视角"></canvas>
           <div class="ix-sc-top"><span data-k="scene-info"></span><div role="group" aria-label="视图"><button type="button" data-view="iso" aria-pressed="true">等距</button><button type="button" data-view="side" aria-pressed="false">侧面</button><button type="button" data-view="top" aria-pressed="false">俯视</button></div></div>
           <div class="ix-sc-bottom"><span class="ix-legend"><i class="ix-left"></i>左臂<i class="ix-right"></i>右臂<i class="ix-trail"></i>魔方路径<i class="ix-touch"></i>接触点</span><span class="ix-optics"><button data-k="zoomout" type="button" aria-label="缩小视图" title="缩小">−</button><button data-k="zoomin" type="button" aria-label="放大视图" title="放大">+</button><button data-k="reset-view" type="button" title="重置视角" aria-label="重置视角">${svgIcon('fit')}</button></span></div>
        </div>
        <div class="ix-origin"><span data-k="origin"></span><span>三维示意 · 格线仅示整体朝向</span></div>
        <div class="ix-playbar">
           <div><button data-k="prev" type="button" aria-label="上一帧">${svgIcon('prev')}</button><button class="ix-play" data-k="play" type="button" aria-label="播放">${svgIcon('play')}</button><button data-k="next" type="button" aria-label="下一帧">${svgIcon('next')}</button></div>
           <span data-k="time"></span><label class="ix-rate">速度 <select data-k="rate" aria-label="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
        </div>
        <div class="ix-chart-wrap"><div class="ix-chart-label"><span>离桌高度 <small>mm</small></span><span data-k="chart-limit"></span></div><div class="ix-chart"><svg data-k="height" viewBox="0 0 600 45" preserveAspectRatio="none" aria-hidden="true"></svg><div data-k="cursor" class="ix-cursor"></div><input data-k="seek" type="range" min="0" max="0" value="0" aria-label="拖动轨迹时间轴"></div><div class="ix-chart-ticks"><span>0 s</span><span data-k="duration"></span></div></div>
      </section>
      <aside class="ix-card ix-readout"><div class="ix-heading"><div><div class="ix-eyebrow">FRAME INSPECTOR</div><h2>当前帧观测</h2></div></div>
        <div class="ix-frame-meta"><span data-k="phase" class="ix-phase"></span><span data-k="frame"></span></div>
        <div class="ix-height-number"><small>魔方离桌高度</small><p><b data-k="mm"></b> <span>mm</span></p></div>
        <dl class="ix-values"><dt>左手接触冲量</dt><dd data-k="li"></dd><dt>右手接触冲量</dt><dd data-k="ri"></dd><dt>桌面接触</dt><dd data-k="table"></dd><dt>记录的层转角</dt><dd data-k="face"></dd></dl>
        <div class="ix-report"><h3>整条轨迹结果</h3><dl class="ix-values"><dt>最高离桌</dt><dd data-k="peak"></dd><dt>最长连续拿稳</dt><dd data-k="hold"></dd><dt>整颗转动 / 单层拧动</dt><dd data-k="turns"></dd><dt>结束时接触桌面</dt><dd data-k="endtable"></dd></dl><p data-k="end"></p></div>
      </aside>
    </div>
    <section class="ix-card ix-policy">
      <div class="ix-heading"><div><div class="ix-eyebrow">POLICY INFERENCE</div><h2>单帧策略推理</h2></div><button class="ix-submit" data-k="predict" type="button">${svgIcon('play')} 运行当前帧</button></div>
      <p class="ix-desc">用所选模型读取该帧决策前的观测，在 CPU 上计算 28 个左侧关节的目标角度。计算结果不改变上方的历史轨迹。</p>
      <div data-k="pstate" class="ix-pstate" role="status">选择一帧，点击「运行当前帧」查看模型输出。</div>
      <div data-k="presult" hidden><div class="ix-psummary"><div><span>推理输入</span><b data-k="pid"></b></div><div><span>与归档策略的平均绝对差</span><b data-k="mae"></b></div><div><span>最大绝对差</span><b data-k="merr"></b></div></div>
      <div class="ix-jhead"><h3>关节目标 <small>rad</small></h3><label>关节组 <select data-k="jfilter" aria-label="筛选关节"><option value="arm">手臂</option><option value="hand">手指</option><option value="all">全部</option></select></label></div>
      <div class="ix-jtable"><table><thead><tr><th>关节</th><th>角度对照</th><th>本次输出</th><th>归档策略</th><th>实际下发</th></tr></thead><tbody data-k="jrows"></tbody></table></div>
      <p class="ix-jlegend"><i></i>本次输出 <b></b>归档策略 <em></em>实际下发 <span data-k="axisnote"></span></p>
      <p class="ix-foot">归档策略是原模型当时的输出；实际下发包含当时的控制修正。两者都是对照值，动作差异不能证明新模型的抓取成功率。</p></div>
    </section>
  </div>
</div>`;

function svgPath(svg,d,attrs={}){const p=document.createElementNS(NS,'path');p.setAttribute('d',d);for(const [k,v] of Object.entries(attrs))p.setAttribute(k,v);svg.append(p)}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
function node(tag,txt){const e=document.createElement(tag);e.textContent=txt;return e;}
async function api(path,body,signal) {
  const r=await fetch('/api/inference/'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal}:{signal,cache:'no-store'});
  let j;try{j=await r.json()}catch{throw new Error('服务返回了无效数据（HTTP '+r.status+'）')}
  if(!r.ok||j.error)throw new Error(j.error?.message || j.error || '请求失败（HTTP '+r.status+'）');
  return j;
}
const subsets={arm:n=>/^joint\d/i.test(n),hand:n=>!/^joint\d/i.test(n),all:()=>true};
const minus=(a,b)=>a.map((v,i)=>v-b[i]);
const plus=(a,b)=>a.map((v,i)=>v+b[i]);
const times=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const quatRotate=([w,x,y,z],[a,b,c])=>[(1-2*y*y-2*z*z)*a+2*(x*y-z*w)*b+2*(x*z+y*w)*c,2*(x*y+z*w)*a+(1-2*x*x-2*z*z)*b+2*(y*z-x*w)*c,2*(x*z-y*w)*a+2*(y*z+x*w)*b+(1-2*x*x-2*y*y)*c];
const views={iso:[.87,.56],side:[Math.PI/2,.06],top:[Math.PI/2,Math.PI/2-.015]};
function bounds(frames,table) {
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity]; let count=0;
 for(const f of frames)for(const k of ['cube','left_palm','right_palm','left_elbow','right_elbow','left_wrist','left_shoulder']){
  const v=f[k];if(!vector(v,3))continue;count++;
  for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],v[i]);hi[i]=Math.max(hi[i],v[i]);}
 }
 if(!count)return {center:null,diam:.4};
 if(finite(table))lo[2]=Math.min(lo[2],table);const center=times(plus(lo,hi),.5);
 const diam=Math.max(.30,Math.hypot(...minus(hi,lo))+.095);
 return {center,diam};
}
function renderScene(canvas,scene,frames,idx,camera) {
 const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(!w||!h)return;
 const dpr=Math.min(globalThis.devicePixelRatio||1,2);
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}
 const c=canvas.getContext('2d');if(!c)return;c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
 const f=frames[idx],center=scene.center;if(!center||!f){c.fillStyle='#a1bdc2';c.textAlign='center';c.font='13px sans-serif';c.fillText('本帧没有可绘制的位置数据',w/2,h/2);return}
 const [az,el]=camera;const co=Math.cos(az),si=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
 const hor=[-si,co,0],up=[-co*se,-si*se,ce],dep=[co*ce,si*ce,se],scale=Math.min(w*.93,h*.88)/scene.diam*(scene.zoom||1);
 const xy=p=>{const d=minus(p,center);return [w*.5+dot(d,hor)*scale,h*.50-dot(d,up)*scale]};
 const line=(a,b,color,width=1)=>{c.beginPath();c.moveTo(...xy(a));c.lineTo(...xy(b));c.strokeStyle=color;c.lineWidth=width;c.stroke()};
 const z=scene.table;
 if(finite(z)){const x=Math.round(center[0]*10)/10,y=Math.round(center[1]*10)/10;for(let i=-12;i<=12;i++){
  const color=i%2?'rgba(104,148,158,.13)':'rgba(104,148,158,.23)';line([x+i*.05,y-.6,z],[x+i*.05,y+.6,z],color);line([x-.6,y+i*.05,z],[x+.6,y+i*.05,z],color);
 }}
 c.beginPath();let begun=false;const stride=Math.max(1,Math.floor(frames.length/800));for(let i=0;i<=idx;i+=stride){if(!vector(frames[i].cube,3)){begun=false;continue}const p=xy(frames[i].cube);if(!begun)c.moveTo(...p);else c.lineTo(...p);begun=true}
 if(begun&&vector(f.cube,3))c.lineTo(...xy(f.cube));c.strokeStyle='#9ddcea';c.lineWidth=1.5;c.stroke();
 const dotAt=(p,r,fill)=>{const [x,y]=xy(p);c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fillStyle=fill;c.fill()};
 const label=(p,t,color,dx=8,dy=-6)=>{const [x,y]=xy(p);c.font='500 11px system-ui,sans-serif';c.textAlign='left';c.fillStyle=color;c.fillText(t,x+dx,y+dy)};
 // Only join adjacent observed landmarks. No unobserved fingers or mesh are inferred.
 for(const [side,color] of [['left','#73e6c3'],['right','#a3b5fc']]){
   const pts=[f[side+'_shoulder'],f[side+'_elbow'],f[side+'_wrist'],f[side+'_palm']];
   for(let i=1;i<pts.length;i++)if(vector(pts[i-1],3)&&vector(pts[i],3))line(pts[i-1],pts[i],color,2);
   for(const p of pts)if(vector(p,3))dotAt(p,3,color);
 }
 if(vector(f.cube,3)){
  if(vector(f.quat,4)&&scene.edge>0){
   const qnorm=Math.hypot(...f.quat),q=qnorm?f.quat.map(v=>v/qnorm):null;
   if(q){
    const faces=[['#829dcb',[1,0,0],[0,1,0],[0,0,1]],['#7a99a7',[-1,0,0],[0,1,0],[0,0,1]],['#da916c',[0,1,0],[1,0,0],[0,0,1]],['#df9a74',[0,-1,0],[1,0,0],[0,0,1]],['#d9e6c6',[0,0,1],[1,0,0],[0,1,0]],['#95b2a0',[0,0,-1],[1,0,0],[0,1,0]]];
    for(const [color,normal,u,v] of faces){
     if(dot(quatRotate(q,normal),dep)<=0)continue;
     const p=(a,b)=>plus(f.cube,quatRotate(q,times(plus(normal,plus(times(u,a),times(v,b))),scene.edge/2)));
     c.beginPath();[[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([a,b],j)=>j?c.lineTo(...xy(p(a,b))):c.moveTo(...xy(p(a,b))));c.closePath();c.fillStyle=color;c.fill();c.strokeStyle='#23363b';c.lineWidth=1.5;c.stroke();
     for(const t of [-1/3,1/3]){line(p(-1,t),p(1,t),'rgba(24,45,49,.72)',1.2);line(p(t,-1),p(t,1),'rgba(24,45,49,.72)',1.2)}
    }
   }else dotAt(f.cube,4,'#dfe8d5');
  }else dotAt(f.cube,4,'#dfe8d5');
 }
 for(const hit of Array.isArray(f.finger_contacts)?f.finger_contacts:[]){if(!vector(hit.xyz,3))continue;dotAt(hit.xyz,5,'rgba(255,213,101,.25)');dotAt(hit.xyz,2.6,'#ffda76')}
 for(const [s,cn,col] of [['left','L · 左掌','#a0f6de'],['right','R · 右掌','#c1cdff']])if(vector(f[s+'_palm'],3))label(f[s+'_palm'],cn,col,8,s==='left'?-6:13);
}

let mountID=0;
export function mountInference(container){
 if(!container)throw new Error('Missing inference mount point');
 container.innerHTML=template;const root=container.firstElementChild;const refs={};for(const n of root.querySelectorAll('[data-k]'))refs[n.dataset.k]=n;
 const rid='ix'+(++mountID);for(const l of root.querySelectorAll('[data-l]')){refs[l.dataset.l].id=rid+'-'+l.dataset.l;l.htmlFor=refs[l.dataset.l].id}
 const k=n=>refs[n];let cat=null,record=null,frames=[],idx=0,scene={},camera=[...views.iso],active=true,playing=false,req=0,frameStart=0,clockStart=0,playSpeed=1,dataVersion=0,policyVersion=0,pred=null,drag=null;
 const listeners=new AbortController();let trajectoryReq=null,policyReq=null,catalogReq=null;const on=(n,evt,fn)=>n.addEventListener(evt,fn,{signal:listeners.signal});
 const sceneDraw=()=>{if(active&&frames.length)renderScene(k('scene'),scene,frames,idx,camera)};
 const setNotice=(msg,err=false)=>{k('notice').hidden=!msg;k('notice').textContent=msg;k('notice').classList.toggle('ix-error',err)};
 function chosenModel(){return cat?.models?.find(m=>m.id===k('model').value)}
 function selectView(key,custom=false){camera=custom?camera:[...views[key]];for(const b of root.querySelectorAll('[data-view]'))b.setAttribute('aria-pressed',String(!custom&&b.dataset.view===key));sceneDraw()}
 function clearPrediction(){
  policyReq?.abort();policyVersion++;pred=null;k('presult').hidden=true;k('predict').disabled=!chosenModel()||!frames.length;k('pstate').classList.remove('ix-error');
  k('pstate').textContent=k('model').value?'选择一帧，点击「运行当前帧」查看模型输出。':'没有可加载的检查点。';
 }
 function drawHeight(){
  const el=k('height');el.replaceChildren();const ys=frames.map(f=>finite(f.height_m)?f.height_m*1000:null);const values=ys.filter(finite);
  if(!values.length){k('chart-limit').textContent='未记录';return}let min=Math.min(0,...values),max=Math.max(1,...values);
  k('chart-limit').textContent='峰值 '+fixed(Math.max(...values))+' mm';
  const y=n=>42-(n-min)/(max-min)*37;svgPath(el,'M0 '+y(0)+'H600',{stroke:'#dbe6e7','stroke-width':1});
  let d='',lastx=0;const first=frames[0].step,last=frames.at(-1).step;ys.forEach((n,i)=>{if(n===null)return;const x=600*(frames[i].step-first)/Math.max(1,last-first);if(d==='')d='M'+x.toFixed(2)+' '+y(n).toFixed(2);else d+='L'+x.toFixed(2)+' '+y(n).toFixed(2);lastx=x});
  if(!d)return;svgPath(el,d+'L'+lastx+' 45L0 45Z',{fill:'#dcf1e9'});svgPath(el,d,{stroke:'#208573','stroke-width':1.5,fill:'none','vector-effect':'non-scaling-stroke'});
 }
 function drawRows(){
  const body=k('jrows');body.replaceChildren();if(!pred)return;
  const max=Math.max(.1,...[...pred.predicted,...pred.reference,...pred.applied].filter(finite).map(Math.abs));
  const limit=Math.ceil(max*10)/10,pos=v=>(Math.max(-limit,Math.min(limit,v))/limit*.5+.5)*100;
  k('axisnote').textContent='共用刻度：−'+fixed(limit)+' 至 +'+fixed(limit)+' rad';
  pred.names.forEach((name,i)=>{if(!subsets[k('jfilter').value](name))return;
   const tr=document.createElement('tr'),td=document.createElement('td'),scale=document.createElement('div');scale.className='ix-jscale';scale.setAttribute('aria-hidden','true');
   const v=pred.predicted[i];if(finite(v)){const bar=document.createElement('span');bar.style.left=Math.min(50,pos(v))+'%';bar.style.width=Math.abs(pos(v)-50)+'%';scale.append(bar)}
   for(const [tag,n] of [['b',pred.reference[i]],['em',pred.applied[i]]])if(finite(n)){const mark=document.createElement(tag);mark.style.left=pos(n)+'%';scale.append(mark)}
   td.append(scale);tr.append(node('td',name),td);for(const a of [pred.predicted,pred.reference,pred.applied])tr.append(node('td',fixed(a[i],4)));body.append(tr)
  });
 }
 const elapsed=(f)=>frames.length?(f.step-frames[0].step)/(record.episode.hz||30):0;
 function renderFrame(i){
  if(!frames.length)return;const next=Math.max(0,Math.min(frames.length-1,i));if(next!==idx&&pred)k('pstate').textContent='下方保留上次推理结果；点击「运行当前帧」可更新。';
  idx=next;const f=frames[idx],t=elapsed(f),end=elapsed(frames.at(-1));k('seek').value=String(idx);k('seek').setAttribute('aria-valuetext','第 '+f.step+' 帧，'+fixed(t,2)+' 秒');k('cursor').style.left=(end>0?t/end*100:0)+'%';
  k('frame').textContent='STEP '+f.step;k('phase').textContent=phaseName(f.phase);k('mm').textContent=fixed(finite(f.height_m)?f.height_m*1000:null);k('li').textContent=finite(f.left_impulse)?fixed(f.left_impulse,4)+' N·s':'—';k('ri').textContent=finite(f.right_impulse)?fixed(f.right_impulse,4)+' N·s':'—';
  k('table').textContent=truth(f.table_contact,'有接触','无接触');k('face').textContent=finite(f.face_angle_deg)?fixed(f.face_angle_deg)+'°':'—';k('time').textContent=fixed(t,2)+' / '+fixed(end,2)+' s';
  k('prev').disabled=idx===0;k('next').disabled=idx===frames.length-1;sceneDraw();
 }
 function pause(){playing=false;cancelAnimationFrame(req);k('play').innerHTML=svgIcon('play');k('play').setAttribute('aria-label','播放')}
 function tick(now){
  if(!playing||!active||!frames.length)return;const target=frameStart+(now-clockStart)/1000*(record.episode.hz||30)*playSpeed;
  let lo=idx,hi=frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].step<=target)lo=mid;else hi=mid-1}if(lo!==idx)renderFrame(lo);if(idx===frames.length-1){pause();return}req=requestAnimationFrame(tick)
 }
 function play(){if(playing){pause();return}if(!frames.length||!active)return;if(idx===frames.length-1)renderFrame(0);frameStart=frames[idx].step;clockStart=performance.now();playing=true;k('play').innerHTML=svgIcon('pause');k('play').setAttribute('aria-label','暂停');req=requestAnimationFrame(tick)}
 function showEpisode(data){
  if(!Array.isArray(data?.frames)||!data.frames.length)throw new Error('该轨迹没有可读帧。');
  record=data;frames=data.frames;const ep=data.episode,summary=data.summary||{};scene={...bounds(frames,data.table_z_m),zoom:1,table:data.table_z_m,edge:finite(ep.edge_mm)?ep.edge_mm/1000:null};
  setNotice('');k('body').hidden=false;k('scene-info').textContent='SEED '+ep.seed+(finite(ep.edge_mm)?'  /  '+ep.edge_mm+' mm':'')+'  ·  位姿示意';k('origin').textContent='轨迹来源：'+(cat.models.find(m=>m.id===ep.origin_model_id)?.label||ep.origin_model_id||'归档记录');k('seek').max=String(frames.length-1);k('peak').textContent=finite(summary.peak_height_m)?fixed(Math.max(0,summary.peak_height_m*1000))+' mm':'—';k('hold').textContent=finite(summary.best_hold_s)?fixed(summary.best_hold_s,2)+' s':'—';
  k('turns').textContent=[summary.whole_turn,summary.face_turn].map(v=>truth(v,'达到','未达到')).join(' / ');k('endtable').textContent=truth(summary.final_table_contact,'是','否');k('end').textContent='终止记录：'+(summary.end_reason||'未记录');k('duration').textContent=fixed(elapsed(frames.at(-1)),2)+' s';
  idx=0;selectView('iso');drawHeight();renderFrame(0);clearPrediction();
 }
 async function fetchEpisode(){
  pause();const seq=++dataVersion;trajectoryReq?.abort();trajectoryReq=new AbortController();frames=[];record=null;clearPrediction();k('body').hidden=true;setNotice('读取已有物理轨迹…');
  try{const data=await api('trajectory?episode='+encodeURIComponent(k('episode').value),null,trajectoryReq.signal);if(seq===dataVersion)showEpisode(data)}catch(e){if(seq===dataVersion&&e.name!=='AbortError')setNotice('轨迹读取失败：'+e.message,true)}
 }
 async function loadCatalog(){
  catalogReq?.abort();catalogReq=new AbortController();const signal=catalogReq.signal;
  const ep=k('episode'),mod=k('model'),priorEpisode=ep.value,priorModel=mod.value;k('refresh').disabled=true;
  try{
   cat=await api('catalog',null,signal);
   ep.replaceChildren(...cat.episodes.map(e=>option(e.id,e.label||('种子 '+e.seed))));mod.replaceChildren(...cat.models.map(m=>option(m.id,m.label||m.id)));
   ep.disabled=!ep.options.length;mod.disabled=!mod.options.length;
   const pick=(select,old,defaultID,list)=>{const id=[old,defaultID].find(id=>id&&list.some(x=>x.id===id));if(id)select.value=id};
   pick(ep,priorEpisode,cat.default_episode,cat.episodes);pick(mod,priorModel,cat.default_model,cat.models);
   if(!ep.options.length){pause();frames=[];record=null;k('body').hidden=true;setNotice('暂无可回放的实测轨迹。');return}
   if(priorEpisode===ep.value&&record?.episode?.id===priorEpisode&&frames.length){setNotice('');if(priorModel!==mod.value)clearPrediction()}else await fetchEpisode();
  }
  catch(e){if(e.name==='AbortError')return;setNotice('无法读取推理目录：'+e.message,true)}
  finally{if(!signal.aborted)k('refresh').disabled=false}
 }
 async function predict(){
  if(!frames.length||!chosenModel())return;pause();const step=frames[idx].step,model=chosenModel(),ep=record.episode;policyReq?.abort();policyReq=new AbortController();const seq=++policyVersion;
  k('predict').disabled=true;k('pstate').textContent='正在 CPU 上计算 STEP '+step+'…';k('pstate').classList.remove('ix-error');
  try{const d=await api('predict',{model:model.id,episode:ep.id,step},policyReq.signal);if(seq!==policyVersion)return;pred=d;k('presult').hidden=false;k('pid').textContent='STEP '+step+' · '+(model.label||model.id);k('mae').textContent=fixed(d.mae_rad,5)+' rad';k('merr').textContent=fixed(d.max_abs_error_rad,5)+' rad';k('pstate').textContent='CPU 推理完成 · 种子 '+ep.seed+' · '+phaseName(d.phase)+(frames[idx].step===step?'':' · 已切换到其他帧');drawRows()}
  catch(e){if(e.name!=='AbortError'&&seq===policyVersion){pred=null;k('presult').hidden=true;k('pstate').textContent='本次推理失败：'+e.message;k('pstate').classList.add('ix-error')}}
  finally{if(seq===policyVersion)k('predict').disabled=false}
 }
 on(k('refresh'),'click',loadCatalog);on(k('episode'),'change',fetchEpisode);on(k('model'),'change',clearPrediction);on(k('jfilter'),'change',drawRows);on(k('predict'),'click',predict);on(k('seek'),'input',()=>{pause();renderFrame(Number(k('seek').value))});on(k('rate'),'change',()=>{let was=playing;pause();playSpeed=Number(k('rate').value);if(was)play()});
 on(k('play'),'click',play);on(k('prev'),'click',()=>{pause();renderFrame(idx-1)});on(k('next'),'click',()=>{pause();renderFrame(idx+1)});for(const b of root.querySelectorAll('[data-view]'))on(b,'click',()=>selectView(b.dataset.view));on(k('reset-view'),'click',()=>{scene.zoom=1;selectView('iso')});on(k('zoomin'),'click',()=>{scene.zoom=Math.min(2.5,(scene.zoom||1)*1.22);sceneDraw()});on(k('zoomout'),'click',()=>{scene.zoom=Math.max(.65,(scene.zoom||1)/1.22);sceneDraw()});
 const can=k('scene');on(can,'pointerdown',ev=>{if(ev.button!==0)return;drag=[ev.pointerId,ev.clientX,ev.clientY,...camera];can.setPointerCapture(ev.pointerId)});on(can,'pointermove',ev=>{if(!drag||ev.pointerId!==drag[0])return;camera=[drag[3]-(ev.clientX-drag[1])*.009, Math.min(1.56,Math.max(.03,drag[4]+(ev.clientY-drag[2])*.007))];selectView('',true)});on(can,'lostpointercapture',()=>{drag=null});on(can,'pointerup',()=>{drag=null});on(can,'keydown',ev=>{if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(ev.key))return;ev.preventDefault();camera=[camera[0]+(ev.key==='ArrowRight'?.12:ev.key==='ArrowLeft'?-.12:0),Math.min(1.56,Math.max(.03,camera[1]+(ev.key==='ArrowUp'?.1:ev.key==='ArrowDown'?-.1:0)))];selectView('',true)});
 on(document,'visibilitychange',()=>{if(document.hidden)pause()});const observer=new ResizeObserver(sceneDraw);observer.observe(can);loadCatalog();
 return {setActive(value){active=!!value;if(!active)pause();else requestAnimationFrame(sceneDraw)},destroy(){pause();active=false;listeners.abort();catalogReq?.abort();trajectoryReq?.abort();policyReq?.abort();observer.disconnect();container.replaceChildren()}};
}

// Shared by the closed-loop evidence player; inputs must be real world-space measurements.
export {bounds as replayBounds, renderScene as drawReplayScene, phaseName as replayPhaseName};
