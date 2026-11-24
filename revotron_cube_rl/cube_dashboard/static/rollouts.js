import {replayBounds,drawReplayScene,replayPhaseName} from './inference.js';
const goodNum=n=>typeof n==='number'&&Number.isFinite(n);
const fmt=(n,places=1)=>goodNum(n)?(Number(n.toFixed(places))||0).toFixed(places):'—';
const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n};
const options=(v,text)=>{const n=el('option',text);n.value=v;return n};
const statusText=s=>({running:'运行中',completed:'已完成',failed:'失败',interrupted:'已中断'}[s]||'未知');
const UI=`<section class="ro">
 <header class="ro-head"><div><span>PHYSICS ROLLOUT</span><h2>新模型闭环仿真</h2></div><span class="ro-badge" data-r="badge">检查运行条件…</span></header>
 <p class="ro-description">从复位开始，由所选模型控制左手，重新运行 90 mm 魔方的物理仿真。完成后可回放新生成的轨迹。</p>
 <div class="ro-control"><label>待验证的模型<select data-r="model" aria-label="闭环仿真模型" disabled></select></label><button type="button" data-r="start" disabled>启动闭环验证</button></div>
 <div data-r="availability" class="ro-note" role="status">正在读取 GPU 和训练状态…</div>
 <div data-r="feedback" class="ro-feedback" aria-live="polite"></div>
 <div class="ro-history"><div class="ro-title"><h3>闭环任务记录</h3><button type="button" data-r="refresh">刷新</button></div><div data-r="empty" class="ro-empty">正在读取任务…</div><div data-r="jobs" class="ro-jobs"></div></div>
 <div data-r="player" class="ro-player" hidden>
  <div class="ro-title"><div><span class="ro-kicker">CLOSED-LOOP REPLAY</span><h3 data-r="title"></h3></div><button type="button" data-r="close">收起回放</button></div>
  <div class="ro-framegrid"><div class="ro-view"><canvas data-r="canvas" aria-label="新模型在独立物理仿真中产生的三维测量轨迹"></canvas><span data-r="viewlabel"></span><div class="ro-views" aria-label="视角"><button data-view="iso" aria-pressed="true">等距</button><button data-view="side" aria-pressed="false">侧面</button><button data-view="top" aria-pressed="false">俯视</button></div></div>
  <div class="ro-stats"><div><small>当前阶段</small><b data-r="phase"></b></div><div><small>离桌高度</small><b data-r="height"></b></div><div><small>桌面接触</small><b data-r="contact"></b></div><div><small>整条轨迹最长拿稳</small><b data-r="hold"></b></div></div></div>
  <div class="ro-transport"><button type="button" data-r="play">播放</button><input data-r="seek" type="range" min="0" max="1" value="0" step="1" aria-label="闭环回放进度"><output data-r="time"></output><select data-r="speed" aria-label="闭环回放速度"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
  <div class="ro-caption" data-r="summary"></div>
 </div>
</section>`;
async function json(path,signal,body){
 const params={cache:'no-store',signal};
 if(body!==undefined)Object.assign(params,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const r=await fetch(path,params);let d;try{d=await r.json()}catch{throw new Error('服务器返回了无效响应（HTTP '+r.status+'）')}
 if(!r.ok||d.error)throw new Error(d.error?.message||d.error||'HTTP '+r.status);return d;
}
const views={iso:[.87,.56],side:[Math.PI/2,.06],top:[Math.PI/2,1.55]};
export function mountRollouts(container){
 container.innerHTML=UI;const root=container.firstElementChild, refs={};for(const n of root.querySelectorAll('[data-r]'))refs[n.dataset.r]=n;
 const k=n=>refs[n],listeners=new AbortController(),on=(n,name,cb)=>n.addEventListener(name,cb,{signal:listeners.signal});
 let status=null,models=[],lastModels=0,active=true,destroyed=false,posting=false,refreshing=false,timer=0,refreshingModels=null,dataReq=null,statReq=null,postReq=null,rec=null,scene=null,frames=[],index=0,angle=views.iso,playing=false,raf=0,t0=0,step0=0,selected='',token=0;
 function say(s,err=false){k('feedback').textContent=s;k('feedback').classList.toggle('ro-err',err)}
 function colorState(name){k('badge').dataset.state=name}
 function startState(){const known=status?.available===true&&typeof status.reason!=='undefined';k('start').disabled=!known||!k('model').value||posting; k('start').textContent=posting?'正在提交…':'启动闭环验证';k('model').disabled=!models.length||posting}
 async function loadModels(){
  if(refreshingModels||Date.now()-lastModels<60000)return refreshingModels;
  refreshingModels=(async()=>{
   const d=await json('/api/inference/catalog',listeners.signal);if(destroyed)return;
   const old=k('model').value;models=d.models||[];k('model').replaceChildren(...models.map(m=>options(m.id,m.label||m.id)));
   const preferred=models.some(m=>m.id===old)?old:models.at(-1)?.id;if(preferred)k('model').value=preferred;lastModels=Date.now();startState()
  })().finally(()=>{refreshingModels=null});return refreshingModels;
 }
 function jobsRows(jobs){
   const host=k('jobs');host.replaceChildren();k('empty').hidden=jobs.length>0;k('empty').textContent='暂无闭环任务。点击「启动闭环验证」后，记录会出现在这里。';
   for(const j of jobs){
    const line=el('div','');line.className='ro-job';const left=el('div',''),label=el('strong',j.label||j.model||'未命名模型'),seed=goodNum(j.seed)?' · Seed '+j.seed:'';
    const meta=el('small',(j.created_at?String(j.created_at).replace('T',' ').slice(0,19):j.id)+seed+(j.sha256?' · '+j.sha256.slice(0,10):''));left.append(label,meta);
    if(j.reason){const why=el('p',j.reason);left.append(why)}const right=el('div','');right.className='ro-actions';const tag=el('span',statusText(j.status));tag.className='ro-jobstate';tag.dataset.state=j.status;right.append(tag);
    if(j.status==='completed'){const btn=el('button',selected===j.id?'重新载入':'回放');btn.type='button';btn.addEventListener('click',()=>openJob(j));right.append(btn)}line.append(left,right);host.append(line)
   }
 }
 function drawStatus(d){
   status=d;const avail=d.available===true,why=String(d.reason||'');const running=!!d.active_job;
   k('badge').textContent=avail?'可以启动':running?'闭环任务运行中':'暂不可启动';colorState(avail?'ready':running?'running':'busy');
   k('availability').textContent=(avail?'GPU 空闲。点击后会实际运行仿真，启动前将再次检查训练锁。':(why||'无法确认 GPU 空闲，本次不能启动。'))+(Array.isArray(d.panel_seeds)?' 验证种子：'+d.panel_seeds.join('、')+'；回放 Seed '+d.seed+'。':'');
   jobsRows(Array.isArray(d.jobs)?d.jobs:[]);startState()
 }
 async function refresh(force=false){
  if(destroyed||!active||refreshing)return;refreshing=true;statReq=new AbortController();k('refresh').disabled=true;
  try{const [d,m]=await Promise.allSettled([json('/api/rollouts',statReq.signal),loadModels()]);if(destroyed)return;
    if(d.status==='rejected')throw d.reason;drawStatus(d.value);if(m.status==='rejected'&&!models.length){say('模型列表不可用：'+m.reason.message,true)}
  }catch(e){if(e.name==='AbortError')return;status=null;k('badge').textContent='状态未知';colorState('busy');k('availability').textContent='无法读取运行条件：'+e.message;k('empty').textContent='无法读取任务记录。';startState()}
  finally{refreshing=false;if(!destroyed){k('refresh').disabled=false;clearTimeout(timer);if(active)timer=setTimeout(refresh,status?.active_job?4000:12000)}}
 }
 async function start(){
  if(posting||status?.available!==true||!models.some(m=>m.id===k('model').value))return;
  posting=true;postReq=new AbortController();startState();say('正在请求启动，服务器会重新检查训练与 GPU 状态…');
  try{const j=await json('/api/rollouts',postReq.signal,{model:k('model').value});if(destroyed)return;status=null;say('闭环任务已提交'+(j.id?'：'+j.id:'')+'。完成后可在下方回放。')}
  catch(e){if(e.name!=='AbortError'){status=null;say('这次没有确认启动：'+e.message,true)}}
  finally{posting=false;if(!destroyed){startState();refresh(true)}}
 }
 const hz=()=>rec?.episode?.hz||30;
 const draw=()=>{if(active&&frames.length&&!k('player').hidden)drawReplayScene(k('canvas'),scene,frames,index,angle)};
 function pause(){playing=false;cancelAnimationFrame(raf);k('play').textContent='播放'}
 function render(i){
  index=Math.max(0,Math.min(frames.length-1,i));if(!frames.length)return;const f=frames[index];
  k('phase').textContent=replayPhaseName(f.phase);k('height').textContent=goodNum(f.height_m)?fmt(f.height_m*1000)+' mm':'—';k('contact').textContent=f.table_contact===true?'有接触':f.table_contact===false?'无接触':'未记录';
  k('seek').value=String(index);const t=(f.step-frames[0].step)/hz(),end=(frames.at(-1).step-frames[0].step)/hz();k('seek').setAttribute('aria-valuetext','第 '+f.step+' 步，'+fmt(t,2)+' 秒');k('time').textContent=fmt(t,2)+' / '+fmt(end,2)+' s';k('viewlabel').textContent='SEED '+(rec.episode.seed??'—')+'   /   STEP '+f.step;draw()
 }
 function tick(ts){if(!playing||!active)return;const step=step0+(ts-t0)/1000*hz()*Number(k('speed').value);let a=index,b=frames.length-1;while(a<b){const m=Math.ceil((a+b)/2);if(frames[m].step<=step)a=m;else b=m-1}if(a!==index)render(a);if(index===frames.length-1){pause();return}raf=requestAnimationFrame(tick)}
 function play(){if(playing){pause();return}if(!active||!frames.length)return;if(index===frames.length-1)render(0);playing=true;k('play').textContent='暂停';t0=performance.now();step0=frames[index].step;raf=requestAnimationFrame(tick)}
 function close(){token++;dataReq?.abort();pause();rec=null;scene=null;frames=[];selected='';k('player').hidden=true}
 async function openJob(j){
   close();const seq=token;selected=j.id;say('正在读取已完成的真实轨迹…');dataReq=new AbortController();
   try{
    const d=await json('/api/rollouts/frames?job='+encodeURIComponent(j.id),dataReq.signal);if(destroyed||seq!==token)return;
    if(!Array.isArray(d.frames)||!d.frames.length||!d.episode)throw new Error('未找到可播放的物理帧。');
    rec=d;frames=d.frames;const edge=goodNum(d.episode.edge_mm)?d.episode.edge_mm/1000:null;scene={...replayBounds(frames,d.table_z_m),table:d.table_z_m,edge,zoom:1};angle=views.iso;
    for(const b of root.querySelectorAll('[data-view]'))b.setAttribute('aria-pressed',String(b.dataset.view==='iso'));
    k('title').textContent=(j.label||j.model||'模型')+' · 独立闭环';k('seek').max=String(frames.length-1);k('player').hidden=false;
    const s=d.summary||{};k('hold').textContent=goodNum(s.best_hold_s)?fmt(s.best_hold_s,2)+' s':'未记录';
    k('summary').textContent='终止：'+(s.end_reason||'未记录')+'　·　整颗转动：'+(s.whole_turn===true?'达到':s.whole_turn===false?'未达到':'未记录')+'　·　单层转动：'+(s.face_turn===true?'达到':s.face_turn===false?'未达到':'未测量')+(d.checkpoint_sha256?'　·　检查点 '+d.checkpoint_sha256.slice(0,12):'');
    say('');render(0);jobsRows(status?.jobs||[])
   }catch(e){if(seq===token&&e.name!=='AbortError'){selected='';say('轨迹读取失败：'+e.message,true)}}
 }
 on(k('refresh'),'click',()=>{lastModels=0;refresh(true)});on(k('model'),'change',startState);on(k('start'),'click',start);on(k('play'),'click',play);on(k('seek'),'input',()=>{pause();render(Number(k('seek').value))});on(k('speed'),'change',()=>{const was=playing;pause();if(was)play()});on(k('close'),'click',()=>{close();jobsRows(status?.jobs||[])});
 for(const b of root.querySelectorAll('[data-view]'))on(b,'click',()=>{angle=views[b.dataset.view];for(const c of root.querySelectorAll('[data-view]'))c.setAttribute('aria-pressed',String(c===b));draw()});
 on(document,'visibilitychange',()=>{if(document.hidden)pause()});const observer=new ResizeObserver(draw);observer.observe(k('canvas'));refresh();
 return {setActive(on){active=!!on;clearTimeout(timer);if(!active)pause();else{refresh();requestAnimationFrame(draw)}},destroy(){destroyed=true;active=false;clearTimeout(timer);statReq?.abort();postReq?.abort();close();listeners.abort();observer.disconnect();container.replaceChildren()}}
}
