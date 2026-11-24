import {setCompareRun, comparisonPair, showComparison, compareFailure, onPairChange} from './compare.js';
import {showQuality,showConvergence} from './quality.js';
const $=id=>document.getElementById(id);
const drawings={sliders:"M4 6h7m5 0h4M4 12h2m5 0h9M4 18h11m5 0h1M11 3v6M6 9v6M15 15v6",activity:'M2 12h4l3-8 6 16 3-8h4',wave:'M2 17l5-8 5 5 5-10 5 2',layers:'M3 8l9-5 9 5-9 5z M3 12l9 5 9-5 M3 16l9 5 9-5',cube:'M3 7l9-5 9 5v10l-9 5-9-5zM3 7l9 5 9-5M12 12v10',grid:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',chip:'M6 6h12v12H6z M9 9h6v6H9z M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m-3 6h3m12-6h3m-3 6h3',terminal:'M3 5h18v14H3z M6 9l3 3-3 3m6 0h4',refresh:'M20 10a8 8 0 00-14-4L3 9m0-5v5h5 M4 14a8 8 0 0014 4l3-3m0 5v-5h-5',target:'M12 3a9 9 0 100 18 9 9 0 000-18z M12 7a5 5 0 100 10 5 5 0 000-10z M12 11a1 1 0 100 2 1 1 0 000-2z',clock:'M12 3a9 9 0 100 18 9 9 0 000-18z M12 6v6l4 2'};
for(const e of document.querySelectorAll('[data-icon]'))e.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="'+(drawings[e.dataset.icon]||'')+'"/></svg>';
const fmt=v=>Number.isFinite(v)?v.toLocaleString('zh-CN'):'—';
const mse=v=>Number.isFinite(v)?v.toPrecision(3):'—';
const timeFmt=v=>{const d=new Date(v);return v&&!Number.isNaN(+d)?d.toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}):'—'};
const txt=(id,v)=>$(id).textContent=v;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusMap={running:'运行中',interrupted:'已中断',completed:'已完成',stale:'进程已退出',unknown:'进程未核实'};
const stageMap={collection:'采样',fit:'拟合',validate:'闭环验证',round_complete:'本轮完成',init:'初始化',waiting_gpu:'等待 GPU',benchmark:'转腕路径批宽对照',benchmark_complete:'批宽对照已完成'};
let remoteUI=null,remotePromise=null;
let rolloutUI=null,rolloutPromise=null,controlsUI=null,controlsPromise=null,requestedControlRun=null;
let page='training',data=null,mode='sample',lossRound=null,inferenceUI=null,inferencePromise=null,serial=0,busy=false,catalogTime=0,hardwareBusy=false;
async function api(path){const r=await fetch(path,{cache:'no-store',signal:AbortSignal.timeout(12000)});const j=await r.json();if(!r.ok)throw Error(j.error||('HTTP '+r.status));return j}
function connect(ok){$('connection-dot').className=ok?'ok':'off';txt('connection-word',ok?'已连接 · 本地':'连接中断')}
function note(v){$('problem').hidden=!v;txt('problem',v||'')}
function val(id,n,unit=''){ $(id).innerHTML=esc(n)+(unit?' <em>'+esc(unit)+'</em>':'') }
function currFit(s){const a=(s.history||[]).filter(h=>h.fit?.curve?.length||Number.isFinite(h.fit?.mse));return a[a.length-1]||null}
function drawChart(){
 const s=data;if(!s)return;
 for(const [id,key] of [['mode-sample','sample'],['mode-loss','loss'],['mode-val','val']])$(id).setAttribute('aria-pressed',mode===key);
 const fit=(s.history||[]).find(h=>h.round===lossRound&&h.fit)||currFit(s),sample=mode==='sample',physical=mode==='val';
 const points=(sample?s.timeline:physical?(s.history||[]).filter(h=>h.validation?.envs).map(h=>({round:h.round,held:100*h.validation.held/h.validation.envs,ending:100*h.validation.still_held_at_end/h.validation.envs,turn:100*h.validation.whole_turn/h.validation.envs})):fit?.fit?.curve)||[];
 const sets=sample?[{name:'活跃环境',key:'active',color:'#087f74'},...(points.some(p=>Number.isFinite(p.hold))?[{name:'曾拿稳环境',key:'hold',color:'#b87a2d'}]:[])]:physical?[{name:'曾拿稳',key:'held',color:'#087f74'},{name:'结束仍稳持',key:'ending',color:'#b87a2d'},{name:'整颗转动',key:'turn',color:'#605bb1'}]:[{name:'留出样本 MSE',key:'mse',color:'#087f74'}];
 $('chart-meta').innerHTML=sets.map(z=>'<span class="legend"><span class="swatch" style="background:'+z.color+'"></span>'+z.name+'</span>').join('')+'<span class="range">'+(sample?'当前批次':physical?'已验证的各轮':'第 '+fmt(fit?.round)+' 轮')+'</span>';
 txt('chart-foot',sample?(s.stage==='validate'?'横轴：仿真控制步。曾拿稳为本批独立验证种子的累计计数，左手由学生模型控制。':'横轴：仿真控制步。曾拿稳是当前批次的累计计数，采样中包含 teacher 动作。'):physical?'横轴：完成的轮次。纵轴：独立验证种子达到对应物理指标的比例（%）。':'横轴：参数更新次数。纵轴：留出样本的归一化均方误差。点轮次编号可查看该轮损失。');
 const xkey=sample?'step':physical?'round':'update', arr=points.filter(p=>Number.isFinite(p[xkey])&&sets.some(z=>Number.isFinite(p[z.key])));
 if(sample && points.some(p=>Number.isFinite(p.refills))) txt('chart-foot','横轴为整批控制步，纵轴为正在执行回合的物理槽数。');
 if(!arr.length){$('chart').innerHTML='<span class="empty">'+(sample?'当前批次还没有采样步记录':physical?'还没有完成的闭环验证轮次':'还没有可绘制的拟合记录')+'</span>';return}
 const w=Math.max(280,$('chart').clientWidth-24),h=Math.max(150,$('chart').clientHeight-5),L=sample?42:58,R=15,T=12,B=30,ww=w-L-R,hh=h-T-B;
 const high=Math.max(1,...arr.map(p=>p[xkey])), rawMax=Math.max(...arr.flatMap(p=>sets.map(z=>Number(p[z.key])||0))),ymax=sample?Math.max(s.progress?.envs||0,rawMax,1):physical?100:Math.max(rawMax,1e-9)*1.03;
 const x=v=>L+ww*v/high, y=v=>T+hh*(1-v/ymax), labelY=v=>physical?fmt(Math.round(v))+'%':sample?fmt(Math.round(v)):Number(v.toPrecision(2)).toString();
 let svg='<defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#159b8b" stop-opacity=".19"/><stop offset="100%" stop-color="#159b8b" stop-opacity=".015"/></linearGradient></defs>';
 for(let i=0;i<=4;i++){const v=ymax*i/4, yy=y(v);svg+='<line x1="'+L+'" x2="'+(w-R)+'" y1="'+yy+'" y2="'+yy+'" stroke="#dce5e6" stroke-dasharray="3 4"/><text x="'+(L-7)+'" y="'+(yy+3)+'" text-anchor="end">'+labelY(v)+'</text>'}
 const ticks=Math.min(w<380?3:5,high);for(let i=0;i<=ticks;i++){const v=Math.round(high*i/ticks);svg+='<text x="'+x(v)+'" y="'+(h-8)+'" text-anchor="middle">'+fmt(v)+'</text>'}
 sets.forEach((z,i)=>{const a=arr.filter(p=>Number.isFinite(p[z.key]));if(!a.length)return;const line=a.map(p=>x(p[xkey]).toFixed(2)+','+y(p[z.key]).toFixed(2)).join(' '),last=a.at(-1);
 if(i===0)svg+='<polygon points="'+x(a[0][xkey])+','+y(0)+' '+line+' '+x(last[xkey])+','+y(0)+'" fill="url(#area)"/>';
 svg+='<polyline points="'+line+'" fill="none" stroke="'+z.color+'" stroke-linejoin="round" stroke-width="2.5"/><circle cx="'+x(last[xkey])+'" cy="'+y(last[z.key])+'" r="3.5" stroke="white" stroke-width="1.5" fill="'+z.color+'"/>';
 });
 // Transparent keyboard/pointer markers expose exact recorded values without interpolating them.
 arr.forEach(p=>{const desc=(sample?'控制步 ':physical?'轮次 ':'更新 ')+p[xkey]+' · '+sets.map(z=>z.name+' '+(sample?fmt(p[z.key]):physical?p[z.key].toFixed(1)+'%':mse(p[z.key]))).join(' · ');svg+='<circle cx="'+x(p[xkey])+'" cy="'+y(p[sets[0].key]||0)+'" r="7" tabindex="0" class="hit" aria-label="'+esc(desc)+'"><title>'+esc(desc)+'</title></circle>'});
 $('chart').innerHTML='<svg style="font:10px var(--mono);fill:#6f8388" viewBox="0 0 '+w+' '+h+'">'+svg+'</svg>';
 $('chart').setAttribute('aria-label',(sample?'采样环境数':physical?'独立验证成功率':'验证损失')+'，共 '+arr.length+' 个记录点');
}
const perfLabels={'physics.step':'物理推进','teacher.turn_path':'教师转动路径规划','refill.begin':'开始槽位重置','refill.finish':'重置读回与检查','refill.reset_ik':'补位逆运动学','bank.add_step':'训练样本筛选与回传'};
function renderPerformance(s){
 const perf=s.performance||{}, p=s.progress||{};
 const current=perf.scope==='current_collection'||(!perf.scope&&s.stage==='collection');
 const counter=(key,logKey)=>perf[key]??(current?p[logKey]:null);
 const beforeSampling=s.stage==='waiting_gpu'&&!perf.available;
 txt('perf-scope',(beforeSampling?'诊断采样尚未开始':current?'当前采样批次':'最近已验收采样批次')+(Number.isFinite(perf.round)?' · 第 '+fmt(perf.round)+' 轮':''));
 txt('perf-stamp',perf.wall_time_utc?'快照 '+timeFmt(perf.wall_time_utc):current&&s.log_updated_at?'日志 '+timeFmt(s.log_updated_at):'—');
 const metrics=[['批次控制步',perf.rollout_batch_control_steps??counter('batch_control_steps','step')],['活跃物理槽',counter('active_slots','active'),counter('physical_slots','envs')],['已结束回合',counter('completed_episodes','completed'),counter('total_episodes','total_episodes')],['补入回合',counter('refills','refills')]];
 if(Number.isFinite(perf.elapsed_rollout_s)&&perf.elapsed_rollout_s>0&&Number.isFinite(perf.episode_control_steps))metrics.push(['有效环境步 / 秒',+(perf.episode_control_steps/perf.elapsed_rollout_s).toFixed(1)]);
 $('perf-metrics').innerHTML=metrics.filter(m=>Number.isFinite(m[1])).map(([label,v,total])=>'<div><dt>'+esc(label)+'</dt><dd>'+fmt(v)+(Number.isFinite(total)?'<small> / '+fmt(total)+'</small>':'')+'</dd></div>').join('');
 const items=Object.entries(perf.counters||{}).sort((a,b)=>b[1].host_wall_s-a[1].host_wall_s);
 const ordered=[],seen=new Set();
 function add(item,depth){const [name,v]=item;if(seen.has(name))return;seen.add(name);ordered.push([name,v,depth]);items.forEach(other=>{if(other[1].parent===name)add(other,depth+1)});}
 const names=new Set(items.map(([name])=>name));
 items.forEach(item=>{if(!names.has(item[1].parent))add(item,0)});items.forEach(item=>add(item,0));
 $('perf-table').hidden=!perf.available||!ordered.length;
 txt('perf-message',perf.available?'主机累计墙钟时间包含设备等待。父项已包含下级调用的耗时。':beforeSampling?(s.status==='running'?'诊断在取得 GPU 后开始采样并写入函数计时快照。':'诊断尚未开始采样，暂无函数计时快照。'):perf.reason==='unreadable_snapshot'?'函数计时快照暂时无法读取。':'此采样批次尚无函数计时快照。');
 $('perf-rows').innerHTML=ordered.map(([name,v,depth])=>'<tr><td>'+esc('　'.repeat(Math.min(depth,3))+(perfLabels[name]||name))+'</td><td>'+fmt(+v.host_wall_s.toFixed(3))+' 秒</td><td>'+fmt(v.calls)+'</td><td>'+(v.calls?fmt(+(1000*v.host_wall_s/v.calls).toFixed(2))+' ms':'—')+'</td></tr>').join('');
 const rates=perf.throughput;txt('perf-throughput',rates?'第 '+fmt(rates.round)+' 轮已验收采样 · '+fmt(+rates.effective_steps_per_worker_second.toFixed(2))+' 有效步/秒 · 倾转保持 '+fmt(+rates.entered_tilt_hold_per_worker_hour.toFixed(2))+' 回合/小时 · 达到腕端 '+fmt(+rates.wrist_endpoint_per_worker_hour.toFixed(2))+' 回合/小时（含教师执行回合）':'已验收采样吞吐 · —');
 const bench=perf.turn_path_benchmark;$('perf-benchmark').hidden=!bench;
 $('perf-bench-rows').innerHTML=(bench?.results||[]).map(v=>'<tr><td>'+(v.mode==='preserve_actual_groups'?'保留原分组':'跨控制步合并')+'</td><td>'+fmt(v.width)+'</td><td>'+fmt(+(v.median_seconds*1000).toFixed(2))+' ms</td><td>'+mse(v.maximum_abs_delta_rad)+' rad</td><td>'+(v.bitwise_equal?'是':'否')+'</td></tr>').join('');
}
function render(s){
 if(data?.id!==s.id)lossRound=null;data=s;const c=s.settings||{},p=s.progress||{},v=s.validation,fit=currFit(s);const k=s.active_round??s.completed_round;
 val('round-value',fmt(k),'/ '+fmt(c.rounds));txt('round-sub','已完成 '+fmt(s.completed_round)+' 轮 · '+(stageMap[s.stage]||'等待读取'));
 val('env-value',s.stage==='collection'?fmt(p.active):'—');txt('env-sub',Number.isFinite(p.total_episodes)?'本批 '+fmt(p.envs)+' 个物理槽 · '+fmt(p.total_episodes)+' 个回合':'每轮 '+fmt(c.envs)+' 个环境'+(p.envs?' · 本批 '+fmt(p.envs):''));
 val('val-value',v&&v.envs?(v.held/v.envs*100).toLocaleString('zh-CN',{maximumFractionDigits:1}):'—',v&&v.envs?'%':'');txt('val-sub',v?'最近闭环验证 · '+fmt(v.held)+' / '+fmt(v.envs)+' 拿稳':'尚无独立闭环验证');
 val('loss-value',mse(fit?.fit?.mse??fit?.fit?.curve?.at(-1)?.mse));txt('loss-sub',fit?'第 '+fmt(fit.round)+' 轮 · 留出样本 MSE':'尚无拟合指标');
 const isOn=s.status==='running',state=statusMap[s.status]||s.status||'未知';
 txt('status-pill',isOn&&s.stage==='waiting_gpu'?stageMap.waiting_gpu:state);$('status-pill').className='pill '+(isOn?'live':['stale','interrupted'].includes(s.status)?'warn':'');
 let collectionDetail=fmt(c.envs)+' 个采样环境';
 if(s.stage==='collection'){
  collectionDetail=Number.isFinite(p.total_episodes)
   ?'已结束 '+fmt(p.completed)+' / '+fmt(p.total_episodes)+' 回合 · 批次步 '+fmt(p.step)
   :'控制步 '+fmt(p.step)+' / '+fmt(c.horizon);
 }
 const collection=['collection','物理采样',collectionDetail];
 const fitDetail=s.stage==='fit'&&Number.isFinite(p.update)?fmt(p.update)+' / '+fmt(c.updates)+' 次更新':fmt(c.updates)+' 次更新 · batch '+fmt(c.batch);
 let phases=[collection,['fit','策略拟合',fitDetail],['validate','闭环验证',fmt(c.val_envs)+' 个独立种子 · 学生左手']];
 const turnProbe=s.kind==='diagnostic'&&(/^contact_hold90_turn_probe(?:_|$)/.test(s.id)||['waiting_gpu','benchmark','benchmark_complete'].includes(s.stage));
 if(turnProbe){
  phases=[['waiting_gpu','等待 GPU','等待共享 GPU 资源'],collection,['benchmark','转腕路径批宽对照','比较 32、64、128 三种批宽']];
 }
 const ix=phases.findIndex(([stage])=>stage===s.stage);
 $('pipeline').innerHTML=phases.map(([,label,detail],i)=>{const done=(ix>=0&&i<ix)||s.stage==='round_complete'||s.stage==='benchmark_complete';const sel=i===ix&&(!turnProbe||isOn);return '<li class="'+(done?'done':sel?'active':'')+'"><div class="stage-node">'+(done?'✓':'0'+(i+1))+'</div><div class="stage-info">'+esc(label)+'<small>'+esc(detail)+'</small></div></li>'}).join('');
 txt('last-event','最近日志 '+timeFmt(s.log_updated_at||s.updated_at));
 const rows=[...(s.history||[])].filter(h=>h.completed||h.validation).reverse();
 txt('round-count',fmt(s.completed_round)+' 轮完成');const ratio=(a,v)=>v&&Number.isFinite(v[a])?fmt(v[a])+' / '+fmt(v.envs):'—';
 $('history').innerHTML=rows.length?rows.map(h=>{const v=h.validation;return '<tr>'+['R'+String(h.round).padStart(2,'0'),fmt(h.fit?.updates),mse(h.fit?.mse),ratio('held',v),ratio('still_held_at_end',v),ratio('whole_turn',v),Number.isFinite(v?.mean_peak_height_m)?(v.mean_peak_height_m*100).toFixed(2)+' cm':'—'].map((a,i)=>'<td>'+(i===0&&h.fit?'<button class="roundbtn" data-round="'+Number(h.round)+'" title="查看本轮损失">'+esc(a)+'</button>':esc(a))+'</td>').join('')+'</tr>'}).join(''):'<tr><td class="emptyrow" colspan="7">本实验还没有完成整轮验证。</td></tr>';
 const logs=$('log-content'),bottom=logs.scrollHeight-logs.scrollTop-logs.clientHeight<25,old=logs.dataset.run;logs.textContent=(s.log||[]).join('\n')||'当前阶段没有可用日志。';logs.dataset.run=s.id;if(bottom||old!==s.id)logs.scrollTop=logs.scrollHeight;
 note(s.status==='stale'?'进程已经退出；这里保留最后一次写入的进度。':s.status==='interrupted'?'运行已中断。'+(s.last_error?' '+s.last_error:' 可查阅下方最后一段日志。'):s.status==='unknown'?'当前无法核实进程，以下数据取自最近写入的日志。':'');
 renderPerformance(s);drawChart()
}
async function refresh(force=false){
 const qualityPage=page==='trends'||page==='compare';
 if(page!=='training'&&!qualityPage)return;
 if(busy&&!force)return;
 const tag=++serial;busy=true;
 try{
  if(force||!catalogTime||Date.now()-catalogTime>60000){const {runs}=await api('/api/runs');if(tag!==serial)return;const sel=$('run-choice'),old=sel.value;sel.replaceChildren(...runs.map(r=>new Option((r.kind==='diagnostic'?'[短测] ':'')+r.id,r.id)));if(sel.dataset.selectNext && runs.some(r=>r.id===sel.dataset.selectNext)){sel.value=sel.dataset.selectNext;delete sel.dataset.selectNext}else if(runs.some(r=>r.id===old))sel.value=old;catalogTime=sel.dataset.selectNext?0:Date.now();if(!runs.length)throw Error('没有发现 90 毫米训练目录。')}
  const id=$('run-choice').value;
  if(page==='training'||page==='trends')showConvergence(page,id);
  if(qualityPage){
   if(page==='compare')setCompareRun(id);else showQuality(page,id);
   const quality=await api('/api/quality?run='+encodeURIComponent(id));
   if(tag!==serial)return;
   if(quality.id!==id||!Array.isArray(quality.rounds))throw Error('验证结果与所选实验不一致');
   if(page==='compare'){
    const pair=comparisonPair(id,quality.rounds);
    if(pair){
      const params=new URLSearchParams({run:id,base:pair.base,target:pair.target});
      const data=await api('/api/compare?'+params);
      if(tag!==serial)return;
      showComparison(data);
    }
   }else{showQuality(page,id,quality);showConvergence(page,id,quality)}
   note('');
  }else{
   const urlRun=encodeURIComponent(id);
   const [snap,analysis]=await Promise.allSettled([api('/api/training?run='+urlRun),api('/api/quality?run='+urlRun)]);
   if(tag!==serial)return;
   if(snap.status==='rejected')throw snap.reason;
   render(snap.value);
   if(analysis.status==='fulfilled'&&analysis.value?.id===id&&Array.isArray(analysis.value.rounds)){
    showConvergence(page,id,analysis.value);
   }else{note('跨轮趋势暂时未能更新；采样进度仍来自本次读取。')}
  }
  connect(true);
 }catch(e){if(tag===serial){connect(false);if(page==='compare')compareFailure();note('无法读取训练数据：'+e.message+'。可点击刷新重试。')}}finally{if(tag===serial)busy=false}
}
async function loadGpu(){if(hardwareBusy)return;hardwareBusy=true;try{const h=await api('/api/hardware');txt('perf-device',h.available?'GPU 0 整机利用率 '+fmt(h.util_percent)+'% · 显存 '+fmt(h.used_mib)+' / '+fmt(h.total_mib)+' MiB · '+timeFmt(h.sampled_at):'GPU 设备遥测暂不可用');txt('compact-gpu',h.available?'本机 GPU '+h.util_percent+'% · '+(h.used_mib/1024).toFixed(1)+' GiB':'本机 GPU —');txt('gpu-name',h.available?h.name.replace(/^NVIDIA GeForce /,'').replace(' Laptop GPU',' Laptop'):'遥测未连接');txt('gpu-util',h.available?fmt(h.util_percent)+'%':'—');txt('gpu-memory',h.available?fmt(h.used_mib)+' / '+fmt(h.total_mib)+' MiB':'—');$('gpu-bar').style.width=(h.available?Math.min(h.util_percent,100):0)+'%'}catch{txt('perf-device','GPU 设备遥测暂不可用');txt('compact-gpu','本机 GPU —');txt('gpu-name','遥测未连接');txt('gpu-util','—');txt('gpu-memory','—');$('gpu-bar').style.width='0'}finally{hardwareBusy=false}}
async function navigate(){
 let hash=location.hash.slice(1);
 if(hash==='outcomes'){hash='compare';history.replaceState(null,'','#compare')}
 page=['inference','rollout'].includes(hash)?'inference':['controls','trends','compare','remote'].includes(hash)?hash:'training';
 serial++;busy=false;
 const titles={training:'训练监控',controls:'训练调参',inference:'推理工作台',trends:'训练趋势',compare:'模型对比',remote:'远端训练'};
 const runsPage=['training','trends','compare'].includes(page);
 $('run-header').hidden=!runsPage;
 $('train-page').hidden=page!=='training';$('trend-page').hidden=page!=='trends';$('compare-page').hidden=page!=='compare';
 $('infer-page').hidden=page!=='inference';$('control-page').hidden=page!=='controls';txt('crumb',titles[page]);note('');
 $('remote-page').hidden=page!=='remote';remoteUI?.setActive?.(page==='remote');
 if(runsPage){
  $('training-eyebrow').hidden=page!=='training';
  txt('run-title',titles[page]);
  txt('run-description',({training:'90 毫米魔方 · GPU 并行采样与闭环验证',trends:'逐轮核对拟合误差、权重变化和独立验证的达标表现。',compare:'在相同验证种子上，查看换一轮模型后新增和退步的回合。'})[page]);
 }
 for(const a of document.querySelectorAll('.nav')){const active=a.getAttribute('href')==='#'+page;a.classList.toggle('selected',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')}
 controlsUI?.setActive?.(page==='controls');
 if(page==='remote'){
  window.scrollTo(0,0);inferenceUI?.setActive?.(false);rolloutUI?.setActive?.(false);
  if(!remotePromise){$('remote-root').textContent='正在连接远端训练…';remotePromise=import('/static/remote.js').then(m=>m.mountRemote($('remote-root'))).then(ui=>{remoteUI=ui;ui?.setActive?.(page==='remote');return ui}).catch(e=>{remotePromise=null;$('remote-root').textContent='远端页面加载失败：'+e.message})}
  api('/api/health').then(()=>connect(true)).catch(()=>connect(false));
 }else if(page==='controls'){
  window.scrollTo(0,0);inferenceUI?.setActive?.(false);rolloutUI?.setActive?.(false);
  if(!controlsPromise){$('controls-root').textContent='正在读取可调参数…';controlsPromise=import('/static/controls.js').then(m=>m.mountControls($('controls-root'),{onStarted:(id)=>{catalogTime=0;note('');$('run-choice').dataset.selectNext=id}})).then(ui=>{controlsUI=ui;ui?.setActive?.(page==='controls');return ui}).catch(e=>{controlsPromise=null;$('controls-root').textContent='参数页加载失败：'+e.message})}
  if(requestedControlRun){const id=requestedControlRun;requestedControlRun=null;controlsPromise?.then(ui=>{if(page==='controls')ui?.setRun?.(id)})}
  api('/api/health').then(()=>connect(true)).catch(()=>connect(false));
 }else if(page==='inference'){
  window.scrollTo(0,0);
  const fresh=hash==='rollout';
  $('inference-root').hidden=fresh;$('rollouts-root').hidden=!fresh;
  for(const [id,on] of [['archive-tab',!fresh],['rollout-tab',fresh]]){const a=$(id);if(on)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')}
  if(fresh&&!rolloutPromise){$('rollouts-root').textContent='正在检查闭环推演可用性…';rolloutPromise=import('/static/rollouts.js').then(m=>m.mountRollouts($('rollouts-root'))).then(ui=>{rolloutUI=ui;ui?.setActive?.(location.hash==='#rollout');return ui}).catch(e=>{rolloutPromise=null;$('rollouts-root').textContent='闭环视图加载失败：'+e.message})}
  rolloutUI?.setActive?.(fresh);
  if(!fresh&&!inferencePromise){$('inference-root').textContent='正在读取模型与轨迹…';inferencePromise=import('/static/inference.js').then(m=>m.mountInference($('inference-root'))).then(ui=>{inferenceUI=ui;ui?.setActive?.(page==='inference'&&location.hash!=='#rollout');return ui}).catch(e=>{inferencePromise=null;$('inference-root').textContent='推理面板加载失败：'+e.message;connect(false)})}
  inferenceUI?.setActive?.(!fresh);api('/api/health').then(()=>connect(true)).catch(()=>connect(false));
 }else{
  inferenceUI?.setActive?.(false);rolloutUI?.setActive?.(false);
  if(page==='trends')showQuality(page,$('run-choice').value);
  if(page==='compare')setCompareRun($('run-choice').value);
  await refresh(true);
  if(hash!==location.hash.slice(1))return;
  if(['rounds','logs'].includes(hash))$(hash).scrollIntoView();else window.scrollTo(0,0);
 }
}
onPairChange(()=>{if(page==='compare')refresh(true)});
$('history').addEventListener('click',e=>{const b=e.target.closest('button[data-round]');if(!b)return;lossRound=Number(b.dataset.round);mode='loss';drawChart();$('chart-meta').scrollIntoView({block:'center'})});
$('mode-val').onclick=()=>{mode='val';drawChart()};
document.querySelector('.config-link').addEventListener('click',()=>{requestedControlRun=$('run-choice').value||null});
$('run-choice').addEventListener('change',()=>{
 if(page==='trends')showQuality(page,$('run-choice').value);
 if(page==='compare')setCompareRun($('run-choice').value);
 refresh(true);
});$('refresh').addEventListener('click',()=>{refresh(true);loadGpu()});
$('mode-sample').onclick=()=>{mode='sample';drawChart()};$('mode-loss').onclick=()=>{mode='loss';drawChart()};
$('log-toggle').onclick=()=>{const e=$('log-content');e.hidden=!e.hidden;$('log-toggle').setAttribute('aria-expanded',String(!e.hidden));txt('log-toggle',e.hidden?'展开日志':'收起日志')};
window.addEventListener('hashchange',navigate);window.addEventListener('resize',()=>{if(page==='training')drawChart();else if(page==='trends')showQuality(page,$('run-choice').value)});document.addEventListener('visibilitychange',()=>{if(!document.hidden){refresh();loadGpu()}});
setInterval(()=>{if(!document.hidden){refresh();loadGpu()}},5000);navigate();loadGpu();
