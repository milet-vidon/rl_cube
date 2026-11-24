/* Standalone training workbench; a field is offered only when the trainer declares it. */
const ENDPOINT='/api/training/control';
const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e};
const opt=(id,label)=>new Option(label,id);
const labelStatus=v=>({running:'运行中',starting:'启动中',interrupted:'已中断',failed:'失败',completed:'已完成',stale:'进程已退出',unknown:'状态未知',stopped:'已停止',exited:'已退出',submitted:'已提交'}[v]||String(v||'状态未知'));
const nice=n=>Number.isFinite(n)?(Math.abs(n)>=10000?Number(n).toLocaleString('zh-CN',{maximumFractionDigits:7}):String(n)):'—';
const time=s=>{if(!s)return '';let d=new Date(s);return Number.isNaN(+d)?'':d.toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})};
const eq=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)&&a===b;
const asNumList=o=>Object.entries(o||{}).every(([,v])=>Number.isFinite(v));
const proposal=d=>d?.pending?.values||d?.effective||d?.defaults||{};
const revision=d=>d?.pending?.revision??d?.effective_revision??null;
const defaultParent=cards=>cards.find(x=>x.id==='r15'||x.run==='contact_hold_cuda_long_v1')?.id||cards[0]?.id||'';
const groupDefs=[
 {id:'scale',icon:'01',en:'SCHEDULE',label:'训练轮数与批次',hint:'总轮数决定训练何时结束。每轮更新次数和批量大小分别控制优化步数和每步抽取的样本数。',keys:['rounds','updates','batch']},
 {id:'reward',icon:'02',en:'OUTCOME WEIGHTS',label:'任务奖励权重',hint:'提高某项权重，会让取得该结果的训练轨迹在动作拟合时占更大比重。没有达成该结果的样本时，这一项不生效。',keys:['hold_score','ending_score','turn_score']},
 {id:'phase',icon:'03',en:'PHASE SAMPLING',label:'阶段采样权重',hint:'调整各阶段训练样本被选中的相对机会。1 表示原始比例。',keys:['grasp_weight','lift_weight','turn_weight']},
 {id:'advanced',icon:'04',en:'OPTIMIZER & SIMULATION',label:'优化器与仿真环境',hint:'调整学习率、回放窗口与并行采样规模。只在新建时可设置的参数会标出。',keys:['lr','replay_rounds','envs','val_envs','horizon']}
];
const fieldLabels={rounds:'目标总轮数',updates:'每轮更新次数',batch:'批次大小',hold_score:'曾经拿稳',ending_score:'结束时仍拿稳',turn_score:'整颗转动',grasp_weight:'抓取阶段',lift_weight:'抬升阶段',turn_weight:'转动阶段',lr:'学习率',replay_rounds:'回放轮数',envs:'并行采样环境',val_envs:'独立验证环境',horizon:'每回合最多控制步'};
const UI=`
<div class="ct-workspace">
  <div class="ct-scope">
    <div class="ct-modebox"><span class="ct-overline">配置对象</span><div class="ct-mode" role="group" aria-label="配置对象"><button type="button" data-ct="new" aria-pressed="true">新建实验</button><button type="button" data-ct="existing" aria-pressed="false">已有实验</button></div></div>
    <label class="ct-target"><span data-ct="targetLabel">新实验名称</span><input data-ct="name" type="text" maxlength="72" autocomplete="off" spellcheck="false" aria-label="新实验名称"><select data-ct="run" aria-label="要配置的已有实验" hidden></select></label>
    <div class="ct-reload"><button type="button" data-ct="refresh" class="ct-link" aria-label="刷新配置和启动条件">刷新状态</button></div>
  </div>
  <p class="ct-load" data-ct="load" role="status">正在读取训练引擎支持的参数…</p>
  <div data-ct="content" class="ct-layout" hidden>
   <div class="ct-main">
    <div data-ct="groups"></div>
    <section class="ct-card ct-source" data-ct="sourceCard" hidden><div class="ct-sectionhead"><span class="ct-overline">INITIAL POLICY</span><h2>起始模型</h2></div><label><span>初始化权重</span><select data-ct="source" aria-label="起始模型"></select></label><p data-ct="sourceHint"></p></section>
   </div>
   <aside class="ct-side" aria-label="配置摘要和操作">
    <section class="ct-card ct-review">
     <div class="ct-reviewhead"><div><span class="ct-overline">CONFIG PREVIEW</span><h2>提交预览</h2></div><span class="ct-dotpill" data-ct="state"></span></div>
     <div class="ct-metrics"><div><span>目标轮数</span><strong data-ct="rounds">—</strong></div><div><span>每批样本</span><strong data-ct="batch">—</strong></div></div>
     <div class="ct-budget" data-ct="budget"></div>
     <div class="ct-effect" data-ct="effect"></div><div data-ct="stale" class="ct-pending" hidden>其他窗口已提交新版本。<button type="button" data-ct="reloadDraft" class="ct-link">重载服务端版本</button></div>
     <div data-ct="compare" class="ct-compare"></div>
     <p class="ct-reason" data-ct="reason" hidden></p>
     <div class="ct-actions"><button class="ct-primary" data-ct="start" type="button">启动新实验</button><button class="ct-primary" data-ct="apply" type="button" hidden>提交下一轮配置</button><button class="ct-secondary" data-ct="resume" type="button" hidden>从断点继续训练</button>
      <button data-ct="fork" type="button" class="ct-secondary" hidden>复制到新实验</button></div>
     <div class="ct-shelf"><div class="ct-shelfhead"><span>参数草稿</span><button type="button" class="ct-link" data-ct="save">保存草稿</button></div><div class="ct-shelfbody"><small data-ct="draftInfo"></small><button type="button" class="ct-link" data-ct="restore" hidden>载入草稿</button></div></div>
     <div class="ct-pending" data-ct="pending" hidden><span data-ct="pendingText"></span><button type="button" data-ct="loadPending" class="ct-link">查看待生效参数</button></div>
     <div role="status" aria-live="polite" class="ct-feedback" data-ct="feedback" hidden></div>
     <a href="#training" data-ct="monitor" class="ct-link ct-monitor" hidden>查看训练监控 →</a>
    </section>
    <section class="ct-card ct-jobs" data-ct="jobBox" hidden><span class="ct-overline">ACTIVITY</span><h3>最近启动记录</h3><div data-ct="jobs"></div></section>
   </aside>
  </div>
</div>`;
async function request(path,signal,body){
 const options={cache:'no-store',signal};if(body!==undefined)Object.assign(options,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const r=await fetch(path,options);let payload;try{payload=await r.json()}catch{throw Error('服务暂不可用（HTTP '+r.status+'）')}
 if(!r.ok||payload.error)throw Error(payload.error?.message||payload.error||'HTTP '+r.status);return payload;
}
export function mountControls(container,{onStarted}={}){
 container.innerHTML=UI;
 const refs=Object.fromEntries([...container.querySelectorAll('[data-ct]')].map(n=>[n.dataset.ct,n]));const q=k=>refs[k];
 const listeners=new AbortController();const on=(n,ev,handler)=>n.addEventListener(ev,handler,{signal:listeners.signal});
 let active=true,destroyed=false,posting=false,target='',preferred='',data=null,runs=[],controls=new Map(),requestId=0,reading=null,timer=0,pane='new',valid=true,editCache=new Map(),formRevision=null,formBasis={},conflict=false,fieldListeners=null;
 const setText=(key,v)=>{q(key).textContent=v};
 function note(text,bad=false){q('feedback').hidden=!text;setText('feedback',text);q('feedback').classList.toggle('ct-error',!!bad)}
 const actual=()=>Object.fromEntries([...controls].map(([k,{input}])=>[k,input.value===''?NaN:input.valueAsNumber]));
 const fromData=()=>({...data?.defaults,...data?.effective});
 function remember(){if(!data)return;editCache.set(target,{raw:Object.fromEntries([...controls].map(([k,{input}])=>[k,input.value])),name:q('name').value,source:q('source').value,revision:formRevision,basis:{...formBasis}})}
 function setValues(v){for(const [key,{input,range}] of controls){if(Object.hasOwn(v||{},key)){input.value=String(v[key]);if(range&&Number.isFinite(input.valueAsNumber))range.value=input.value}}preview()}
 function disabledButton(key,x){q(key).disabled=!!x}
 function runsOptions(){
  const old=q('run').value;const options=runs.filter(r=>r?.id).map(r=>opt(r.id,(r.kind==='diagnostic'?'短测 · ':'')+r.id));
  q('run').replaceChildren(...(options.length?options:[opt('','未找到已有实验')]));
  const value=[target,preferred,old].find(v=>v&&runs.some(r=>r.id===v));if(value)q('run').value=value;
 }
 function createField(f,group){
  const row=el('div','ct-field'),label=el('label','ct-fieldtitle',fieldLabels[f.key]||f.label||f.key);
  const input=el('input','ct-num');input.type='number';input.id='ct-value-'+f.key;label.htmlFor=input.id;
  input.autocomplete='off';input.required=true;input.min=f.min;input.max=f.max;
  input.step=f.type==='int'?(f.step||1):'any'; // limits are validated by the backend as well.
  input.value=String((fromData())[f.key]??f.default);input.disabled=!!data?.run&&f.mutable===false;
  const head=el('div','ct-fieldhead');head.append(label);
  if(input.disabled)head.append(el('small','ct-locked','仅新建'));
  const line=el('div','ct-numberline');let range=null;
  if(group==='reward'||group==='phase'){
    range=el('input','ct-range');range.type='range';range.min=f.min;range.max=f.max;range.step=f.step||.1;range.value=input.value;
    range.disabled=input.disabled;range.setAttribute('aria-label',label.textContent+'权重滑块');
    range.addEventListener('input',()=>{input.value=range.value;preview()},{signal:fieldListeners.signal});line.append(range);
  }
  row.append(head);line.append(input);row.append(line);
  const help=el('small','ct-fieldhelp',f.help||f.description||'');help.id='ct-help-'+f.key;
  if(!help.textContent)help.textContent='范围 '+nice(f.min)+' – '+nice(f.max);
  input.setAttribute('aria-describedby',help.id);row.append(help);
  input.addEventListener('input',()=>{if(range&&Number.isFinite(input.valueAsNumber))range.value=input.value;preview()},{signal:fieldListeners.signal});
  controls.set(f.key,{input,range,f,row});return row;
 }
 function buildGroups(){
   fieldListeners?.abort();fieldListeners=new AbortController();q('groups').replaceChildren();controls=new Map();const schema=data.schema||[];const map=new Map(schema.map(f=>[f.key,f]));const used=new Set();
   for(const d of groupDefs){
    let list=d.keys.map(k=>map.get(k)).filter(Boolean);
    if(d.id==='advanced'){const known=new Set(groupDefs.flatMap(x=>x.keys));list=list.concat(schema.filter(f=>!known.has(f.key)))}
    list=list.filter(f=>!used.has(f.key));if(!list.length)continue;
    const card=el('section','ct-card ct-group ct-'+d.id),header=el('div','ct-sectionhead'),sub=el('div','ct-kicker');
    sub.append(el('b','ct-id',d.icon),el('span','ct-overline',d.en));header.append(sub,el('h2','',d.label));
    if(d.id==='reward')header.append(el('span','ct-method-tag','DAgger / BC'));
    card.append(header,el('p','ct-sectionhint',d.hint));const fields=el('div','ct-fields');
    for(const f of list){fields.append(createField(f,d.id));used.add(f.key)}
    card.append(fields);
    if(d.id==='reward'){const example=el('div','ct-examples');example.append(el('span','','示例 · 归一化前的轨迹损失倍率：'));for(const [key,title] of [['hold','曾拿稳'],['end','曾拿稳且末帧稳持'],['turn','三项全中']]){example.append(document.createTextNode(' '+title+' '));const val=el('b');val.dataset.ex=key;example.append(val)}card.append(example)}
    q('groups').append(card)
   }
 }
 function getParents(){return data?.parents||[]}
 function sourceOptions(){
   const before=q('source').value;const arr=getParents();
   q('sourceCard').hidden=!!data.run||arr.length===0;
   q('source').replaceChildren(...arr.filter(i=>i.id).map(i=>opt(i.id,i.label||i.id)));
   q('source').value=arr.some(i=>i.id===before)?before:defaultParent(arr);
   setText('sourceHint','新实验从选定权重开始，轮次从 1 计。不会覆盖来源实验和检查点。');
 }
 function preview(){
   if(!data)return;
   const isNew=pane==='new',r=data.run,baseline=fromData(),values=actual(),changes=[],errors=[];
   for(const [k,{input,range,f,row}] of controls){
    const n=values[k],bad=!Number.isFinite(n)||n<f.min||n>f.max||(f.type==='int'&&(!Number.isInteger(n)||input.validity.stepMismatch));
    // Check browser-native step alignment only for integer fields.
    input.setAttribute('aria-invalid',String(bad));row.classList.toggle('ct-invalid',bad);if(bad)errors.push(fieldLabels[k]||f.label||k);
    if(range)range.style.setProperty('--ct-fill', (Math.max(0,Math.min(1,(n-f.min)/(f.max-f.min)))||0)*100+'%');
    if(!eq(n,baseline[k])&&!bad)changes.push({label:fieldLabels[k]||f.label||k,from:baseline[k],to:n});
   }
   const name=q('name').value.trim();const badName=isNew&&!/^contact_hold90_ui_[a-zA-Z0-9][a-zA-Z0-9_-]{0,53}$/.test(name);
   q('name').setAttribute('aria-invalid',String(badName));if(badName)errors.push('实验名须以 contact_hold90_ui_ 开头，加 1–54 位字母、数字、下划线或横线，首位为字母或数字');
   valid=errors.length===0;
   const multipliers={hold:1+values.hold_score,end:1+values.hold_score+values.ending_score,turn:1+values.hold_score+values.ending_score+values.turn_score};
   for(const n of q('groups').querySelectorAll('[data-ex]')){const count=multipliers[n.dataset.ex];n.textContent=(Number.isFinite(count)?nice(Number(count.toFixed(4))):'—')+'×'}
   q('stale').hidden=!conflict;
   setText('rounds',nice(values.rounds));setText('batch',nice(values.batch));setText('budget','每轮执行 '+nice(values.updates)+' 次参数更新。');
   setText('state',isNew?'新建草案':labelStatus(r?.status));q('state').dataset.status=isNew?'new':r?.status||'unknown';
   setText('effect',isNew?('初始权重：'+(q('source').selectedOptions[0]?.textContent||'无可用检查点')+'。新实验从第 1 轮计。'):r?.online?(r.status==='completed'?'该实验已结束，可选其检查点新建实验。':('第 '+nice((r.frozen_round??r.active_round??r.completed_round??0)+1)+' 轮起可以使用新参数；已经开始的轮次配置保持不变。')):(r?.reason||'这个实验使用原训练器，不能在运行中换参数。'));
   const c=q('compare');c.replaceChildren(el('strong','',changes.length?(changes.length+' 项参数有变更'):(isNew?'当前采用引擎默认值':'与当前有效参数一致')));
   if(changes.length){const dl=el('dl','');for(const change of changes){dl.append(el('dt','',change.label),el('dd','',nice(change.from)+' → '+nice(change.to)))}c.append(dl)}
   q('start').hidden=!isNew;q('apply').hidden=isNew||!r?.online;q('resume').hidden=isNew||!r?.online||r?.status==='running';q('fork').hidden=isNew;
   q('restore').hidden=!data.draft?.values;q('loadPending').disabled=posting;q('pending').hidden=!data.pending?.values;
   setText('pendingText',data.pending?.values?'有一份待生效配置，最早第 '+nice(data.pending.earliest_round)+' 轮读取。':'');
   setText('draftInfo',data.draft?.values?'已保存 '+(time(data.draft.updated_at)||''):'尚未保存。草稿不会影响训练。');
   let block='';if(conflict)block='提交版本已更新，请重载参数后再编辑。';else if(errors.length)block='请检查：'+errors.join('、')+'。';
   else if(isNew&&!getParents().length)block='没有可核验的起始模型；可保存参数草稿。';
   else if(isNew&&!data.available)block=data.reason||'当前暂不能启动训练，可以继续调参并保存草稿。';
   else if(!isNew&&!r?.can_apply&&!r?.can_resume)block=r?.reason||'当前不能提交或恢复。';
   else if(!isNew&&!data.available&&r?.status!=='running'&&!r?.can_resume)block=data.reason||'暂不能启动续训。';
   q('reason').hidden=!block;setText('reason',block);
   disabledButton('save',posting||!valid||!asNumList(values)||controls.size===0);
   disabledButton('restore',posting);disabledButton('start',posting||!valid||!getParents().length||data.available!==true);
   disabledButton('apply',posting||conflict||!valid||!r?.can_apply||!Object.keys(values).some(k=>!eq(values[k],proposal(data)[k])));
   disabledButton('resume',posting||conflict||!valid||!r?.can_resume||data.available!==true);disabledButton('fork',posting||!valid);
 }
 function renderJobs(){const jobs=Array.isArray(data.jobs)?data.jobs:[];q('jobBox').hidden=!jobs.length;q('jobs').replaceChildren();for(const j of jobs.slice(0,4)){const row=el('div','ct-job');row.append(el('b','',j.run||'训练任务'),el('small','',labelStatus(j.status)+' · '+(time(j.created_at)||'刚刚')));if(j.reason)row.append(el('small','',j.reason));q('jobs').append(row)}}
 function updateControls(initial){
   q('load').hidden=true;q('content').hidden=false;const stored=editCache.get(target);
   if(initial){
     q('name').value=stored?.name||data.suggested_name||'contact_hold90_ui_experiment';buildGroups();sourceOptions();formRevision=stored?.revision??revision(data);formBasis=stored?.basis||{...proposal(data)};setValues(proposal(data));
     if(stored?.raw)for(const [k,v] of Object.entries(stored.raw)){const t=controls.get(k);if(t){t.input.value=v;if(t.range)t.range.value=v}}
     if(stored?.source&&getParents().some(i=>i.id===stored.source))q('source').value=stored.source;if(formRevision!==revision(data)){const current=actual();conflict=Object.keys(current).some(k=>!eq(current[k],formBasis[k])&&!eq(current[k],proposal(data)[k]));if(!conflict){formRevision=revision(data);formBasis={...proposal(data)};setValues(formBasis)}}else conflict=false;
   }
   renderJobs();preview();
 }
 async function load(initial=false){
   if(!active||destroyed||posting)return;
   const seq=++requestId;reading?.abort();reading=new AbortController();clearTimeout(timer);
   if(initial){q('content').hidden=true;q('load').hidden=false;setText('load','正在读取训练参数…');note('');q('monitor').hidden=true}
   disabledButton('refresh',true);
   try{
    const [d,list]=await Promise.all([request(ENDPOINT+(target?'?run='+encodeURIComponent(target):''),reading.signal),request('/api/runs',reading.signal)]);
    if(seq!==requestId||destroyed)return;if(!Array.isArray(d.schema)||!d.schema.length)throw Error('训练引擎没有返回可编辑的参数');
    const first=initial||!data; const oldParents=JSON.stringify(getParents().map(x=>x.id));
    data=d;runs=list.runs||[];runsOptions();updateControls(first);
    if(!first){
     if(oldParents!==JSON.stringify(getParents().map(x=>x.id)))sourceOptions();
     const revised=revision(d);if(revised!==formRevision){const values=actual(),changed=Object.keys(values).some(k=>!eq(values[k],formBasis[k])),same=Object.keys(values).every(k=>eq(values[k],proposal(d)[k]));if(changed&&!same)conflict=true;else{conflict=false;formRevision=revised;formBasis={...proposal(d)};setValues(formBasis)}}
    }
    preview();
   }catch(e){if(e.name==='AbortError'||seq!==requestId)return;if(!data||initial){q('content').hidden=true;q('load').hidden=false;setText('load','配置未能读取：'+e.message+'。可点「刷新状态」重试。')}else{data.available=false;if(data.run){data.run.can_apply=false;data.run.can_resume=false}preview();note('状态未能更新：'+e.message,true)}}
   finally{if(seq===requestId&&!destroyed){disabledButton('refresh',false);if(active)timer=setTimeout(()=>load(false),18000)}}
 }
 function tab(name,id){
   if(posting||destroyed)return;
   if(name==='existing'&&!id){id=q('run').value||preferred||runs[0]?.id;if(!id){if(data)note('尚未发现可以配置的 90 毫米实验。');return}}
   const next=name==='new'?'':id;if(data&&name===pane&&target===next)return;
   remember();pane=name;target=next||'';data=null;controls.clear();
   q('targetLabel').textContent=name==='new'?'新实验名称':'已有实验';q('name').hidden=name!=='new';q('run').hidden=name==='new';if(id)q('run').value=id;
   q('existing').setAttribute('aria-pressed',String(name==='existing'));q('new').setAttribute('aria-pressed',String(name==='new'));load(true);
 }
 async function send(action){
   if(!data||posting||destroyed||q(action==='draft'?'save':action).disabled)return;
   const body={run:action==='start'?q('name').value.trim():target||null,values:actual()};
   if((action==='start'||(action==='draft'&&pane==='new'))&&q('source').value)body.parent=q('source').value;
   if(action==='draft'&&pane==='new')body.name=q('name').value.trim();
   if(['apply','resume'].includes(action)&&formRevision!=null)body.expected_revision=formRevision;
   posting=true;clearTimeout(timer);requestId++;reading?.abort();note('正在提交…');q('monitor').hidden=true;preview();
   for(const key of ['new','existing','refresh','run','name','source'])q(key).disabled=true;
   for(const t of controls.values()){t.input.disabled=true;if(t.range)t.range.disabled=true}
   try{
     const result=await request(ENDPOINT+'/'+action,listeners.signal,body);if(destroyed)return;
     note(result.message||({draft:'草稿已保存，训练未变更。',apply:'配置已提交；以训练器确认生效的轮次为准。',resume:'续训已提交。',start:'新实验已提交。'}[action]));
     if(['start','resume'].includes(action)){onStarted?.(result.run||body.run);q('monitor').hidden=false}
   }catch(e){if(e.name!=='AbortError')note(e.message,true)}
   finally{if(!destroyed){posting=false;for(const key of ['new','existing','refresh','run','name','source'])q(key).disabled=false;for(const t of controls.values()){t.input.disabled=!!data.run&&t.f.mutable===false;if(t.range)t.range.disabled=t.input.disabled}preview();await load(false)}}
 }
 function copy(){if(!data||posting)return;const values=actual(),old=target,arr=getParents();let parent=arr.filter(i=>i.run===old).sort((a,b)=>b.round-a.round)[0]?.id||'';tab('new','');editCache.set('',{raw:values,source:parent})}
 on(q('new'),'click',()=>tab('new',''));on(q('existing'),'click',()=>tab('existing',null));on(q('run'),'change',()=>tab('existing',q('run').value));
 on(q('refresh'),'click',()=>load(!data));on(q('name'),'input',preview);on(q('source'),'change',preview);on(q('fork'),'click',copy);
 on(q('reloadDraft'),'click',()=>{conflict=false;formRevision=revision(data);formBasis={...proposal(data)};setValues(formBasis);note('已重载服务端最新配置。')});
 on(q('restore'),'click',()=>{if(pane==='new'){if(data.draft?.name)q('name').value=data.draft.name;if(data.draft?.parent&&getParents().some(i=>i.id===data.draft.parent))q('source').value=data.draft.parent}setValues(data.draft?.values);note('草稿已载入表单，训练尚未变更。')});
 on(q('loadPending'),'click',()=>{setValues(data.pending?.values);note('已显示等待生效的值。')});
 for(const action of ['draft','apply','start','resume'])on(q(action==='draft'?'save':action),'click',()=>send(action));
 load(true);
 return {setActive(v){const before=active;active=!!v;clearTimeout(timer);if(active&&!before)load(!data);else if(!active){reading?.abort()}},setRun(id){if(typeof id!=='string'||!id)return;preferred=id;if(!posting)tab('existing',id)},destroy(){destroyed=true;active=false;clearTimeout(timer);reading?.abort();fieldListeners?.abort();listeners.abort();container.replaceChildren()}}
}
