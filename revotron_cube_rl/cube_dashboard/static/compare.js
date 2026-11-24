// Paired student validation. Read only; targets and references always belong to the selected run.
const field = id => document.getElementById('cmp-'+id);
const labels = {lost:'退步', gained:'新增通过', both_fail:'两轮均未通过', both_pass:'两轮均通过'};
const order = ['gained','lost','both_pass','both_fail'];
const rules = {
  joint_pass: '同一回合达到腕部终点、结束仍稳持，且曾完成整颗转动。',
  end_held: '验证回合结束时，左手仍稳稳拿住魔方。',
  endpoint_held: '验证回合在腕部终点结束，且此时仍稳持。',
  whole_turn: '验证期间曾在稳持条件下达到整颗转动要求。'
};
const percent = (n,d) => (100*n/d).toLocaleString('zh-CN',{maximumFractionDigits:1})+'%';
const secs = n => Number.isFinite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:1}):'—';
let run=null, base=null, target=null, payload=null, rounds=[], manual=false, onChange=()=>{};

export function onPairChange(fn){onChange=fn}
function hideData(message){
  payload=null;
  field('result').hidden=field('detail').hidden=true;
  field('empty').textContent=message;field('empty').hidden=false;
  field('coverage').textContent='';field('kpis').replaceChildren();field('rows').replaceChildren();
}
function options(id,rows,current){
  const select=field(id), list=[...rows].reverse();
  if(list.length!==select.options.length||list.some((r,i)=>String(r.round)!==select.options[i].value)){
    select.replaceChildren(...list.map(r=>new Option('第 '+r.round+' 轮',String(r.round))));
  }
  if(current!==null)select.value=String(current);
  select.disabled=list.length<1;
}
export function setCompareRun(id){
  if(run===id)return;
  run=id;base=target=null;rounds=[];manual=false;
  options('base',[],null);options('target',[],null);
  field('metric').value='joint_pass';field('metric').disabled=true;
  field('filter').value='changed';field('seed').value='';
  field('auto').hidden=true;field('default').textContent='自动选轮';
  hideData('正在读取已验收的验证轮次…');
}
function bestOther(candidates,number){
  const prior=candidates.filter(r=>r.round<number), pool=prior.length?prior:candidates.filter(r=>r.round!==number);
  return pool.reduce((best,r)=>!best||r.joint_pass/r.envs>best.joint_pass/best.envs?r:best,null)?.round??null;
}
export function comparisonPair(id,all){
  setCompareRun(id);
  rounds=all.filter(r=>r.detail_status==='verified'&&r.envs>0&&Number.isInteger(r.joint_pass)&&Number.isInteger(r.round)).sort((a,b)=>a.round-b.round);
  const known=r=>rounds.some(it=>it.round===r);
  const oldBase=base,oldTarget=target;
  if(!manual||!known(target)){target=rounds.at(-1)?.round??null}
  if(!manual||!known(base)||base===target)base=bestOther(rounds,target);
  options('base',rounds.filter(r=>r.round!==target),base);
  options('target',rounds,target);
  field('auto').hidden=!manual;
  field('default').textContent=manual?'手动选轮':'最新验证轮 对比 此前三项同达率最高轮';
  field('metric').disabled=rounds.length<2;
  if(rounds.length<2){
    hideData(rounds.length===1?'只有第 '+target+' 轮的逐回合验证已验收。至少需要两轮，才能判断哪些回合有进退。':'此实验还没有通过核对的逐回合验证，暂不能比较模型。');
    return null;
  }
  if(oldBase!==base||oldTarget!==target)hideData('正在对齐两轮验证种子…');
  return {base,target};
}
export function compareFailure(){hideData('读取模型对照失败，请刷新重试。')}
export function showComparison(d){
  if(d?.id!==run||d.base_round!==base||d.target_round!==target)throw Error('模型对照与选择的轮次不匹配');
  if(d.status!=='ready'){
    const why={no_overlap:'两轮没有共同验证种子，无法进行配对比较。',protocol_mismatch:'两轮的评测配置不同，不能把这次差异算作模型进退。请换两轮对照。',protocol_unknown:'证据不足以确认两轮使用相同的评测配置，请换两轮对照。'};
    hideData(why[d.status]||'所选轮次的完整逐回合证据未通过核对，暂不显示比较。');
    if(d.status==='protocol_mismatch'||d.status==='protocol_unknown')field('coverage').textContent='共有 '+d.matched+' 个种子 · '+(d.status==='protocol_mismatch'?'配置有差异':'配置缺少记录')+
      ((d.protocol_fields||[]).length?'：'+d.protocol_fields.join('、'):'。');
    if(d.status==='no_overlap')field('coverage').textContent='参考轮 '+d.base_envs+' 个种子，对比轮 '+d.target_envs+' 个种子。';
    return;
  }
  if(!Array.isArray(d.rows)||d.rows.length!==d.matched||!(d.matched>0)||!d.metrics)throw Error('无有效配对结果');
  payload=d;field('empty').hidden=true;field('result').hidden=field('detail').hidden=false;
  field('coverage').textContent='已配对 '+d.matched+' 个相同种子 · 参考轮共 '+d.base_envs+' 个，对比轮共 '+d.target_envs+' 个'+
    (d.base_only||d.target_only?'；仅参考轮 '+d.base_only+' 个、仅对比轮 '+d.target_only+' 个，均不计入对照。':'，全部配对。');
  renderStats();
}
function renderStats(){
  if(!payload)return;
  const key=field('metric').value,m=payload.metrics[key],n=payload.matched;
  const delta=m.target_pass-m.base_pass;
  field('summary').textContent='R'+target+' 对比 R'+base+'：新增 '+m.gained+' 个，退步 '+m.lost+' 个，'+
    (delta>0?'净多 '+delta+' 个达标':delta<0?'净少 '+(-delta)+' 个达标':'达标总数持平')+'。';
  field('rates').textContent='共同种子达标率　R'+base+'：'+m.base_pass+'/'+n+'（'+percent(m.base_pass,n)+'） → R'+target+'：'+m.target_pass+'/'+n+'（'+percent(m.target_pass,n)+'）';
  field('rule').textContent='指标定义：'+rules[key];
  field('kpis').replaceChildren(...order.map(k=>{
    const b=document.createElement('button'),v=document.createElement('strong'),name=document.createElement('span');
    b.type='button';b.dataset.outcome=k;b.dataset.count=String(m[k]);b.setAttribute('aria-controls','cmp-rows');
    name.textContent=labels[k];v.textContent=m[k]+' 个';b.append(name,v);return b;
  }));
  field('base-heading').textContent='参考 R'+base;
  field('target-heading').textContent='对比 R'+target;
  renderTable();
}
function outcome(row,key){
  return row.base[key]?(row.target[key]?'both_pass':'lost'):(row.target[key]?'gained':'both_fail');
}
function badge(cell,text,pass){
  const tag=document.createElement('span');tag.className='cmp-badge '+(pass?'pass':'fail');tag.textContent=text;cell.append(tag);
}
function unmet(lane,key){
  if(lane[key]||key!=='joint_pass')return '';
  const missed=[];
  if(!lane.end_held)missed.push('结束未稳持');
  if(lane.reason!=='teacher_wrist_endpoint')missed.push('未在腕端结束');
  if(!lane.whole_turn)missed.push('转动未达标');
  return missed.join('、');
}
function renderTable(){
  if(!payload)return;
  const key=field('metric').value, group=field('filter').value, query=field('seed').value.trim();
  const ranks=['lost','gained','both_fail','both_pass'];
  const rows=payload.rows.map(r=>[r,outcome(r,key)]).filter(([r,k])=>
    (group==='all'||group===k||group==='changed'&&['lost','gained'].includes(k))&&String(r.seed).includes(query))
    .sort(([a,k],[b,l])=>ranks.indexOf(k)-ranks.indexOf(l)||a.seed-b.seed);
  for(const b of field('kpis').querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.outcome===group));
  field('tally').textContent='显示 '+rows.length+' / '+payload.matched+' 个配对种子';
  const body=field('rows');body.replaceChildren();
  if(!rows.length){
    const c=body.insertRow().insertCell();c.colSpan=5;c.className='emptyrow';c.textContent='当前筛选条件下没有验证回合。';return;
  }
  for(const [r,k] of rows){
    const tr=body.insertRow();tr.dataset.seed=r.seed;tr.dataset.class=k;
    tr.insertCell().textContent=String(r.seed);const result=tr.insertCell();result.textContent=labels[k];result.className='cmp-direction '+k;
    for(const lane of [r.base,r.target]){
      const c=tr.insertCell();badge(c,lane[key]?'通过':'未通过',lane[key]);
      const why=unmet(lane,key);if(why){const hint=document.createElement('small');hint.textContent=why;c.append(hint)}
    }
    tr.insertCell().textContent=secs(r.base.hold_s)+' → '+secs(r.target.hold_s)+' 秒';
  }
}
for(const key of ['base','target'])field(key).addEventListener('change',()=>{
  if(key==='base')base=Number(field('base').value);
  else{target=Number(field('target').value);if(base===target)base=bestOther(rounds,target)}
  manual=true;hideData('正在对齐两轮验证种子…');onChange();
});
field('auto').addEventListener('click',()=>{manual=false;hideData('正在读取默认模型对照…');onChange()});
field('metric').addEventListener('change',renderStats);
field('filter').addEventListener('change',renderTable);
field('seed').addEventListener('input',renderTable);
field('kpis').addEventListener('click',e=>{
  const b=e.target.closest('button[data-outcome]');
  if(b){field('filter').value=b.dataset.outcome;renderTable()}
});
