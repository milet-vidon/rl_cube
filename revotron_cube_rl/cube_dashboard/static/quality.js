const el = id => document.getElementById(id);
const number = (v, digits = 1) => Number.isFinite(v)
  ? v.toLocaleString('zh-CN', {maximumFractionDigits: digits}) : '—';
const percent = value => number(value * 100) + '%';
const rates = [
  {key: 'held', label: '曾拿稳', color: '#087d73'},
  {key: 'end_held', label: '结束仍稳持', color: '#b36b19'},
  {key: 'endpoint_held', label: '终点仍稳持', color: '#6458aa'}
];
const duration = {key: 'mean_hold_s', label: '平均最长连续稳持', color: '#28788a'};
let runId = null;
let report = null;
let selected = null;

function svgNode(tag, attributes, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([attr, value]) => node.setAttribute(attr, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function valueFor(row, key) {
  if (!Number.isFinite(row[key])) return null;
  if (key === duration.key) return row[key];
  return row.envs > 0 ? row[key] / row.envs : null;
}

function detailText(row, metric) {
  if (!Number.isFinite(valueFor(row, metric.key))) return '缺少数据';
  if (metric.key === duration.key) return number(row.mean_hold_s, 2) + ' 秒';
  return number(row[metric.key]) + ' / ' + number(row.envs) + '（' + percent(row[metric.key] / row.envs) + '）';
}

function roundChart(container, rows, metrics, current) {
  const width = Math.max(250, container.clientWidth || 640);
  const height = metrics === rates ? 222 : 158;
  const left = 54, right = width - 23, top = 14, bottom = height - 34;
  const seconds = metrics[0].key === duration.key;
  const values = rows.flatMap(r => metrics.map(m => valueFor(r, m.key))).filter(Number.isFinite);
  if (!values.length) {
    container.textContent = '暂无可绘制的记录';
    return;
  }
  const maxY = seconds ? Math.max(4, 4 * Math.ceil(Math.max(...values) / 4)) : 1;
  const first = rows[0].round, last = rows.at(-1).round;
  const x = r => first === last ? (left + right) / 2 : left + (right - left) * (r - first) / (last - first);
  const y = v => bottom - (bottom - top) * v / maxY;
  const svg = svgNode('svg', {viewBox: '0 0 ' + width + ' ' + height, role: 'img',
    'aria-label': seconds ? '各轮平均最长连续稳持时间，单位秒' : '各轮闭环验证达标率'});
  for (let i = 0; i <= 4; i++) {
    const v = maxY * i / 4;
    svg.append(svgNode('line', {x1: left, y1: y(v), x2: right, y2: y(v), class: 'quality-grid'}));
    svg.append(svgNode('text', {x: left - 8, y: y(v) + 4, 'text-anchor': 'end'}, seconds ? number(v) + ' s' : percent(v)));
  }
  const tickCount = Math.min(Math.max(2, Math.floor((right - left) / 54)), last - first + 1);
  for (let i = 0; i < tickCount; i++) {
    const r = tickCount === 1 ? first : Math.round(first + i * (last - first) / (tickCount - 1));
    svg.append(svgNode('text', {x: x(r), y: bottom + 23, 'text-anchor': 'middle'}, 'R' + r));
  }
  svg.append(svgNode('line', {x1: x(current), y1: top, x2: x(current), y2: bottom, class: 'quality-guide'}));
  for (const metric of metrics) {
    let path = '', prev = null;
    const points = [];
    for (const row of rows) {
      const value = valueFor(row, metric.key);
      if (value === null) { prev = null; continue; }
      const px = x(row.round), py = y(value);
      path += (prev === row.round - 1 ? ' L' : ' M') + px.toFixed(2) + ' ' + py.toFixed(2);
      prev = row.round;
      const circle = svgNode('circle', {cx: px, cy: py, r: row.round === current ? 4 : 2.7,
        fill: metric.color, stroke: 'white', 'stroke-width': 1, 'data-round': row.round,
        'data-metric': metric.key, 'data-value': row[metric.key]});
      circle.append(svgNode('title', {}, '第 ' + row.round + ' 轮 · ' + metric.label + ' ' + detailText(row, metric)));
      points.push(circle);
    }
    svg.append(svgNode('path', {d: path, stroke: metric.color, fill: 'none', 'stroke-width': 2.3, 'stroke-linejoin': 'round'}), ...points);
  }
  container.replaceChildren(svg);
}

function updateMenu(select, rows, current) {
  const options = rows.toReversed ? rows.toReversed() : [...rows].reverse();
  if (select.options.length !== options.length ||
      options.some((row, i) => select.options[i].value !== String(row.round))) {
    select.replaceChildren(...options.map(r => new Option('第 ' + r.round + ' 轮', String(r.round))));
  }
  if (current) select.value = String(current.round);
  select.disabled = !current;
}

function draw() {
  const rows = (report?.rounds || []).filter(row => row.detail_status !== 'missing_summary');
  const current = rows.find(r => r.round === selected) || rows.at(-1);
  updateMenu(el('trend-round'), rows, current);
  const empty = report ? '暂无已完成的闭环验证' : '正在读取验证结果…';
  el('trend-empty').textContent = empty;
  el('trend-empty').hidden = !!current;
  el('trend-body').hidden = !current;
  if (!current) {
    el('quality-rates').replaceChildren();
    el('quality-duration').replaceChildren();
    return;
  }
  const cards = rates.map(m => {
    const group = document.createElement('div');
    const name = document.createElement('dt');
    const value = document.createElement('dd');
    name.textContent = m.label;
    value.textContent = detailText(current, m);
    group.append(name, value);
    return group;
  });
  el('quality-latest').replaceChildren(...cards);
  el('duration-latest').textContent = '第 ' + current.round + ' 轮 · ' + detailText(current, duration);
  roundChart(el('quality-rates'), rows, rates, current.round);
  roundChart(el('quality-duration'), rows, [duration], current.round);
}

function pick(round) {
  selected = Number(round);
  draw();
}

for (const metric of rates) {
  const key = document.createElement('span'), swatch = document.createElement('i');
  swatch.style.background = metric.color;
  key.append(swatch, document.createTextNode(metric.label));
  el('quality-legend').append(key);
}
el('trend-round').addEventListener('change', e => pick(e.target.value));
for (const id of ['quality-rates','quality-duration']) {
  el(id).addEventListener('click', e => {
    const circle = e.target.closest('circle[data-round]');
    if (circle) pick(circle.dataset.round);
  });
}
export function showQuality(page, id, payload = null) {
  if (id !== runId) { selected = null; report = null; runId = id; }
  if (payload) report = payload;
  draw();
}

const fitMetric = [
  {key:'fit_mse', label:'选中模型', color:'#087d73'},
  {key:'fit_last_mse', label:'轮末模型', color:'#8c6cac', dashed:true}
];
const passMetric = [
  {key:'whole_turn', label:'整颗转动', color:'#587594'},
  {key:'end_held', label:'结束仍稳持', color:'#af7323', dashed:true},
  {key:'joint_pass', label:'三项同达', color:'#097b6f'}
];
const weightMetric = [{key:'parameter_delta_pct', label:'相对变化量', color:'#5867a2'}];
const units = {rates: '验证达标率', loss:'留出 MSE', weights:'父子模型权重变化'};
const metricSets = {rates: passMetric, loss: fitMetric, weights: weightMetric};
let cvRun = null, cvReport = null, cvPicked = null;
const finite = Number.isFinite;
const mseText = n => n === 0 ? '0' : (+n.toPrecision(3)).toString();
const signed = (n, digits=1) => (n>0?'+':'') + number(n,digits);
function cvValue(row,key,kind) {
  const v = row?.[key];
  if (!finite(v) || v<0) return null;
  if (kind==='rates') return row.envs>0 && v<=row.envs ? 100*v/row.envs : null;
  return v;
}
function cvFmt(n,kind) { return kind==='loss' ? mseText(n) : number(n,kind==='weights'?2:1)+'%'; }
function cvText(row,key,kind) {
  const v=cvValue(row,key,kind);
  return !finite(v)?'—':kind==='rates'?cvFmt(v,kind)+' ('+row[key]+'/'+row.envs+')':cvFmt(v,kind);
}
function cvChart(kind,rows,chosen) {
  const plot=el('cv-'+kind), readout=el('cv-'+kind+'-values'), metrics=metricSets[kind];
  readout.replaceChildren(...metrics.map(m=>{
    const line=document.createElement('div'), label=document.createElement('span'), dot=document.createElement('i'), value=document.createElement('b');
    dot.style.borderColor=m.color;
    if(m.dashed)dot.style.borderTopStyle='dashed';
    label.append(dot,document.createTextNode(m.label));value.textContent=cvText(chosen,m.key,kind);
    line.append(label,value);return line;
  }));
  const values=rows.flatMap(r=>metrics.map(m=>cvValue(r,m.key,kind))).filter(finite);
  if(!values.length){
    const blank=document.createElement('p');
    blank.className='cv-blank';
    blank.textContent=kind==='rates'?'尚无已完成的学生闭环验证':kind==='weights'?
      (rows.some(r=>r.parameter_delta_status==='pending')?'正在核对已有模型的权重…':'暂无可比的父子模型权重'):'尚无已完成的拟合记录';
    plot.replaceChildren(blank);return;
  }
  const w=Math.max(210,plot.clientWidth||320), h=156, l=kind==='loss'?60:47, right=w-17, top=16, bottom=h-28;
  const maxV=Math.max(...values), max=kind==='rates'?100:(maxV>0?maxV*1.06:1);
  const first=rows[0].round,last=rows.at(-1).round, x=n=>last===first?(right+l)/2:l+(right-l)*(n-first)/(last-first), y=v=>bottom-v/max*(bottom-top);
  const svg=svgNode('svg',{viewBox:'0 0 '+w+' '+h,role:'img','aria-label':units[kind]+'，第 '+first+' 至 '+last+' 轮'});
  for(let i=0;i<3;i++){
    const v=i*max/2;
    svg.append(svgNode('line',{x1:l,x2:right,y1:y(v),y2:y(v),class:'quality-grid'}),svgNode('text',{x:l-6,y:y(v)+3,'text-anchor':'end'},cvFmt(v,kind)));
  }
  const ticks=Math.min(last-first+1,Math.max(2,Math.floor((right-l)/52)));
  for(let i=0;i<ticks;i++){
    const n=ticks===1?first:Math.round(first+(last-first)*i/(ticks-1));
    svg.append(svgNode('text',{x:x(n),y:bottom+19,'text-anchor':'middle'},'R'+n));
  }
  svg.append(svgNode('line',{x1:x(chosen.round),x2:x(chosen.round),y1:top,y2:bottom,class:'quality-guide'}));
  for(const m of metrics){
    let prev=null, d='';const dots=[];
    for(const row of rows){
      const v=cvValue(row,m.key,kind);
      if(!finite(v)){prev=null;continue;}
      d+=(prev===row.round-1?' L':' M')+x(row.round).toFixed(2)+' '+y(v).toFixed(2);prev=row.round;
      const dot=svgNode('circle',{cx:x(row.round),cy:y(v),r:row.round===chosen.round?4.2:2.5,fill:m.color,'data-round':row.round,stroke:'#fff','stroke-width':0.8});
      dot.append(svgNode('title',{},'R'+row.round+' · '+m.label+' '+cvText(row,m.key,kind)));dots.push(dot);
    }
    const attrs={d,stroke:m.color,fill:'none','stroke-width':2};
    if(m.dashed) attrs['stroke-dasharray']='4 3';
    svg.append(svgNode('path',attrs),...dots);
  }
  plot.replaceChildren(svg);
}
function comparisonCard(label, delta, foot){
  const d=document.createElement('div'),k=document.createElement('span'),v=document.createElement('b'),p=document.createElement('small');
  k.textContent=label;v.textContent=delta;p.textContent=foot;d.append(k,v,p);return d;
}
function cvCompare(rows, mode){
  const cards=el('cv-comparison'),stamp=el('cv-window'),advice=el('cv-advice');
  const last=rows.filter(r=>r.detail_status!=='missing_summary').at(-1)?.round;
  const recent=rows.filter(r=>r.round>last-10&&r.round<=last);
  const ready=recent.length===10&&recent.every((r,i)=>r.round===last-9+i&&r.detail_status==='verified'&&r.envs===recent[0].envs);
  stamp.textContent=ready?'R'+(last-9)+'–'+(last-5)+' 对比 R'+(last-4)+'–'+last:'需连续 10 轮已核对的同规模验证';
  cards.replaceChildren();
  if(!ready){
    advice.textContent='暂不能比较两个完整窗口。可先看逐轮曲线，缺少记录的轮次不会用 0 填充。';return;
  }
  const mean=a=>a.reduce((sum,v)=>sum+v,0)/a.length;
  const signals=[];
  for (const [key,kind,label] of [['joint_pass','rates','三项同达率'],['end_held','rates','结束仍稳持率'],['fit_mse','loss','选中模型 MSE'],['parameter_delta_pct','weights','权重相对变化量']]){
    const series=recent.map(r=>cvValue(r,key,kind));
    if(!series.every(finite)){cards.append(comparisonCard(label,'—',key==='parameter_delta_pct'?'部分轮次无可核对权重':'部分轮次缺少记录'));continue;}
    const prev=mean(series.slice(0,5)),tail=series.slice(-5),now=mean(tail),delta=now-prev;
    const change=kind==='loss'?(prev>0?signed(100*delta/prev)+'%':'—'):signed(delta,kind==='weights'?2:1)+' 个百分点';
    const foot=cvFmt(prev,kind)+' → '+cvFmt(now,kind)+'；近 5 轮 '+cvFmt(Math.min(...tail),kind)+'–'+cvFmt(Math.max(...tail),kind);
    cards.append(comparisonCard(label,change,foot));signals.push({key,now,prev,delta});
  }
  const joint=signals.find(i=>i.key==='joint_pass');
  const start=joint?'三项同达率的近 5 轮均值'+(Math.abs(joint.delta)<1e-8?'与前 5 轮持平':joint.delta>0?'高于前 5 轮':'低于前 5 轮')+'。':'';
  const recommendation=mode==='legacy_bc'?'当前是 DAgger 行为克隆。若损失持续下降而达标率停滞，可从同一检查点另开对照实验，只改示范损失权重，以相同种子复验。':mode==='weighted_bc'?'若拟合改善而达标率停滞，可固定起始模型和种子，只改示范损失权重做对照。':'若达标率停滞，先固定起始模型和种子，再对比目标权重或示范覆盖。';
  const latest=recent.at(-1);
  const comparable=rows.filter(r=>r.round<=last&&r.detail_status==='verified'&&r.envs===latest.envs&&finite(r.joint_pass));
  const best=comparable.reduce((a,r)=>!a||r.joint_pass>a.joint_pass?r:a,null);
  const peak=best&&finite(latest.joint_pass)?'最新 R'+last+' 为 '+latest.joint_pass+'/'+latest.envs+
    '，同规模验证中首次达到最高值的是 R'+best.round+'（'+best.joint_pass+'/'+best.envs+'）。':'';
  advice.textContent=start+peak+'单看这些窗口不能证明已经收敛或奖励不合理。'+recommendation;
}
function drawConvergence(){
  const board=el('convergence');
  if(!board||!board.offsetWidth)return;
  const rows=(cvReport?.rounds||[]).filter(r=>Number.isInteger(r.round)&&r.round>0);
  const useful=rows.some(r=>finite(r.fit_mse)||finite(r.envs)||finite(r.parameter_delta_pct));
  el('cv-empty').hidden=useful;el('cv-body').hidden=!useful;
  el('cv-empty').textContent=cvReport?'本实验还没有可比较的拟合或独立验证记录。':'正在读取跨轮记录…';
  const select=el('cv-round');
  if(!useful){select.replaceChildren();select.disabled=true;return;}
  const end=rows.at(-1).round, n=Number(el('cv-range').value);
  const shown=n?rows.filter(r=>r.round>end-n):rows;
  const chosen=shown.find(r=>r.round===cvPicked)||shown.filter(r=>r.detail_status==='verified').at(-1)||shown.filter(r=>finite(r.fit_mse)||finite(r.envs)).at(-1)||shown.at(-1);
  updateMenu(select,shown,chosen);
  for(const kind of Object.keys(metricSets))cvChart(kind,shown,chosen);
  cvCompare(rows,cvReport.objective_mode);
}
export function showConvergence(page,id,payload=null){
  const board=el('convergence'),anchor=page==='trends'?el('trend-empty'):document.querySelector('#train-page .chart-layout');
  if(board.nextElementSibling!==anchor)anchor.before(board);
  if(id!==cvRun){cvRun=id;cvPicked=null;cvReport=null;el('cv-range').value='0';}
  if(payload)cvReport=payload;
  drawConvergence();
}
el('cv-range').addEventListener('change',drawConvergence);
el('cv-round').addEventListener('change',e=>{cvPicked=Number(e.target.value);drawConvergence();});
el('convergence').addEventListener('click',e=>{
  const mark=e.target.closest('circle[data-round]');
  if(mark){cvPicked=Number(mark.dataset.round);drawConvergence();}
});
window.addEventListener('resize',drawConvergence);
