const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>Number.isFinite(v)?v.toLocaleString('zh-CN'):'—';
const fixed=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
const time=v=>v?new Date(v).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'—';
const statuses={running:'运行中',paused:'进程已暂停',completed:'已结束',failed:'运行失败',interrupted:'进程已退出',blocked:'未满足启动条件',not_started:'尚未启动',qualified_15deg_state_specialist:'独立验收通过',qualified_90deg_state_specialist:'独立验收通过',independent_test_not_met:'独立验收未通过'};
const roles={smoke:'短测',train:'训练采样 · 更新前',validate:'冻结模型验证'};
const yes=v=>v===true||v===1?'是':v===false||v===0?'否':'—';
const degree=(value,digits=0)=>Number.isFinite(value)?(value>0?'+':'')+value.toFixed(digits)+'°':'—';
const fraction=(count,total)=>Number.isFinite(count)&&Number.isFinite(total)?num(count)+' / '+num(total):'—';
const cells=values=>'<tr>'+values.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>';
const emptyRow=(message,columns)=>'<tr><td class="emptyrow" colspan="'+columns+'">'+esc(message)+'</td></tr>';
const firstKinds={none:'无触层',index_only:'仅食指',other_only:'仅其他手指',mixed:'食指及其他手指',aggregate_only:'汇总冲量',unattributed:'归属未定'};
const runLabel=r=>({bimanual_training:'[双手对握 PPO] ',sweep:'[冻结选点] ',approach_anchor:'[冻结接近诊断] ',pad_orientation:'[冻结指垫朝向] ',pad_preshape:'[冻结食指预形] ',hover_cd:'[C/D 对照] ',cartesian_probe:'[Placo 零增量验证] ',cartesian_training:'[末端增量 PPO] ',cartesian_contact:'[接触与 +2° PPO] ',cartesian_contact_to15:'[接触策略续训 +15°] ',diagnostic:'[短测] '}[r.kind]||'[训练] ')+r.id;

export function mountRemote(root){
 root.innerHTML=`<div class="titlebar"><div><div class="eyebrow"><span></span> REMOTE TRAINING</div><h1>远端训练</h1><p>查看远端状态与结果，实验和模型保存在原机器。</p></div><div class="controls"><label for="rm-run">远端实验</label><select id="rm-run" aria-label="选择远端训练实验"></select><button type="button" id="rm-refresh">刷新</button></div></div>
 <div class="rm-source"><div><b id="rm-host">连接远端中…</b><span id="rm-source-note">独立数据来源 · 本机历史实验保留</span></div><span class="pill" id="rm-link">连接中</span></div>
 <p class="alert" id="rm-error" role="status" hidden></p>
 <div id="rm-body" hidden>
 <div class="stats">
  <article class="stat"><div><span id="rm-round-label">当前迭代</span><span class="pill" id="rm-status">—</span></div><strong id="rm-round">—</strong><small id="rm-stage">—</small></article>
  <article class="stat"><div><span>当前采样</span></div><strong id="rm-step">—</strong><small id="rm-progress">—</small></article>
  <article class="stat"><div><span id="rm-result-label">本批结果</span></div><strong id="rm-success">—</strong><small id="rm-unsafe">尚无完整回合报告</small></article>
  <article class="stat"><div><span>远端 GPU</span></div><strong id="rm-gpu">—</strong><small id="rm-memory">—</small></article>
 </div>
 <article class="panel rm-info"><div class="panel-head"><div><div class="section-tag">TASK & POLICY</div><h2 id="rm-algorithm">—</h2><p id="rm-target"></p></div><span class="capsule" id="rm-model-status"></span></div>
 <dl class="rm-facts" id="rm-facts"></dl><p class="table-note" id="rm-stamp"></p><p class="table-note rm-path" id="rm-path"></p></article>
 <div id="rm-sweep" hidden>
 <article class="panel"><div class="panel-head"><div><h2>落点对照</h2><p>接触与驱动按 C 组固定种子统计；安全性按全部配对种子统计。</p></div></div><div class="table-scroll"><table><thead><tr><th>落点方位</th><th>安全首次食指触层</th><th>安全有效驱动</th><th>同时首触且驱动 ≥2°</th><th>最小间隙中位数</th><th>全组不安全</th><th>新增不安全种子</th></tr></thead><tbody id="rm-offsets"></tbody></table></div></article>
 <article class="panel"><div class="panel-head"><div><h2>后续训练</h2><p id="rm-next-action"></p></div></div><div id="rm-next-stages"></div></article>
 <article class="panel" id="rm-seed-panel"><div class="panel-head"><h2>逐种子结果</h2><div class="controls"><label for="rm-angle">落点</label><select id="rm-angle"><option value="0">C 原点</option><option value="7" selected>+7°</option><option value="-7">−7°</option></select><label for="rm-cohort">种子</label><select id="rm-cohort"><option value="fixed">固定 11 个</option><option value="all">全部 16 个</option></select></div></div><div class="table-scroll"><table><thead><tr><th>种子</th><th>首次触层</th><th>安全食指首触</th><th>安全有效驱动</th><th>驱动峰值</th><th>回合末层角</th><th>不安全</th><th>结束原因</th></tr></thead><tbody id="rm-seeds"></tbody></table></div></article></div>
 <article class="panel" id="rm-anchor-panel" hidden><div class="panel-head"><div><h2 id="rm-anchor-title"></h2><p id="rm-anchor-note"></p></div></div><div class="table-scroll"><table><thead id="rm-anchor-head"></thead><tbody id="rm-anchor-rows"></tbody></table></div></article>
 <article class="panel" id="rm-contact-next" hidden><div class="panel-head"><div><h2>启动接触训练</h2><p>在 Nexus 项目目录运行下列命令。接触策略通过配对验证后，同一任务继续训练单层 +15°。</p></div></div>
 <pre>bash run_natural_face90_cartesian_contact.sh long</pre><p class="table-note">启动后查看 <a href="?run=r27_recipe_cartesian_contact_ppo#remote">接触与 +2° 训练</a> · <a href="?run=r27_recipe_cartesian_contact_to15#remote">单层 +15° 续训</a></p></article>
 <article class="panel" id="rm-learning-panel"><div class="panel-head"><div><h2>本次 PPO 更新</h2><p id="rm-update-note">参数更新与物理验收分别记录。</p></div></div><dl class="rm-facts" id="rm-learning"></dl></article>
 <article class="panel" id="rm-episode-panel"><div class="panel-head"><div><h2 id="rm-episodes-title">回合结果</h2><p id="rm-episodes-note"></p></div></div><div class="table-scroll"><table><thead id="rm-episodes-head"></thead><tbody id="rm-episodes"></tbody></table></div></article>
 <article class="panel" id="rm-history-panel"><div class="panel-head"><div><h2>迭代记录</h2><p>训练采样来自更新前策略；验证使用更新后冻结模型。</p></div><span class="capsule" id="rm-history-count"></span></div><div class="table-scroll"><table><thead id="rm-history-head"></thead><tbody id="rm-history"></tbody></table></div></article>
 <article class="panel logs"><div class="panel-head"><div><h2>远端运行日志</h2><p id="rm-log-stamp"></p></div><button type="button" id="rm-log-toggle" aria-expanded="true">收起日志</button></div><pre id="rm-log" tabindex="0"></pre></article>
 </div>`;
 const $=id=>root.querySelector('#'+id),text=(id,v)=>$(id).textContent=v;
 let active=false,busy=false,serial=0,catalogAt=0,controller=null,shownSweep=null;
 const facts=(id,items)=>{$(id).innerHTML=items.map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')};
 function connection(s){
  const c=s.connection||{};
  text('rm-host',(s.remote_host||s.source?.label||'Nexus-5090-1')+(s.source?.host?' · '+s.source.host:''));
  text('rm-link',c.ok?'远端已连接':c.stale?'连接中断 · 缓存数据':'远端未连接');
  $('rm-link').className='pill '+(c.ok?'live':'warn');
  const error=c.error||s.error||'';text('rm-error',error);$('rm-error').hidden=!error;
  return c.ok;
 }
 function renderSeeds(){
  if(!shownSweep)return;
  const rows=shownSweep.angles?.[$('rm-angle').value]?.per_seed||[];
  const fixedSeeds=new Set(shownSweep.fixed_C_no_contact_horizon_seeds||[]);
  const selected=$('rm-cohort').value==='fixed'?rows.filter(r=>fixedSeeds.has(r.seed)):rows;
  $('rm-seeds').innerHTML=selected.map(r=>cells([r.seed,firstKinds[r.first_contact?.kind]||'—',yes(r.safe_qualified_index_first),
   yes(r.safe_qualified_drive),degree(r.peak_drive_response_deg,3),degree(r.final_face_deg,3),yes(r.unsafe),r.end_reason])).join('')||emptyRow('等待该落点的配对统计。',8);
 }
 function renderSweep(s){
  const sweep=s.sweep||{},angles=sweep.angles||{};
  shownSweep=sweep;
  const labels={'0':'C 原点','7':'+7°','-7':'−7°'};
  $('rm-offsets').innerHTML=['0','7','-7'].filter(key=>angles[key]).map(key=>{
   const a=angles[key],fixedGroup=a.fixed_C_no_contact_horizon||{},all=a.all_16||{},gap=a.safe_paired_min_approach_gap_mm?.candidate_median;
   return cells([labels[key],fraction(fixedGroup.safe_qualified_index_first,fixedGroup.episodes),fraction(fixedGroup.safe_qualified_drive,fixedGroup.episodes),
    fraction(a.qualifying_fixed_seeds?.length,fixedGroup.episodes),Number.isFinite(gap)?fixed(gap,2)+' mm':'—',fraction(all.unsafe,all.episodes),
    key==='0'?'基线':Array.isArray(a.new_unsafe_seeds)?a.new_unsafe_seeds.join('、')||'无':'—']);
  }).join('')||emptyRow('两个候选完成后显示与 C 原点的配对结果。',7);
  text('rm-next-action',sweep.decision==='no_candidate'?'先检查食指尚未触层时的接近跟踪，再按同一首触、驱动和安全条件复核。':sweep.decision==='selected'?'已选定落点，可使用新的训练种子启动 +15° PPO。':'等待冻结选点结果。');
  $('rm-next-stages').innerHTML=(sweep.stages||[]).map(stage=>{
   const label=degree(stage.target_deg)+' PPO';
   const title=stage.run_id?'<a href="?run='+encodeURIComponent(stage.run_id)+'#remote">'+esc(label)+'</a>':esc(label);
   return '<div class="rm-next-stage"><strong>'+title+'</strong><span>'+esc(statuses[stage.status]||stage.status)+'</span><p>'+esc(stage.reason)+'</p></div>';
  }).join('');
  $('rm-seed-panel').hidden=!Object.keys(angles).length;
  renderSeeds();
 }
 function comparisonRows(rows,measures){
  const columns=['条件','安全首次食指触层','安全有效驱动','有效驱动 ≥2°',...measures,'全组不安全'];
  $('rm-anchor-head').innerHTML='<tr>'+columns.map(v=>'<th>'+esc(v)+'</th>').join('')+'</tr>';
  return rows.filter(([,panel])=>panel).map(([label,panel,all,...numbers])=>cells([label,
   fraction(panel.safe_qualified_index_first,panel.episodes),fraction(panel.safe_qualified_drive,panel.episodes),
   fraction(panel.safe_drive_2deg,panel.episodes),...numbers,fraction(all?.unsafe,all?.episodes)])).join('')||emptyRow('等待对照结果。',columns.length);
 }
 function render(s){
  const connected=connection(s),p=s.progress||{},m=s.summary,l=s.learning||{},g=s.hardware||{};
  const isSweep=s.kind==='sweep',isAnchor=s.kind==='approach_anchor',isPreshape=s.kind==='pad_preshape',isPad=isPreshape||s.kind==='pad_orientation',isHover=s.kind==='hover_cd',pointTraining=s.settings?.schema==='natural_face90_contact_point_ppo_v1';
  const bimanual=s.kind==='bimanual_training';
  const cartesianProbe=s.kind==='cartesian_probe',contactTraining=s.kind==='cartesian_contact';
  const cartesianTraining=['cartesian_training','cartesian_contact_to15','cartesian_contact'].includes(s.kind);
  const hasComparison=isAnchor||isPad||isHover||cartesianProbe;
  const cart=s.cartesian||{},cartesianFacts=[['策略输入 → 输出',(cart.actor_dim??19)+' 维状态 → '+(cart.action_dim??3)+' 维末端位移增量'],
   ['输出方向','径向朝外、正转切向、法向向内'],['右臂控制器','Placo 逆运动学'],['食指预形',degree(s.settings?.pad_preshape_offset_deg)]];
  if(Number.isFinite(cart.residual_norm_cap_m))cartesianFacts.push(['每步策略增量上限',fixed(cart.residual_norm_cap_m*1000,2)+' mm']);
  const doneSweep=isSweep&&['no_candidate','selected'].includes(s.sweep?.decision);
  $('rm-sweep').hidden=!isSweep;
  $('rm-anchor-panel').hidden=!hasComparison;
  $('rm-contact-next').hidden=!(cartesianProbe&&s.status==='completed'&&s.cartesian_probe?.paired_contact_gate_met===false);
  $('rm-learning-panel').hidden=isSweep||hasComparison;
  $('rm-history-panel').hidden=isSweep||hasComparison;
  $('rm-episode-panel').hidden=doneSweep;
  shownSweep=null;
  $('rm-body').hidden=false;
  text('rm-status',connected?(statuses[s.status]||s.status):'离线缓存');
  $('rm-status').className='pill '+(connected&&s.status==='running'?'live':['failed','interrupted'].includes(s.status)?'warn':'');
  text('rm-round-label',s.kind==='diagnostic'?'短测批次':s.algorithm==='state_ppo'?'当前 PPO 迭代':'当前搜索轮次');
  text('rm-round',num(s.iteration??(s.kind==='diagnostic'?1:s.completed||null))+' / '+num(s.kind==='diagnostic'?1:s.planned));
  text('rm-stage',(s.role==='validate'&&s.algorithm!=='state_ppo'?'参数验证':roles[s.role]||'等待下一批次')+' · '+num(s.lanes)+' 环境');
  text('rm-step',Number.isFinite(p.step)?num(p.step)+' 步':'—');
  text('rm-progress',Number.isFinite(p.prefix)?'拿取 '+num(p.prefix)+' · 后续动作 '+num(p.post)+' · 已结束 '+num(p.closed):'尚无当前批次进度');
  text('rm-result-label',s.role==='validate'?'本批验证成功':s.kind==='diagnostic'?'短测成功回合':s.algorithm==='state_ppo'?'本批采样成功 · 更新前':'本批搜索采样成功');
  text('rm-success',m?num(m.successes)+' / '+num(m.episodes):'—');
  text('rm-unsafe',m?'不安全 '+num(m.unsafe)+' / '+num(m.episodes)+' 回合':'尚无完整回合报告');
  text('rm-gpu',g.available?num(g.util_percent)+'%':'—');
  text('rm-memory',g.available?g.name.replace('NVIDIA GeForce ','')+' · '+num(g.used_mib)+' / '+num(g.total_mib)+' MiB':'远端设备遥测暂不可用');
  text('rm-algorithm',isSweep?'冻结 R70 · 食指落点对照':isAnchor?'冻结 R70 · 无载接近跟踪':isHover?'冻结 R70 · 悬停 C/D 对照':cartesianTraining?'末端增量 PPO · Placo 逆运动学':pointTraining?'接触点 PPO · 右臂 7 维残差':s.algorithm==='state_ppo'?'状态 PPO · R27 前缀 + 右侧 28 关节':'自然换姿与单层转动 · 六参数搜索');
  text('rm-target',s.target);text('rm-model-status',s.qualified?'已记录独立验收通过':'尚无通过独立验收的模型');
  facts('rm-facts',[['已完成迭代 / 轮次',num(s.completed)],['进入后续动作',num(p.entered_post??m?.prefix_held)],['自然换姿达成',num(m?.natural_observed)],['目标层有接触',num(m?.touched_target)],['远端进程',(s.pids||[]).join(' / ')||'已退出'],['结果记录',s.report_accepted?'报告已接收':'当前过程数据']]);
  if(isSweep){
   const sweep=s.sweep||{},settings=s.settings||{};
   text('rm-round-label','完成候选');text('rm-round',fraction(s.completed,s.planned));
   text('rm-stage',doneSweep?'两个候选已完成冻结评估':'当前落点 '+degree(settings.contact_point_offset_deg));
   text('rm-result-label','选点结果');text('rm-success',sweep.decision==='no_candidate'?'无候选达标':sweep.decision==='selected'?degree(sweep.selected_offset_deg):'等待配对');
   text('rm-unsafe','同一种子安全首触并驱动 ≥2°，至少需 2 个，且全组无新增不安全种子');
   text('rm-model-status','冻结评估');text('rm-target',s.target);
   facts('rm-facts',[['原点基线','C · 0°'],['候选方位','+7° / −7°'],['每个候选的配对种子',num(s.lanes)]]);
   renderSweep(s);
  }else if(isPad||cartesianProbe){
   const record=(cartesianProbe?s.cartesian_probe:s.pad_orientation)||{};
   const arms=[['原 +7° 基线',record.baseline],['仅修正接近跟踪',record.previous],[cartesianProbe?'Placo · 零策略增量':isPreshape?'食指预形后调整朝向':'调整空载指垫朝向',record.candidate]];
   text('rm-anchor-title',cartesianProbe?'末端增量控制器对照':isPreshape?'右食指预形与朝向对照':'指垫朝向对照');text('rm-anchor-note','首触和驱动比较固定的 11 个种子，全组安全比较全部 16 个种子。');
   text('rm-algorithm',cartesianProbe?'Placo 中性控制器 · 零策略增量':isPreshape?'冻结 R70 · 空载食指预形':'冻结 R70 · 空载指垫朝向');text('rm-stage','配对验证 · '+num(s.lanes)+' 环境');
   text('rm-round-label','材料落点偏移');text('rm-round',degree(s.settings?.contact_point_offset_deg));
   text('rm-result-label','安全首触并驱动 ≥2°');text('rm-success',fraction(record.qualifying_fixed_seeds?.length,record.candidate?.fixed_11?.episodes));
   text('rm-model-status',cartesianProbe?(record.paired_contact_gate_met===true?'已达到 +15° 训练启动条件':record.paired_contact_gate_met===false?'下一阶段可训练接触与 +2° 驱动':'等待配对验证'):'冻结评估');
   const adjusted=record.pad_orientation?.adjusted_by_lane;
   const screen=s.preshape_screen||{};
   const geometryFacts=isPreshape?[['静态可行交接姿态（原手形）',fraction(screen.counts?.measured?.all,screen.handoffs)],
    ['静态可行交接姿态（食指伸直 6°）',fraction(screen.counts?.['index_-6']?.all,screen.handoffs)]]:[];
   if(cartesianProbe&&record.candidate)text('rm-unsafe','全组不安全 '+fraction(record.candidate.all_16?.unsafe,record.candidate.all_16?.episodes)+' 回合');
   const controllerFacts=cartesianProbe?[...cartesianFacts,['固定组首次接触仅食指',
    fraction(record.candidate?.fixed_11?.first_contact_kind?.index_only,record.candidate?.fixed_11?.episodes)]]:[['调整姿态的回合',Array.isArray(adjusted)?fraction(adjusted.filter(Boolean).length,s.lanes):'—']];
   facts('rm-facts',[...geometryFacts,...controllerFacts,
    ['相对原基线新增不安全种子',Array.isArray(record.new_unsafe_seeds)?record.new_unsafe_seeds.join('、')||'无':'—'],
    ['远端进程',(s.pids||[]).join(' / ')||'已退出']]);
   $('rm-anchor-rows').innerHTML=comparisonRows(arms.map(([label,arm])=>[label,arm?.fixed_11,arm?.all_16]),[]);
  }else if(isAnchor){
   const anchor=s.approach_anchor||{};
   text('rm-anchor-title','无载接近对照');text('rm-anchor-note','在同一 +7° 落点比较原策略与接近跟踪修正。');
   text('rm-round-label','冻结诊断');text('rm-round',degree(s.settings?.contact_point_offset_deg??7));
   text('rm-stage','无载接近 · '+num(s.lanes)+' 环境');text('rm-result-label','本批成功回合');text('rm-model-status','冻结评估');
   facts('rm-facts',[['固定种子',num(anchor.candidate?.fixed_11?.episodes??anchor.baseline?.fixed_11?.episodes)],['远端进程',(s.pids||[]).join(' / ')||'已退出'],
    ['新增不安全种子',Array.isArray(anchor.new_unsafe_seeds)?anchor.new_unsafe_seeds.join('、')||'无':'—']]);
   const anchorRows=[['+7° 基线',anchor.baseline],['跟踪修正',anchor.candidate]].map(([label,a])=>{
    const motion=a?.approach?.fixed_11||{};
    return [label,a?.fixed_11,a?.all_16,fixed(motion.requested_inward_median_mm,3),
     fixed(motion.measured_inward_median_mm,3),fraction(motion.measured_inward_frames,motion.frames),num(motion.returns_to_standoff)];
   });
   $('rm-anchor-rows').innerHTML=comparisonRows(anchorRows,['指令向内位移中位数 (mm)','实测向内位移中位数 (mm)','实测向内 / 接近帧','退回悬停次数']);
  }else if(isHover){
   const groups=s.hover_cd?.group||{},panel=groups.fixed_C_no_contact_horizon||{},gap=panel.safe_paired_min_approach_gap_mm||{};
   text('rm-anchor-title','C/D 配对结果');text('rm-anchor-note','固定 C 组无接触至时限的种子，比较重新对中前后的首次食指触层与驱动。');
   text('rm-round-label','冻结诊断');text('rm-round','C / D');text('rm-stage','原点 · '+num(s.lanes)+' 环境');
   text('rm-result-label','D 组安全食指首触');text('rm-success',fraction(panel.D?.safe_qualified_index_first,panel.D?.episodes));text('rm-unsafe','D 组不安全 '+fraction(groups.D?.unsafe,groups.D?.episodes));
   text('rm-model-status','冻结评估');
   facts('rm-facts',[['固定种子',num(panel.seeds?.length)],['D 组重新对中',fraction(panel.D?.recenter_episodes,panel.D?.episodes)],
    ['D − C 配对最小间隙中位数',Number.isFinite(gap.median_D_minus_C)?fixed(gap.median_D_minus_C,2)+' mm':'—']]);
   $('rm-anchor-rows').innerHTML=comparisonRows([['C 原点',panel.C,groups.C,fixed(gap.C_median,2)],
    ['D 重新对中',panel.D,groups.D,fixed(gap.D_median,2)]],['最小间隙中位数 (mm)']);
  }else if(bimanual){
   const v=s.bimanual_validation||{};
   text('rm-algorithm','双手对握 PPO · Placo 逆运动学');
   text('rm-result-label','冻结验证 · 双手对握达标回合');
   text('rm-success',fraction(v.goal_episodes,v.episodes));
   text('rm-unsafe','验证全组不安全 '+fraction(v.unsafe,v.episodes));
   facts('rm-facts',[['策略输入 → 输出',num(s.settings?.actor_dim)+' 维状态 → '+num(s.settings?.action_dim)+' 维动作'],
    ['当前训练目标',degree(s.settings?.target_deg)],['最近验证目标',degree(v.target_deg)],
    ['双手同时夹持',fraction(v.bilateral_episodes,v.episodes)],['接触驱动峰值',degree(v.peak_bilateral_drive_deg,3)],
    ['撤手后稳持',fraction(v.held_episodes,v.episodes)],['右手策略动作','三维平移、三项手指控制、绕层轴转动']]);
  }else if(contactTraining){
   const gate=s.contact_gate||{},f=gate.fixed_11||{},all=gate.all_16||{};
   text('rm-result-label','配对验证 · 安全首触且驱动 ≥2°');
   text('rm-success',fraction(gate.qualifying_fixed_seeds?.length,f.episodes));
   text('rm-unsafe','全组不安全 '+fraction(all.unsafe,all.episodes)+' 回合');
   text('rm-model-status',s.contact_ready?'接触策略可续训 +15°':gate.iteration?'尚无可续训 +15° 的接触模型':'等待首次配对验证');
   facts('rm-facts',[...cartesianFacts,['配对验证迭代',num(gate.iteration)],['固定组安全合格食指首触',fraction(f.safe_qualified_index_first,f.episodes)],
    ['固定组安全有效驱动 ≥2°',fraction(f.safe_drive_2deg,f.episodes)],['固定组驱动峰值',degree(gate.fixed_peak_drive_deg,3)],
    ['相对原基线新增不安全种子',Array.isArray(gate.new_unsafe_seeds)?gate.new_unsafe_seeds.join('、')||'无':'—']]);
  }else if(pointTraining||cartesianTraining){
   text('rm-model-status',s.qualified?'已通过独立验收':s.recorded_status==='independent_test_not_met'?'独立验收未通过':'尚未通过独立验收');
   facts('rm-facts',[...(cartesianTraining?cartesianFacts:[]),['材料点初始方位',degree(s.settings.contact_point_offset_deg)],['单层目标',degree(s.settings.target_deg)],
    ['已提交迭代',num(s.completed)],['本批角色',roles[s.role]||'—'],['远端进程',(s.pids||[]).join(' / ')||'已退出'],['结果记录',s.report_accepted?'报告已接收':'当前过程数据']]);
  }
  text('rm-stamp','远端读取时间 '+time(s.remote_sampled_at)+(connected?' · 每 5 秒刷新':' · 当前显示最后成功读取的数据'));
  text('rm-path','远端目录：'+s.source_path);
  text('rm-update-note',s.algorithm!=='state_ppo'?'此历史任务使用参数搜索，没有 PPO 梯度更新。':l.actor_updated?'已记录参数更新；动作效果仍需冻结模型后的物理验证。':'等待完整采样与更新报告。短测完成不等于技能验收通过。');
  facts('rm-learning',[['有效学习转移',num(l.real_transitions??l.learned_transitions)],['优化器更新',num(l.optimizer_steps)],['策略损失',fixed(l.policy_loss,5)],['价值损失',fixed(l.value_loss,5)],['参数最大变化',Number.isFinite(l.actor_max_delta)?l.actor_max_delta.toExponential(3):'—'],['近似 KL',fixed(l.approx_kl,5)]]);
  const completed=(s.episodes||[]).length>0,rows=completed?s.episodes:s.live_episodes||[];
  text('rm-episodes-title',completed?'本批回合结果':'进行中的回合观测');
  text('rm-episodes-note',completed?(contactTraining?'下表记录本批物理回合。上方的 +2° 统计来自更新后冻结策略的配对验证。':'成功需满足真实单层转动与撤手保持条件。'):'下表是最近测量值，尚未形成最终成功判定。');
  const trainedRows=(pointTraining||cartesianTraining||cartesianProbe)&&completed;
  if(bimanual)text('rm-episodes-note',completed?'显示左右手同时触及各自目标层的物理反馈。':'等待本批双手夹持和层角读数。');
  const heads=bimanual&&completed?['种子','合格双手夹持帧','两指对侧夹持子步','实测驱动峰值','最终层角','达到本级目标','成功 / 不安全','结束原因']:trainedRows?['种子','合格食指帧数','驱动命令帧数','连续驱动峰值','可归因净进度峰值','最终层角','成功 / 不安全','结束原因']:completed?['槽位 / 种子','拿取通过','自然换姿','接触帧数','最终层角','成功 / 不安全','结束原因']:['槽位 / 种子','控制步','阶段','自然换姿','当前层角','目标层接触','结构检查'];
  $('rm-episodes-head').innerHTML='<tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>';
  $('rm-episodes').innerHTML=rows.length?rows.map(r=>{const values=bimanual&&completed?[r.seed,num(r.bilateral_frames),num(r.opposed_substeps),degree(r.peak_bilateral_drive_deg,3),degree(r.final_face_deg,3),yes(r.bimanual_goal_reached),yes(r.success)+' / '+yes(r.unsafe),r.end_reason]:trainedRows?[r.seed,num(r.qualified_index_frames),num(r.drive_frames),degree(r.peak_drive_response_deg,2),degree(r.peak_isolated_drive_progress_deg,2),degree(r.final_face_deg,2),yes(r.success)+' / '+yes(r.unsafe),r.end_reason]:completed?[num(r.lane)+' / '+num(r.seed),yes(r.prefix_endpoint_held),yes(r.natural_ready),num(r.target_contact_steps),degree(r.final_face_deg,2),yes(r.success)+' / '+yes(r.unsafe),r.end_reason]:[num(r.lane)+' / '+num(r.seed),num(r.step),r.controller_phase,yes(r.natural_ready),degree(r.face_deg,2),yes(r.right_contact),yes(r.structure_ok)==='是'?'通过':yes(r.structure_ok)==='否'?'异常':'—'];return cells(values)}).join(''):emptyRow('等待后续动作观测。',heads.length);
  const history=[...(s.history||[])].reverse();text('rm-history-count',num(history.length)+' 条记录');
  const historyHeads=['迭代 / 轮次','有效转移','策略损失','参数最大变化',contactTraining?'固定组首触且驱动 ≥2°':pointTraining||cartesianTraining?'验证达标':'验证成功',contactTraining?'配对全组不安全':'验证不安全'];
  if(!contactTraining&&(pointTraining||cartesianTraining))historyHeads.push('独立验收达标');
  $('rm-history-head').innerHTML='<tr>'+historyHeads.map(label=>'<th>'+esc(label)+'</th>').join('')+'</tr>';
  $('rm-history').innerHTML=history.length?history.map(r=>{const v=r.validation,learning=r.learning||{},independent=r.independent_test;
   const values=[r.index,num(learning.real_transitions),fixed(learning.policy_loss,5),Number.isFinite(learning.actor_max_delta)?learning.actor_max_delta.toExponential(3):'—'];
   if(contactTraining){
    const gate=r.paired_gate;
    values.push(fraction(gate?.qualifying_fixed_seeds?.length,gate?.fixed_11?.episodes),fraction(gate?.all_16?.unsafe,gate?.all_16?.episodes));
   }else{
    values.push(fraction(pointTraining?v?.qualification_successes:cartesianTraining?v?.index_driven_successes:v?.successes,v?.episodes),fraction(v?.unsafe,v?.episodes));
    if(pointTraining||cartesianTraining)values.push(fraction(pointTraining?independent?.qualification_successes:independent?.index_driven_successes,independent?.episodes));
   }
   return cells(values);
  }).join(''):emptyRow('尚无完成的正式训练迭代。',historyHeads.length);
  text('rm-log-stamp','日志更新 '+time(s.log_updated_at));
  const logs=$('rm-log'),bottom=logs.scrollHeight-logs.scrollTop-logs.clientHeight<35,old=logs.dataset.run;
  logs.textContent=(s.log||[]).join('\n')||'尚无运行日志。';logs.dataset.run=s.id;if(bottom||old!==s.id)logs.scrollTop=logs.scrollHeight;
 }
 async function api(path,signal){const r=await fetch(path,{cache:'no-store',signal});const data=await r.json();if(!r.ok)throw Error(data.error||'读取失败');return data}
 async function refresh(force=false){
  if(!active||document.hidden||busy&&!force)return;
  const tag=++serial;busy=true;controller?.abort();controller=new AbortController();const signal=controller.signal,abort=controller;
  const timeout=setTimeout(()=>abort.abort(),20000);
  try{
   if(force||!catalogAt||Date.now()-catalogAt>15000){
    const cat=await api('/api/remote/runs',signal);if(tag!==serial)return;
    connection(cat);const select=$('rm-run'),requested=new URLSearchParams(location.search).get('run'),old=select.value||requested;
    select.replaceChildren(...(cat.runs||[]).map(r=>new Option(runLabel(r),r.id)));
    if((cat.runs||[]).some(r=>r.id===old))select.value=old;
    else if(requested&&requested===old&&cat.connection?.ok){
     select.selectedIndex=-1;$('rm-body').hidden=true;text('rm-error','远端目录中尚无此实验：'+requested);$('rm-error').hidden=false;return;
    }
    if(!cat.connection?.ok&&!cat.connection?.stale){text('rm-status','连接中断');return}
    catalogAt=Date.now();if(!select.value){text('rm-error','远端尚未发现符合条件的训练实验。');$('rm-error').hidden=false;return}
   }
   const id=$('rm-run').value;
   const snap=await api('/api/remote/training?run='+encodeURIComponent(id),signal);
   if(tag!==serial)return;
   if(snap.id&&snap.id!==id)throw Error('远端返回了不同实验的数据');
   if(snap.id)render(snap);else connection(snap);
  }catch(e){if(tag===serial&&active){text('rm-link','连接中断');$('rm-link').className='pill warn';text('rm-status','连接中断');text('rm-error','远端数据未刷新：'+e.message);$('rm-error').hidden=false}}
  finally{clearTimeout(timeout);if(tag===serial)busy=false}
 }
 $('rm-run').addEventListener('change',()=>{
  const url=new URL(location.href);url.searchParams.set('run',$('rm-run').value);
  window.history.replaceState(null,'',url);$('rm-body').hidden=true;refresh(true);
 });
 $('rm-cohort').addEventListener('change',renderSeeds);
 $('rm-angle').addEventListener('change',renderSeeds);
 $('rm-refresh').onclick=()=>refresh(true);
 $('rm-log-toggle').onclick=()=>{const log=$('rm-log');log.hidden=!log.hidden;$('rm-log-toggle').setAttribute('aria-expanded',String(!log.hidden));text('rm-log-toggle',log.hidden?'展开日志':'收起日志')};
 setInterval(()=>refresh(),5000);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()});
 return {setActive(on){active=on;if(on)refresh(true);else{serial++;busy=false;controller?.abort()}}};
}
