# 居中后继续左食指反馈

native v18抓取和居中通过，但进入右接抓后左index在2063步即失触（右手尚未接触）；右little在2997步首次接触，ring在3007步失触，原retained-grasp门于3016步失败。不是先增加全手刚度。本候选只把原center的index MPR反馈延续到右手接抓/转层阶段：仍用[-.012,0]rad、5e-5rad每步、原冲量区间和原目标面判断。用最后nominal target减其已应用offset恢复reference，首步直接reference+next_offset，避免重复叠加。

启用项--left-index-continuation必须同时启用--full-task-episode、--left-ring-support及j56居中。先在整条原右手计划上联合检查index×ring所有采样组合，包含两指相互、其他指/臂/躯干/桌、静层和活动层间隙；通过后才控制，所有原安全/姿态/接触/速度/终止门保持不变。

实测v18计划的1502path samples×112组合=168224项检查通过，12workers耗时1181.79s；这仅是离线几何。新增v3逐job距离缓存保留全部predicate，只复用完全相同参数的距离并还原last_pair副作用，最多100000条且每job清空。336项真实样本4251632次distance/last_pair trace、336次valid trace逐bit相等，样本30.924→20.753s，不承诺完整run同倍加速。Coal3.0.3实际默认request/Convex隐藏状态经独立核验，不支持改变warmstart请求或几何后复用。

41项相关host测试通过（ROS几何环境及Torch环境）。当前teacher_native_index_continuation_v19_20261104，PID1415628，固定4500总步/15度目标/seed9501；相对v18控制变化仅食指反馈接续，仍待真实完整物理验收。日志work/teacher_optimization_20261103/native_index_v19.log；冻结launch review和源码/预检/缓存证据在docs/evidence/left_index_continuation_20261104。

并行模型路径：matched v4开发回合1654步通过原wrist gate、最大举升.368716m、左位置动作无teacher接管，仍使用外部阶段/右手/原K20日程。v5long纯左模型在619步夹紧失败；正在用其安全失败状态及公开阶段边界训练v6反馈策略。没有右手自由转层或视觉复原成功。
