# 阶段门控状态策略 v6（2026-11-04）

v6 用两条真实开发历史训练：原 v3 teacher/student 混合历史的 1,648 个合格状态，以及失败 v5 闭环中实际执行并审计的 619 个安全状态。合计 2,267 帧，其中 848 帧由模型执行。它是带监督的状态策略训练，不是 PPO，也没有独立测试成功。

原 v5 在无反馈的接近阶段产生微小残差，接触轨迹随后分叉。新模型在 ready/preshape/lower/approach 和 squeeze 的原反馈激活边界前逐位保持冻结 v4 输出，但 GRU 每帧仍更新。squeeze 从公开进度 72/84 开启残差；边界绑定原 teacher 源码与实际 73 点计划。11 个有真实恢复监督的手指关节允许 ±0.006rad 修正，7 个臂关节为 ±0.0005rad，其余手指为 ±1e-6rad，原软限位和实际动作投影仍执行。

最终资格检查：1,269 个 active 状态的全部原始标签严格可达；998 个 masked 状态只剩最大 1.1920929e-7rad 的 float32 差异，原标签全部保留在损失中。没有裁剪标签、过滤不可达样本、改变失败结果或抬高运行时门槛。最终资格 SHA256 为 5cdd1214aa7fd295e9851d2ffeaee4fca28838c8b3b989bbce499cf303fac117。

第一版 GPU 训练 2,000 epochs，最佳为 1,999，checkpoint 位于 runs/teacher_phase_gated_residual_v6_20261104/model.pt，SHA256 523e7163d90da38ace84eb07597ccc0274a5ced14f2ae74d05ddb6dbc30eff04。开发标签 RMSE 1.2876922e-5rad、最大误差 0.0005216rad；模型实际访问状态 RMSE 1.0025875e-5rad。这些仅是离线拟合指标。相同资格的 10,000 epochs 长版已完成，最佳 epoch9995，checkpoint SHA256 0bedb191b59e302abc7a273c6c0071c0400d3687bc5d893ad0a84c566de02d48；开发标签 RMSE2.03165e-6rad、最大0.000142217rad，独立输出目录未覆盖第一版。

闭环评估器从 reset 开始由模型输出全部左侧位置动作，右手、公开阶段及原 K20 增益事件仍由原控制器负责。评估桥的控制、终止和审计流程与冻结 v5 一致，单独绑定新模型、两条数据历史和资格哈希。安装后的 46 项 host tests 通过；实际 checkpoint/source/assets/family 的 preflight 通过。短版真实闭环 runs/left_phase_gated_residual_v6_eval_9501_20261104 已在752步 primary_lift 因原 contact_topology_or_settle_timeout 失败，最大举升0.187796m；左位置动作全部由模型执行，未发生原增益事件，未通过腕部门槛。长版同配置闭环同样在752步 primary_lift 原接触拓扑超时，最大举升0.187920m。短版752帧actor输入和输出逐帧重放bitwise一致，第一处动作差在step499；最后臂位置/速度/高度均合格，食指失触使原五指保持计数为0。需要实际偏离状态的纠正监督，继续增加同一数据的epochs不足以解决。两训练和两完整评估的checkpoint/源/轨迹见 docs/evidence/state_feedback_v6_20261104，原失败与False标记全部保留。

多初态限制已有反例：冻结 v4 在已见开发 seed9502 的 360 步全部停留 ready，原 settle timeout 终止；没有抓起。相对9501，cube 初始位置变化约22mm、方向变化6.014°，原 teacher 重算的臂目标最大变化0.133998rad，而模型仍输出旧的阶段轨迹。末帧关节速度已满足门槛，位置误差0.133987rad大于原0.02rad门槛。因此 v6 当前的 early-phase hard mask 也不能解决这类初态差异，需要多位姿原 teacher 数据及初态条件接近策略。

双手物理路径也未完成：native v19 共3,019步，抓取和居中通过，但 left_grasp_lost_during_opposition 终止，未开始转层。食指反馈短暂恢复接触，2425步达到-.012rad控制范围，2427起再度失触；下一版正对实际几何检查有界 MPR 范围扩展。全部原接触/碰撞/姿态门槛保留，无控制期魔方位姿写入。尚无15°/90°接触转层、自由换面或视觉复原成功。
