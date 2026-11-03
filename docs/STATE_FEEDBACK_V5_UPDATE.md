# 状态历史反馈 v5（2026-11-04）

新增冻结v4阶段基线之上的GRU64左28关节残差模型。284维输入仅包括实际关节、已执行目标、魔方状态、接触、公开阶段与历史；专家目标、标签、上次动作来源不进actor。右28不由模型学习，仍保留原控制器。当前范围是pickup到腕部保持，未训练转层或视觉模型。

训练用真实v3 mixture的1648条已核验安全BC transition（1419 teacher、229 student），外置provenance/源码/资产/投影全部逐hash绑定。原run eligible_training_episode=False与runtime_bridge_validated=False保留；独立资格仅批准BC转移监督，不能当RL完整episode或纯student成功。28关节全部标签投影不可达距离严格0，未裁剪标签。19项训练/数据host tests及独立审查通过。

| 候选 | epochs | 最佳epoch | 实际原投影RMSE(rad) | 最大误差(rad) | GPU训练循环(s) |
|---|---:|---:|---:|---:|---:|
| 冻结v4基线 | — | — | 0.000671184 | 0.004999995 | — |
| v5首轮 | 200 | 199 | 0.000119426 | 0.004390836 | 6.255 |
| v5长训练 | 2000 | 1998 | 0.000008071 | 0.000395268 | 24.483 |

时长不含CPU数据读取与来源核验。只有一条开发回合，无held-out回合；以上均为离线拟合。首次闭环在观察任何闭环结果前，依据开发拟合选择长训练模型，SHA256为1c663e17f6157273ee4c80b1e6ebf1c292b6c9972278041936b56943212db98c；首轮模型保留，未覆盖。

新评估入口evaluate_state_residual_v5.py/state_residual_eval_runtime_v5.py供v5和v4共用。左手从reset首控制步全模型，无teacher左手warmup；GRU每回合清零并逐帧读取本次实际状态。原teacher只供公开阶段、右手动作、原增益日程及诊断专家标签，所有来源与干预计数留证；诊断标签尚未重新资格化。原增益、碰撞、接触、腕部和原生done均保留。12项评估host tests与两模型preflight通过。

首次left_state_residual_v5_long_eval_9501_20261104在0个policy控制步被asset contract拒绝：activate.sh默认旧目录56DOF入口，未覆盖为原训练的wrist-mesh入口；构造阶段存在预热physics，不能称0底层物理步。修正版仅显式导出原机器人/cube资产路径并绑定三个资产SHA，冻结模型/代码/contract断言不变，使用新run名称_v2。已排串行v5long与v4同adapter对照。尚无闭环通过结论。

命令：source /home/brainco/rl_cube/activate.sh 后，在revotron_cube_rl目录使用evaluate_state_residual_v5.py；完整冻结shell及SHA在docs/evidence/state_feedback_v5_20261104/review和初始化失败归档中。当前队列与准确进程见work/teacher_optimization_20261103/state_feedback_matched_eval_queue.json。训练源码和模型证据在docs/evidence/state_feedback_v5_20261104；原数据在left_dagger_training_20261104。

并行teacher路径：v18完整任务3017步在右手闭合时正常触发left_grasp_lost_during_opposition，未执行转层；无终止后继续执行。左index在右手接触前即失触，随后ring失触。正在审查仅续接原center index反馈的新候选，联合index×ring几何预检不放宽原门槛。90°原生初态v2探针24步通过，但只验证表示及被动稳定，不是机器人转动成功。
