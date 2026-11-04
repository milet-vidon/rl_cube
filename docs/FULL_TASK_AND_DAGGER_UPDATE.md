# 完整任务终止与实际状态纠正标签（2026-11-04）

新增可选FullTaskEpisode，解决原pickup环境在双手拿起成功后就返回done、旧转层脚本却继续执行的问题。原environment.py/rewards.py及所有碰撞、速度、限位、抓取/腕部/居中/转层验收函数保持原字节。新任务将generic pickup success明确记为milestone，完整15/30/90度目标各自使用原最终gate；drop/timeout/安全失败锁终止，下一物理step拒绝。task reward固定0，仅用于teacher/BC数据，不能直接作为PPO reward。原reward、done、terms与base计数原样留证，独立task统计避免重复pickup reward混入新任务。

所有三个控制阶段都经过step/finish事务，完成原审计后写独立task_episode_transitions.jsonl；无物理步的规划失败写final event，不伪造transition。构造时冻结全任务horizon，回合中不能延长。15/30度子技能成功与90度face_turn_validated分开，均不自动宣称六面复原。v17历史3309帧CPU shadow复核了3092 pickup milestone、217个raw done计数、最终仍失败，以及注入drop/timeout后不再step；不是新物理回合。安装后的29项host测试（20终止/实际控制流加9native facade）通过。

原生关节接入只通过--passive-cube-articulation显式启用，保留原27体与6hinge/20spherical、自由Core、自碰撞、实测质量/惯量/完整COM frame；native66DOF读回armature=.0001、dry friction=.001、linear damping=.0005，无drive/target/actuation。拒绝与旧D6阻尼/source Core惯量选项叠加。非线性阻尼尚未迁移，不作跨引擎完全等价声称。默认与right-only材质两项120步真实集成smoke均通过；完整teacher结果仍待验证。

## 真实状态采样

v1采集接口在0控制步因USD合法无限限位JSON编码失败；v2只将physics metadata的±Infinity显式标记、NaN拒绝，原观测/动作仍严格finite。v2使用原teacher原float32动作，1647步target/q/cube位置与baseline逐值一致，原wrist保持3.033秒，raw done全false、无碰撞或控制期cube pose写入。它是接口验证，不是student成功。

v3首次真实DAgger混合采样完成1648步，1419 teacher、229 student，原wrist gate通过、最大举升0.368688m。先teacher warmup到反馈缓存初始化，再以显式seed按0.8 teacher概率混合；右28始终保留原native动作，学生只改左28，expert _applied缓存与实际执行目标同步但IK意图/tick保持并记录。1648个expert projected labels用原project_target重算逐bit通过；1419个teacher实际目标逐bit一致，55帧发生真实memory同步。所有raw done/drop/timeout为false，唯一collector terminal在1647，没有后续物理行。此结果不能称纯student或独立泛化成功。

训练标签区分raw expert request、expert projected target和student actual command；不能用student实际动作冒充专家标签。各数据域记录源码、资产、配置、关节/材质/solver等完整physics identity，普通刚体/D6/native域不混训。原run的eligible_training_episode=False保留；后续训练须通过独立资格清单，只允许核验后的安全transition做BC纠正监督，不作为RL完整episode或成功认证。

每帧重复约430KB的immutable provenance已改为一次写盘与257字节hash引用，消费者先校验原字节SHA和physics hash再恢复。真实0/1000帧whole-record恢复逐值相同，teacher/student/abort三路径与12种篡改/非法引用负向测试通过。两个样本复制/序列化耗时约17.35/5.34倍改善，仅是离线组件，不宣称PhysX或整体训练速度提升。v3相关38项host测试及两位独立审查通过。

## 当前执行与证据

当前唯一Isaac：teacher_native_fulltask_v18_20261104，PID1178403，固定--max-steps4500覆盖从reset到15度目标的整个任务；使用新FullTask和native物理、规划系pacer、ring支持及12CPU预检。这是完整物理迁移候选，不是单参数消融。实际结果尚未完成，不可写成通过。

同一时间准备v5状态历史反馈残差模型，冻结v4基线、使用真实mixture的expert projected标签；训练前先审查各关节残差范围/原projection可达性与独立BC资格清单。单次开发轨迹不支持泛化结论，尚无新v5通过。

源码入口：full_task_episode.py、passive_cube_articulation.py、collect_left_dagger_baseline_v2.py/v3.py、dagger_projected_collection_v3.py。完整命令在各run protocol/experiment。证据：docs/evidence/teacher_optimization_20261103与docs/evidence/left_dagger_training_20261104；后者156项索引含三run、gzip轨迹/transition、源码/plan快照和独立审计。旧v11–v17的post-done数据仅为物理诊断，原文件不改。

后续结果：v18已在3017步正常失败终止，未开始转层；v5两轮GPU训练完成，闭环对照继续。详见[状态反馈v5更新](STATE_FEEDBACK_V5_UPDATE.md)。
