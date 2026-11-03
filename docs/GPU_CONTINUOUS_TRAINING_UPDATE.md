# GPU 策略训练与持续闭环迭代（2026-11-03）

用户要求尽量使用远端资源并持续迭代到较好模型。当前任务保持进行：先左手稳定拿持和右手真实转层，再做换面、重抓与视觉求解。没有放宽原验收，也没有把训练 loss 下降当作任务完成。

## 资源与执行方式

远端为 32 CPU 线程、31 GiB 内存、RTX 5070 Laptop 8 GiB。原仿真是单环境 CPU PhysX；现有代码将世界路径、接触回调和动作张量写为单环境，不能直接修改 num_envs/cuda 而继续声称传感器与验收有效。

新增纯 CUDA 神经训练，与 CPU 仿真及离线几何检查并行。GPU v2 运行时观测利用率超过 70%，20,000 次更新耗时约 51.58 秒。离线碰撞包络用 12 个 CPU worker，单 worker 约一个 CPU 核、常驻内存约 178 MiB；完整 13,671 项检查耗时 102.15 秒。单机仍只有一个 Isaac 进程。GPU训练完成后由队列接续物理验证；不会为了占用率重复已停滞的同配方训练。

## 新策略与真实结果

v1 `teacher_phase_contact_gpu_v1_20261103` 使用上一实测 q、qd、applied target、魔方位姿、接触，加记录的 phase/progress，共 198 维输入，预测左侧 28 关节目标增量。四条通过左手抓取与居中的轨迹前缀共 8307 个样本；失败的右手尾段明确排除。全部为已见 seed 9501，验证按共同 phase-progress 区段划分，不是独立泛化测试。

v1 CUDA 训练 8900 updates、19.86 秒，验证无改善后早停。完整闭环只运行至 851 步，最高举升约 4.79 mm，随后左无名指 DIP 与中指 DIP 自碰，原安全门停止。误差从 ready 阶段就逐渐积累；不能把其较小的单步预测误差当作可用抓取模型。

v2 `teacher_phase_absolute_gpu_v2_20261103` 改为学习 phase/progress 对应的绝对目标，并叠加不超过 ±0.01 rad 的实测状态网络修正；屏蔽上一目标特征，减少 identity 拟合与增量误差积累。执行仍受每步 ±0.04 rad 和原环境限位/速度约束。20,000 updates、51.58 秒，best checkpoint step 19300，SHA256 `bd5e799cbc401459473c96eb32d87f8392e0bf7d588ac3dc67b504f00e00aadb`。仍需要记录的阶段与右手日程，不是自主阶段切换、右手转层策略或 PPO。

v2 闭环已排队：`teacher_phase_absolute_gpu_v2_eval_9501`。仅使用实时状态进行左手推理，要求原居中保持与 J5/J6/J7 协调共同通过；不提供参考左手目标给推理，不晋级任何阶段证书。队列 PID 584920、状态 `work/teacher_optimization_20261103/phase_gpu_eval_queue_v2.json`；待 v15 Isaac 退出自动执行。best/last、optimizer、RNG、输入数据和源码哈希均保留，可恢复训练，不覆盖 v1。

## 接触技能并行推进

v14 `teacher_joint_paced_baseline_v14_20261103` 完成 3322 步，双手夹持合格保持 4.033 秒。转层反馈执行 120 步，其中 90 步保持已有目标，右手五指全程有接触，但实测层角只到 0.117°，最终明确报 progress stalled。左食指在该阶段 105/120 帧失触。这说明单独限制推进仍不足以得到合格转层。

v13 的左无名指失触另有明确证据：源阻尼改变了前置自适应抓型，接触更靠角缘；右手闭合后冲量从约 0.00687 Ns 变为 0，实测掌部校准 FK 显示原接触点约 21.5 μm 间隙。新增默认关闭 `--left-ring-support` 候选，在预检后对 ring PIP 施加 `[0,.004] rad` 有界偏置、每步 `.00005 rad`，根据有效接触冲量在 `.0005–.002 Ns` 区间调节。

包络重建移动左指的碰撞几何，检查完整右手路径、全部插值和左指偏置采样，包括左指自碰、躯干、桌面、静止层与旋转层；原距离门槛不变。v13 保存计划的 13,671 项检查通过只说明该包络几何可行，不代表已物理成功。

v15 `teacher_joint_ring_support_v15_20261103` 已启动，PID 575572，日志 `work/teacher_optimization_20261103/ring_support_v15.log`。以 v13 配置新增支持反馈，运行时重新做本回合完整预检。该必要回合同时用 cProfile 保存热点 `ring_support_v15.prof`，用于定位下一步吞吐优化。必须等完整结果再判断；v14 左食指问题尚未由此修正。

## 验证与复现

本轮相关测试累计 91 项通过（原 74、左无名指支持 5、v1 神经特征/来源 8、v2 绝对目标 4）。真实 PhysX 结果与离线测试严格区分。56 DOF、碰撞、限位、速度、原始接触及 3 秒保持门槛保留；无控制中魔方位姿写回。

激活 `/home/brainco/rl_cube/activate.sh`，在 `github_project/revotron_cube_rl` 下设置原 REVOTRON_USD_PATH / ISAAC_MUJOCO_CUBE_ROOT 后，独立评估命令为：

```bash
python evaluate_phase_absolute_v2.py \
  --checkpoint runs/teacher_phase_absolute_gpu_v2_20261103/best.pt \
  --source runs/center_j56_dev_v3_20261103 \
  --output runs/UNIQUE_NEW_EVALUATION --seed 9501
```

训练配置与完整来源以 `docs/evidence/phase_contact_training_20261103/*/experiment.json` 为准；模型、指标、闭环失败轨迹和源码快照使用 LFS 归档。转层与接触证据在 `docs/evidence/teacher_optimization_20261103/`。继续接手时先读两个运行状态，不要重复启动 Isaac；结果失败则按诊断继续修正，不声明自由旋转模型或视觉复原完成。


## 后续真实结果与当前续跑

v15 完成3294步。支持反馈最大偏置0.00115rad、非零199步；保持121/121和转动119/119帧双手十指都接触，核心角速度均在原门槛内，五指合格保持4.033秒。不过层角仅0.027°、pacer连续保持90步后明确停滞，没有转层成功。左食指候选另做了9084项预检，但v15已经不失触，因此暂不把该候选混入下一单变量实验。

v2 神经模型闭环1886步失败，最高举升0.374m；进入center后落桌。进一步逐帧核验发现tilt_hold 94帧从未五指同时接触，不能称为已通过原腕部阶段验收。误差分解显示：427步采用示范实测状态时target误差约0.00128rad，采用模型自身状态时约0.01991rad，有界state feedback饱和扩大了误差。v3正改为同一基线物理源的高精度阶段策略，并添加原腕部接触晋级要求，待完整验证。

v2另保留独立继续训练副本teacher_phase_absolute_gpu_v2_extended_20261103，从20000到32000updates后因验证平台期停止；最佳scaled验证loss仅从.002171改善到.002139，不能视为已解决闭环问题。resume_parent.json记录父checkpoint哈希和命令，原v2保持冻结。

v16 teacher_joint_supported_open_v16_20261103，PID704085，日志work/teacher_optimization_20261103/supported_open_v16.log：保留v15源线性阻尼近似、ring支持和右指刚度10，仅取消turn-feedback-pacing以测量实际驱动与失触边界；原最终验收不变。另启用--preflight-workers 12，v13/v14逐检查bool/角度/失败对、全部关节路径点已验证与原串行一致，最终检查由约75–78秒降至约16–17秒。新原生退出前flush的profile_simulation.py记录该必要回合热点；v15直接用python -m cProfile的尝试因Lab3原生退出没有留下profile文件，不能报告其热点数据。

当前相关单测97项通过（原96加多项式系数语义测试）；v16已冻结本回合源码。更完整的free-base articulation动力学迁移仅做独立审计和probe准备，未替换当前机器人环境。


v3先使用分块holdout时在缺标签区间出现约.097rad最大误差，该run保留但未送闭环。v3 dense使用单轨迹每5帧留1帧的邻帧插值验证，明确不是独立泛化测试；采用12个阶段输出头、关闭未经学会的state correction。完成20700updates/52.54秒，同基线各阶段最大误差≤.003554rad、center_hold RMSE约4.31e-5rad。checkpoint SHA256 354a63778a0f846b09873374c3b36989152d43c0b2ab976ed406bd6b84e27050，等待v16后自动闭环；新eval在进入center前按原腕部/五指连续保持及assembly门槛拒绝不合格轨迹。不能把该版本称为自适应状态策略。

v15反馈坐标审计另发现计划掌角与current Core相对层角并非同一参考系。新增独立turn_feedback_frame helper及回放/单测，目前尚未集成到活动控制；映射保留夹持时基线偏移，不能把固定旧轨迹上允许推进的状态当作新控制器成功。左食指候选和完整动力学probe同样保留为未集成/未验证候选。
