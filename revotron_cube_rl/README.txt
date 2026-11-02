双 Revotron + 双 Revo3 + 胸部相机：魔方强化学习准备工程
实现说明更新：2026-10-31。当前任务是锁层魔方的双手拿起与保持，还不是转层或还原。

已确认与尚待确认
runs/downward_pitch23/results.json 记录：seed=17、width=0.098/0.096/0.094 的向下多指姿态传统控制通过，抬升约 0.115 m，达到 30 个连续低速持稳步，机器人-桌面累计接触冲量为 0；真实接触对以该 run 的记录为准，不能把设定了三根手指等同于三根手指都承重。
该控制器读取初始魔方真实位姿并执行 IK/PD 轨迹，证明的是仿真物理可行性。
runs/traditional_visual_verified 已完成 width=0.096、种子 17,31,47,59,71,83,97,109 的实际胸部 RGB 采集，8/8 传统控制成功，导出 3126 条 v3 示范转移。首个视觉种子 17 为 391 步、约 40.2 秒、最大抬升约 11.496 cm、机器人-桌面接触为 0；胸部和总览图已检查正常且魔方可见。完整依据是 results.json、trajectory_*.json、controller_*.gif 和 demonstrations.npz。相机画面检查不是学习策略的视觉能力验收。
runs/traditional_long_hold 在未用于示范的种子 101,131,157,181,211 上取得 5/5 传统控制成功：线速度 <0.02 m/s、角速度 <0.3 rad/s，连续保持 3 秒；最大抬升 0.1158037 m，所有试验机器人-桌面累计接触冲量为 0。这是更严格的传统控制验证，不是视觉 actor 评估。
runs/visual_warmstart 已完成仅 actor 的 BC 预热：80 epochs、记录耗时 48.87 秒，selected_epoch=78；训练 2345 条、留出 781 条，按完整 episode 分离，留出 MSE=0.00092944。critic 没有预训练。
runs/visual_warmstart_evaluation（seed=307、1200 步）与 runs/visual_dagger_evaluation（seed=401）的纯策略确定性评估均为 0/2，尚未学会抓取。
runs/dagger_round1 采集了 seed=313/331 各 391 步，共 782 条专家查询，加上 3126 条原始成功示范后共 3908 条。采集时 80% 教师 + 20% 策略混合控制取得 2/2 成功；这不是纯策略成功，专家查询标签也不是该步实际执行动作。
runs/ppo_v3_check 已真实完成 10 次 PPO 更新、320 步、34.61 秒；loss 有限且策略参数发生非零更新，但没有完成 episode，因此不能报告任务成功率。
POLICY_HISTORY_DIAGNOSTIC.txt / .json 记录了 50 个轨迹点的局部诊断：动作对上一步动作的导数谱半径均 >1，范围 1.289–1.633，中位 1.410。它支持存在历史动作捷径/放大反馈的判断，并不证明这是唯一失败原因。
runs/visual_feedback_policy 已基于 DAgger 数据完成 200 epochs，启用 --normalize-feedback 与 --disable-action-history；selected_epoch=199，训练 MSE=0.000522522，留出 MSE=0.000946548，记录耗时 144.71 秒。纯策略 runs/visual_feedback_evaluation（seed=503、600 步）为 0/1；无桌面支撑且双侧接触的最大抬升为 0.01490587 m，符合该接触条件的步数共 14 步，未达到 8 cm/连续 1 秒门槛。
runs/visual_iterations 与 runs/visual_iterations_cont1 已完成两批串行 PPO，合计 19200 训练步、14400 评估步，所有候选确定性评估均为 0/3，最佳 checkpoint 没有替换。随后完成 runs/dagger_round2_retry → runs/visual_dagger_warmstart_round2 → runs/visual_ppo_round2：DAgger 新增 1173 条查询，BC 选中第 184 轮且 holdout MSE=0.0003910087，但纯策略仍为 0/3，PPO 后单回合仍为 0/1。又完成 beta=0.5 的 runs/dagger_round3_beta05，并合并三轮 query 为 6253 条数据；round3 BC 选中第 1 轮、holdout MSE=0.0002905159，但 seed=601 的纯策略 1800 步评估仍为 0/3，出现过单侧接触和机器人-桌面冲击。当前没有 Isaac 训练进程；实时结果以 STATUS.txt 和各 run/result.json 为准。checkpoint 存在、MSE 下降、黑图导致动作变化，都不能单独证明已经靠视觉学会抓取。
随后 round4 加入八条新的传统视觉成功示范，与前三轮 query 合并为 9378 条数据；BC holdout MSE 降到 0.00022874，但 seed=901 纯策略仍为 0/3，三段都没有接触魔方。离线审查发现几乎相同的初始图像会触发约 -0.257/-0.222 的左右臂动作和，而教师约为 -0.039/-0.036，说明主要瓶颈是视觉外推和增量目标漂移。
round5 以 beta=0.3 采集 seeds=941,977,1009，3/3 混合控制器成功，新增 1174 条 query；合并后 10552 条数据。加入小幅 RGB 平移/曝光/噪声与本体去噪的 warmstart BC 选中第 41 轮，原始 holdout MSE=0.00039255；seed=1021 纯策略 0/3，但其中一段出现 132 步双侧接触、约 2 mm 抬升。随机初始化增强 BC 仍为 0/3，且有较大机器人-桌面冲量。全程动作 gain=0.5 与前 90 步 ±0.12 限幅诊断均未接触。下一步应优先改成魔方中心视觉表示、绝对关节目标或 Cartesian residual，并对失败窗口加权；暂不把这些混合成功当作学习成功。

当前 v3 接口
- 动作：56 维，左/右各 7 臂关节 + 21 手关节；全部开放，不限定拇指夹持。
- 控制：240 Hz 物理、30 Hz 目标更新；手臂每步最大增量 0.025 rad，手指 0.06 rad，累计目标限幅后由隐式 PD 执行。
- actor 本体觉：224 维 = 56(q-home) + 56(0.1*qdot) + 56 上一步已执行动作 + 56(target-q)。最后一项暴露已知 PD 目标状态。
- 224 维原始协议不变；当前反馈策略通过 checkpoint 内的 actor.proprio_scale 将 112:168 的历史动作槽屏蔽，末 56 维 target-q 按臂 0.025 / 指 0.06 rad 动作尺度归一化。q、速度和 RGB 保留，critic 输入与网络不变。屏蔽历史动作会有意改变 actor 函数；目标误差归一化通过首层权重补偿保持变换当时的函数。
- actor 总输入：49376 维 = 224 + 128*128*3 的 RGB [0,1]。图像是 Isaac 相机实际渲染的仿真胸部画面。
- critic：247 维 = 224 本体觉 + 13 魔方位姿/速度 + 6 双掌相对位置 + 2 接触冲量 + 2 持稳进度/剩余时间。物体状态和接触只供 critic，不能输入 actor。
- 每 episode 最多 600 控制步，即 20 秒；成功、掉落或超时终止。--steps 是一次 PPO rollout 或评估的总步数，不会覆盖 episode 上限。
- checkpoint/demonstrations 必须匹配 config SHA256。v1/v2 与 v3 不是同一个观测和奖励合同。

脚本与数据入口
- config.json：关节顺序、资产版本、相机/PD 参数、v3 维数、600 步时域、discount=0.998、可审计奖励系数。
- environment.py / policy.py：真实渲染观测、接触报告、物理控制、CNN actor 和特权 critic。
- traditional_controller.py：历史水平轨迹与当前 DownwardController（高抬、转腕、下降、合拢、抬起、保持）。教师可以读取仿真位姿，不是视觉策略。
- sweep_controller.py：在指定几何参数和种子下测试传统控制；--visual 记录 RGB/动作并最终只导出成功 episode 的 demonstrations.npz。失败轨迹仍保留用于诊断。
- collect_dagger.py：实际仿真状态下查询教师，用 beta*专家 + (1-beta)*策略的混合动作采集 RGB。NPZ 的 actions 是专家监督标签，executed_actions 是真实混合执行动作；supervision_type=mixed_demonstrations_and_dagger_expert_queries。失败采集也保留，查询标签不能冒充成功执行示范；--include-base 可合并原始成功示范。
- warmstart.py：纯 Torch 监督 actor，不启动 Isaac；actions 始终是训练标签，executed_actions 只校验并报告差异。支持 --initial-checkpoint、默认零噪声的纠偏增广、--normalize-feedback、--disable-action-history；增广没有物理 rollout，不能视作新增成功示范。按完整 episode 留出验证，critic 不更新，PPO optimizer 新初始化。
- checkpoint.pt 按原始留出 MSE 选中；checkpoint_last.pt 保存最终轮，status=final_epoch_not_closed_loop_validated。两者都不是“闭环最佳策略”的保证，应分别实测。actor.proprio_scale 持久保存屏蔽/归一化设置；同观测合同的旧检查点缺该 buffer 时补全为 1，其余缺键仍严格校验。
- train.py / run.ps1：场景检查、确定性 evaluate、PPO train。默认 PPO 使用固定 learning_rate=5e-5、gamma=0.998；train.py 支持 --learning-rate 与 --noise-std。
- iterative_training.py：先评估输入 checkpoint，再串行执行短段 PPO/确定性评估；按成功率、最长持稳、抬升代理量排序保留候选。零成功时，抬升代理量使用双侧真实接触且离桌条件下的 max_bilateral_height_m，不能用自由抛起的最高点代替。manifest.json/status.json 记录实际预算与最佳 checkpoint；循环约每 30 秒更新状态心跳并读取子任务 progress.json。反复用于选模型的固定验证种子不是独立最终测试；流程结束不等同于任务达标。
- start_iterative_training.ps1：已通过 PowerShell 静态语法检查；用隐藏窗口启动本地循环，标准输出/错误重定向到 launch_logs，启动信息写入 active_training_process.json。它保留已有 run，后续另开实验须使用新 Name。
- test_reward_gates.py：无需 Isaac 的成功门槛与潜势奖励测试。
- training_scene.usda / asset_contract.json：场景与资产合同快照；curriculum.json：后续阶段与升级条件。
- check_pickup_feasibility.py：早期 IK/PD 诊断，不能替代当前向下姿态验证或视觉策略评估。
- RESEARCH_AND_PLAN.txt：原始官方资料、相关工作、当前奖励及计算成本；vendor/ 保留复用源码和许可证。

推荐执行顺序
1. 固定一个能在多个初始场景中通过的传统控制配置，检查接触部位、离桌高度、连续持稳和机器人-桌面接触。
2. 完成成功示范采集并确认 demonstrations.npz 已存在；保留教师、配置、种子和轨迹记录。
3. 用 warmstart.py 预热视觉 actor；闭环漂移时可用 collect_dagger.py 收集访问状态上的专家查询继续监督。独立 episode 的 MSE 只衡量模仿误差；混合采集成功与纯策略成功必须分别报告。
4. 用未参与采集的 reset 种子做确定性闭环评估。至少给予完整的 600 步预算；若失败，先检查状态分布偏移、动作累积误差和视觉输入，不以继续加 PPO 步数代替诊断。
5. 从可用 BC checkpoint 开始小固定学习率、保守噪声的短段 PPO；每轮后重新确定性评估并保存各轮结果，反复比较，避免覆盖已验证的较好 checkpoint。
6. 只有拿起/保持的闭环指标通过后，才推进重抓、自由可转层魔方、单面与不同方向连续转面、读色和规划还原。

PowerShell 命令示例
以下路径沿用现有 Isaac Python。新输出名应唯一。直接运行 Isaac 脚本前可在当前 shell 设置：
  $taskRoot='E:\Codex\2026-10-30\w\outputs\revotron_cube_rl'
  $taskPython='E:\IsaacLab\env\python.exe'
  $env:OMNI_KIT_ACCEPT_EULA='YES'
  $env:OMNI_USER_HOME='E:\IsaacLab\userdata'
  $env:OV_USER_HOME='E:\IsaacLab\userdata'
  $env:TEMP='E:\IsaacLab\tmp'
  $env:TMP='E:\IsaacLab\tmp'
  $env:OMP_NUM_THREADS='4'

场景检查：
  & (Join-Path $taskRoot 'run.ps1') -Mode smoke -Steps 64 -Name my_smoke_v3

另建名称采集传统控制示范（已有采集仍在运行时不要同时启动第二个 Isaac）：
  & $taskPython (Join-Path $taskRoot 'sweep_controller.py') --grasp downward --widths 0.096 --seeds 17,31,47,59,71,83,97,109 --visual --name my_traditional_visual

重现 BC 预热的示例；当前数据已生成，输出改用新名称以保留已有 visual_warmstart：
  $taskDemo=Join-Path $taskRoot 'runs\traditional_visual_verified\demonstrations.npz'
  if (-not (Test-Path -LiteralPath $taskDemo)) { throw '成功示范文件尚未生成；请先检查采集结果。' }
  & $taskPython (Join-Path $taskRoot 'warmstart.py') $taskDemo --output (Join-Path $taskRoot 'runs\visual_warmstart_reproduction') --epochs 80 --seed 7

确定性评估和恢复训练使用以下已生成的 v3 BC checkpoint；运行前仍检查文件存在，它不是闭环抓取成功的保证。
  $taskCheckpoint=Join-Path $taskRoot 'runs\visual_warmstart\checkpoint.pt'
  if (-not (Test-Path -LiteralPath $taskCheckpoint)) { throw '请先完成 v3 BC 预热并检查 checkpoint。' }

先做独立种子确定性评估：
  & (Join-Path $taskRoot 'run.ps1') -Mode evaluate -Steps 600 -Seed 131 -Name bc_eval_131 -Checkpoint $taskCheckpoint -Record
  更换未参与示范的 Seed 重复评估；600 步可能包含成功后自动 reset 的下一 episode，统计时应区分已结束与未结束 episode。
  默认 --steps=32 只覆盖约 1.07 秒，不能当作完整拿起测试；evaluate 不使用 --iterations 增加预算。录像选项当前只录前 10 秒，完整约 16 秒教师动作还需检查完整 trajectory。

从 BC 接续小规模 PPO（5e-5 固定学习率，0.05 为待实测的保守探索标准差示例）：
  & $taskPython (Join-Path $taskRoot 'train.py') --mode train --iterations 2 --steps 600 --seed 137 --name ppo_from_bc_v3 --checkpoint $taskCheckpoint --learning-rate 5e-5 --noise-std 0.05
  随后用新的 PPO checkpoint 再做确定性评估。不能将 PPO rollout 的随机探索成功直接等同于确定性策略成功。

奖励检查与循环编排接口：
  & $taskPython (Join-Path $taskRoot 'test_reward_gates.py')
  & $taskPython (Join-Path $taskRoot 'collect_dagger.py') --help
  & $taskPython (Join-Path $taskRoot 'iterative_training.py') --help
  当前没有后台 Isaac 训练进程。读取历史循环状态：
  Get-Content -LiteralPath (Join-Path $taskRoot 'runs\visual_iterations\status.json')
  标准输出/错误位于 launch_logs/visual_iterations.stdout.log 与 .stderr.log。
  以下仅为本批结束后另开实验的启动示例，不要在当前进程运行时重复执行：
  $taskLoopCheckpoint=Join-Path $taskRoot 'runs\visual_feedback_policy\checkpoint.pt'
  if (-not (Test-Path -LiteralPath $taskLoopCheckpoint)) { throw '请先检查反馈策略 checkpoint 及运行结果。' }
  & (Join-Path $taskRoot 'start_iterative_training.ps1') -Checkpoint $taskLoopCheckpoint -Name visual_iterations_next -Cycles 3 -Iterations 50 -RolloutSteps 64 -EvalSteps 1800 -MaxMinutes 120
  MaxMinutes 是轮次之间检查的软预算，已经开始的一轮仍完成训练和评估；9600 步是本批计划预算，不是已完成步数。需要停止当前循环时创建 STOP：
  New-Item -ItemType File -Path (Join-Path $taskRoot 'runs\visual_iterations\STOP') -Force
  STOP 会在当前子任务结束后阻止启动下一子任务，不立即终止正在执行的训练/评估。各轮固定验证种子用于选择模型；选出的 checkpoint 仍需另换种子做最终评估。

奖励合同 potential_difference_v1
  Phi = 0.2*reach + 0.1*bilateral + 0.5*bilateral*clip(height/0.08,0,1) + 0.2*clip(hold_steps/N,0,1)
  r = 0.998*Phi(next) - Phi(previous) - 0.002 + 5*success - 2*drop
      - 0.005*mean((action-previous_action)^2) - 0.001*bounded_normalized_joint_speed
成功、掉落、有限时域超时的 Phi(next) 为 0；每次 reset 初始化新场景潜势。没有每步额外接触/稳定正奖金，避免在接近目标处持续刷旧密集奖励。成功仍要求真实双侧接触、离桌 >=8 cm、线速度 <0.08 m/s、角速度 <0.6 rad/s、无魔方-桌面接触，连续 >=1 秒。机器人-桌面接触和实际接触部位另外查看轨迹与接触对，不能由成功布尔值自行推断。

兼容性、限制与验收
- runs/visual_pickup_pilot 等旧 v1 checkpoint 不能用当前 v3 config 直接加载。重现历史结果需匹配其历史 config、观测代码与资产；不要改写哈希或忽略 shape 错误来强行恢复。
- 保持本目录与相邻 revotron_revo3、isaac_mujoco_cube 的相对位置。机器人使用本地 20261025 资产，不等同于官方 20261014 固定版本权重。
- 当前魔方是原始 26 块几何合成的自由刚体，68.5 g、受重力，内部层锁定；动作中没有绑定手或直接修改魔方位姿。这一阶段的成功不证明可转层自由核心稳定，也不证明会还原。
- 当前仅验证单环境 CPU PhysX + D3D12 RTX 相机；PPO 更新在 CPU。没有验证大量环境并行；不要直接照搬官方大规模环境数。
- 相机启动使用本机进程级 --/app/vulkan=false 绕过，不改变驱动；Kit 清理超时仅结束相应独立子进程。
- actor 接入实际图像且不读取特权状态是必要条件。真正利用图像完成操作仍需未见位姿的闭环成功，以及正常图像与冻结/错配图像等条件的对照；目前不能仅凭接线检查宣称视觉有效性已证。
- 128x128 仅是拿起阶段输入。多面读色、时序记忆、转层技能、离散解法规划和完整复原尚未完成。训练时间和最终成功率均不能先作承诺。

### 最新位置策略进度（2026-11-01）

为遵循“视觉只确定魔方位置”，新增了位置型 actor：胸部 RGB 经过红色目标分割，只向 actor 提供 6 个位置/尺度特征（质心、面积、包围盒、可见性）；actor 仍接收 224 维本体觉并输出 56 个关节目标增量。位置型 DAgger 的 β=0.6 和 β=0.4 两轮各自 3/3 混合教师回合成功，但纯策略评估仍为 0/1，且 β=0.6 评估出现约 89.5 N·s 机器人-桌面冲量。holdout MSE 仅表示离线动作拟合，不能代表抓取成功。相关结果见 `runs/position_dagger_round6_beta06`、`runs/position_warmstart_round6_beta06_active5`、`runs/position_policy_round6_beta06_eval_g05` 和 `runs/position_warmstart_round7_beta04_active5`。

### 最新绝对目标控制进度（2026-11-01）

已把位置型 actor 的输出从逐步关节增量改为 56 个绝对关节目标；运行时将目标解码后转换为当前 Isaac 环境的 PD 增量，仍由传统 PD 执行器控制双臂双手。round8 的绝对目标 BC 纯策略为 0/1，但机器人-桌面冲量约 0.637 N·s，较增量版本显著降低。直接优化反解动作的 round11 纯策略为 0/1、冲量约 7.14 N·s；加入 episode progress 的 round12 为 0/1、冲量约 1.94 N·s。三轮都没有达到双侧接触、离桌和保持门槛，不能称为已学会抓取。

绝对目标 DAgGER 的 beta=0.4 和 beta=0.8 混合采样各为 0/3 教师成功回合，因此没有把混合成功误记为纯策略成功。当前推荐 checkpoint 是 `runs/absolute_warmstart_round12_phase_actionloss/checkpoint.pt`，对应评估证据为 `runs/absolute_policy_round12_phase_eval/result.json`。该策略输入 224 维本体觉、6 维魔方位置/尺度特征和 1 维 episode progress；RGB 只用于确定魔方位置，未读入真值位姿。完整记录见 `STATUS.txt`、`current_status.json` 和 `TRAINING_RESULT_2026-10-31.txt`。

### Linux 迁移与继续训练

完整的 Linux 迁移清单、官方 `isaaclab.sh -p` 启动方式、资产目录布局和逐步验收
命令见 `LINUX_MIGRATION.md`。`run_linux.sh` 现在支持 `check`、`env-check`、
`smoke`、`evaluate`、`train`、`dagger`、`sweep` 和 `turn` 入口；迁移时优先执行
`./run_linux.sh check`、`./run_linux.sh env-check --runtime` 和短 smoke，再继续训练。

官方文档确认 Isaac Sim 和 Isaac Lab 支持 Linux；当前 Isaac Lab 安装指南列出 Ubuntu 22.04+ x86_64 和 Windows 11 x86_64，Isaac Sim 5.1 工作站要求 Ubuntu 22.04/24.04 或 Windows 10/11、至少 32 GB RAM 和 16 GB VRAM。当前工程已新增 `run_linux.sh` 和 `linux_migration_check.py`，并把 Revo3 USD、mujoco_cube 和循环训练的运行路径改成环境变量/平台默认。Linux 上需要把 `ISAACLAB_PYTHON` 指向能导入 Isaac Lab/Isaac Sim 的 Python，并准备相邻的 `revotron_revo3` 与 `isaac_mujoco_cube` 资产目录；本机只能完成静态迁移检查，未实际启动 Linux Isaac Sim。

本轮新增证据：绝对目标 PPO round14 完成 3×600 步更新但训练 0/3，独立评估 0/1 且桌面冲量约 30.5 N·s；修复阶段周期后的 round15 纯策略为 0/1、桌面冲量约 4.48 N·s。阶段特征已改为按示范实际 391 步归一化。open-loop phase round16 的离线拟合更差，因此没有进入闭环评估。

版本注意：当前工作区使用 Isaac Sim 5.1 Python 环境。迁移 Linux 时必须保持 Isaac Sim、Isaac Lab、Python 和 USD/PhysX 资产版本匹配；如果采用官方当前版本，应先做版本升级后的 smoke、传统控制和奖励回归，不能直接混用旧 checkpoint。

### 最新可复现视觉教师（2026-11-01）

新增 `phase_knn_eval.py`，将胸部 RGB 只压缩为魔方位置/尺度的 6 维特征，锁存 episode 初始位置，再附加归一化阶段进度，从成功的 `base:*` 示范中按近邻检索 56 维教师动作。`runs/phase_knn_round19/teacher_manifest.json` 汇总了四个独立种子 1929、1933、1937、1941：四次均双侧接触、离桌超过 8 cm、保持 30 个控制步、魔方-桌面冲量为 0、机器人-桌面冲量为 0。这个结果验证了“视觉确定位置 + 传统 56 关节控制”的抓起链路，但它是非参数视觉教师，不代表神经网络或 PPO 已学会。

`phase_actor_round20` 尝试把该教师蒸馏成 7 输入 MLP；离线误差下降，但独立闭环为 0/1，说明当前动作序列对阶段/位置的映射存在近邻式不连续性。后续应先收集 kNN teacher 的成功 rollout，再做带记忆或 RBF/分段策略蒸馏，随后才继续 PPO；在此之前不应把 MLP checkpoint 当作可用抓取策略。

完整的软件链路、观测合同、控制频率、奖励、验证门槛、训练闭环和 Linux/ROS2 迁移边界见 `FRAMEWORK_CHAIN.md`。

新增 `state_phase_knn_eval.py`：在视觉位置+phase 检索上加入当前 `q/qdot/上一执行动作/target-q`，seed=1945 的独立纯闭环成功，抬升 11.28 cm、保持 30 步、机器人-桌面冲量为 0。它是下一轮时序/RBF 蒸馏的 teacher 候选，仍不是神经 PPO 策略。

2026-11-02 状态策略训练入口与真实独立测试结果见 STATE_TRAINING.txt。
