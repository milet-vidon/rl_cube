# 11601 抓持开发回合（2026-11-07）

固定开发初态 11601，任务从 reset 到整块倾转后的腕部保持。保留 56 维动作接口、接触/速度/姿态门槛、240 Hz 物理和 30 Hz 控制。通过标准仍包含连续 90 帧五指合格接触和角速度低于 0.20 rad/s。

| 本机 run（均在 revotron_cube_rl/runs 下） | 结果 |
| --- | --- |
| `contact_hold_gain30_d1_11601_20261106_01` | 保留全部左指刚度 30，食指接触 301/301；角速度合格 0/301 |
| `contact_hold_pos64_11601_20261107_115644_564860301` | 位置求解迭代改为 64，五指均接触 301/301；角速度合格 192/301，最长联合保持 0.33 秒，未准入 |
| `contact_hold_pair15_pos64_11601_20261107_120008_511634810` | 同样 64 次位置迭代，保持入口用 30 步将拇指和小指刚度从 30 降到 15，食指、中指和无名指保持 30，阻尼保持 1；1290 步，五指接触 106/106，最长合格保持 3.03 秒，通过原验收与离线复核 |

在 `revotron_cube_rl` 目录中执行：

```bash
bash run_contact_hold_training.sh runs/contact_hold_pair15_pos64_11601_20261107_120008_511634810
```

这是新建的状态 BC 模型：198 个状态/阶段特征预测左侧 28 个实际施加的关节目标。完整训练使用 cuda:0、20,000 次更新。两次更新的 CUDA 检查和 CPU 权重重载已在 `runs/contact_hold_gpu_smoke_11601_20261107_120336_200872562` 完成。训练启动器打印本次输出目录；以其中的 result.json 和 best.pt 为准。

要从头复现实验并在通过门槛后开始训练，可运行 `bash run_contact_hold_cycle.sh 11601`。单次采集的成功只属于 teacher 和这个已见开发初态。该模型不兼容 V11 checkpoint；学生的闭环表现和未见种子的泛化仍待评估。

正式拟合 `runs/contact_hold_bc_20261107_143712_491446697` 完成 20,000 次更新，选中第 19,200 次的 best.pt；CPU 重载在同示范的 92 个留出样本上 RMSE 为 0.00036111 rad。

```bash
bash run_contact_hold_evaluation.sh runs/contact_hold_bc_20261107_143712_491446697 11601
```

评估从 reset 开始使用学生的左侧 28 个位置目标，右侧、公开阶段和驱动增益仍由 teacher 提供。首个 reset 动作未出现在训练标签中，仍由学生执行。输入来自执行前的状态和截至当前的阶段时钟。评估结果写入独立的 `runs/contact_hold_student_eval_*` 目录；读取 `result.json` 的 `dynamic_grasp_passed`、`wrist.accepted`、`joint_wrist_hold_s` 和 `failure`。该目录没有 teacher 示范准入文件。

已见 11601 实测：`contact_hold_student_eval_11601_20261107_150520_007040098` 在 498 步以 `contact_topology_or_settle_timeout` 结束，未通过抓持。最后停在 preshape，此阶段只要求手臂到位和速度收敛。原控制器判定时，J3_L 位置误差约 0.161 rad、手臂最大速度约 0.362 rad/s；该回合使用 opposition_grasp_v20.json，位置误差与速度门槛分别为 <0.020 rad、<0.18 rad/s（代码缺省 0.035 未生效）。

`prepare_contact_hold_arm_corrections.py` 从这次失败评估中导出 192 条候选：步骤 305–496 的实际学生前状态与 teacher 左臂七关节请求，经原安全投影计算的目标。先对学生实际施加的动作做投影重放检查，排除末次超时指令。它们不是成功示范，也不是已执行的 teacher 动作。数据和哈希在 `work/teacher_optimization_20261103/contact_hold_preshape_candidates_11601_20261107_150520`，需要单独审核和支持七关节纠正监督的训练入口。

纠正重训入口会核对这份候选清单及原始回合哈希，重放原投影，并从执行前状态独立重建输入。第 305–496 步中的 162 步用于训练，另 30 步用于同一回合内的验证。只监督 Joint1_L 至 Joint7_L；原成功示范仍监督全部 28 个左侧目标。优化器从头开始，模型与归一化从上次 best.pt 加载。选取两个验证集加权标准化均方误差最低的权重（包括第 0 步的原模型）。两个验证集都不能证明闭环成功。

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
fit="runs/contact_hold_corrected_$(date +%Y%m%d_%H%M%S_%N)"
CONTACT_HOLD_BATCH_SIZE=65536 CONTACT_HOLD_UPDATES=20000 bash run_contact_hold_corrected.sh "$fit" && bash run_contact_hold_evaluation.sh "$fit" 11601
```

正式参数是每次更新各采样 32768 个示范和纠正样本（有放回），学习率 0.0001，纠正权重 1。若日志只出现 BC_PROGRESS，还没完成重载检查；TRAIN_COMPLETE 后由上面的命令自动从 reset 复测 11601。以评估输出的 RESULT 为准。

第二次学生闭环评估为 `contact_hold_student_eval_11601_20261107_153509_639692462`。源模型完成 20000 次更新，留出误差选中第 400 步。ready 用时从 305 步缩短至 247 步；preshape 仍为 193 步，以原超时结束。按上一行记录的实测 q、关节速度对齐当前行的 nominal_target，比对双臂 14 个关节：相位内第 72–192 步共 121 次判定，109 次速度过门，0 次位置过门，最大关节误差的最小值为 0.021409 rad，最后一次为 0.03260 rad。该结果不计入成功示范。

第二轮聚合入口：`bash run_contact_hold_corrected.sh "$fit" --round2`，`fit` 必须指向新的 run 目录。保留 1197/92 条原示范训练/留出样本，并分别对两次实际学生分布重放过的七关节投影目标训练（各 162/30）。新轨迹取步骤 247–438；第 439 步超时，不入库。仅用于同 seed 开发。默认 2000 次更新，每 50 次比较留出指标，保留起始权重作为基线；如果没有同时改善加权指标与新分布 RMSE，启动器跳过重复物理评估。拟合后在相同 11601 初态运行原闭环评估，确认是否能从 preshape 进入 lower，最终仍看完整抓持门槛。

### 连续训练入口

在 `/home/brainco/rl_cube/github_project/revotron_cube_rl` 下启动：

```bash
bash run_contact_hold_continuous.sh --run-dir runs/contact_hold_auto --max-rounds 30
bash run_contact_hold_continuous.sh --run-dir runs/contact_hold_auto --status
# 中断后使用同一路径
bash run_contact_hold_continuous.sh --run-dir runs/contact_hold_auto --resume
```

第一条命令启动整个流程，后两条分别供另一个终端查看进度、以及退出后恢复。已有目录不能用于新建。默认从已实测到 squeeze 的 round2 第 150 步权重开始。这个旧 checkpoint 不含 AdamW 状态，因此第一次新建优化器，之后每轮衔接模型、AdamW、更新计数和采样器状态。每轮默认 500 次更新、batch 65536、学习率 0.00005；`--updates`、`--batch-size`、`--lr` 可在新建时指定。这里一轮是一次采集、一次拟合和一组物理验证，不是一次梯度更新。

启动时先在固定验证集实测起点。随后交替使用 11601/11603 进行学生控制的采集，每轮一个 seed。每一步先从实际执行前状态计算输入；左臂与手指的 teacher 请求保留为反事实标签，经该回合 PhysX 读回的关节范围、目标步长和速度上限投影，重放实际学生动作以校验边界。取最后两个到达阶段，排除首步、末步和危险状态以后的记录，逐关节排除越界标签。失败回合永远不记为成功示范。历史成功演示、两批已审查的左臂标签和累计的新样本均保留。GPU 拟合与 Isaac 实测串行，GPU 被别的进程占用会等待。

每轮保留最后权重继续训练；单独以验证集完整抓持成功数、安全回合数、到达阶段和保持时间排序，保留闭环表现最好的模型。默认 21611/21612 两个固定验证 seed，连续两组全部成功后冻结最佳模型，才运行一次固定的三个留出 seed（32611/32612/32613）。该结果是这三个初态上的表现。若留出失败，不反馈进训练。也可因连续 6 轮验证评分没进展而停止；到总轮数上限的任务可用 `--resume --max-rounds 60` 扩大总上限。目标比率、连续次数和停滞轮数分别由 `--target`、`--confirmations` 和 `--patience` 配置。另起实验前指定未使用的留出种子，不能把已看过的验收结果再当新测试。

`state.json` 为当前轮及 champion，`round_NNNN/fit/last.pt` 为最近优化器断点（第 1 步、每 50 步、最后一步保存），`round_NNNN/fit/best.pt` 为兼容现有评估器的该轮末模型；实际跨轮最优以 state.json 的 champion 为准。`round_NNNN/validation/panel.json` 包含每个 seed 的阶段、失败原因、保持秒数、checkpoint 和完整 Isaac 证据路径。中断时只从上个已保存更新继续；已完成并通过哈希检查的实测回合会复用，未完成回合会换新目录重跑。状态 `plateau`、`heldout_target_not_met` 均不是成功，`confirmed_on_test` 表示达到了配置的留出标准。

### 多种子长训（2026-11-10）

此前多种子采集中的 11603、11604 均在 tilt_hold 超时，11605 完整成功。追加的 11606 也完成了完整抓持；与原 11601 一起，各有 3.033 s 通过腕部门槛。三条完整示范分别重新验收。运行新的入口：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_long_gpu.sh --run-dir runs/contact_hold_long_v1 --rounds 40 --updates 5000 --batch-size 32768
```

首次新建目录，从 round2 第 150 步的已评估权重开始，重建优化器；后续恢复优化器和采样器。开局还会核验上两次开发训练的 16 条 collection 回合，用其逐关节投影标签（包括失败状态；不作为成功示范）。每轮再采一个开发回合，以每个 batch 各半的示范/纠正数据做 CUDA 更新；每 250 步保存一次，最后在 21611、21612 从 reset 做闭环验证。比较每轮的完整成功数、安全回合数、到达阶段、保持时间，默认 12 轮没有验证改进就停止。两轮验证全过后冻结最佳模型，在预留 32621–32623 上做最终评估；不将验证或留出数据作为训练样本。训练上限指闭环轮数和每轮优化次数，非并行 Isaac 环境数。

中断后用 `bash run_contact_hold_long_gpu.sh --run-dir runs/contact_hold_long_v1 --resume`。参数与源码在创建时冻结，已完成的验证可复用。查看 run 下的 `state.json`（champion、最后完成轮次及验证成功率）和 `round_NNN/validation/panel.json`（阶段和失败原因）。状态为 plateau 或留出未达标均不表示成功。GPU 计算集中在拟合段；单环境 Isaac 采集和验收仍然串行。

### 魔方 GPU 多环境长训（2026-11-10）

上一个 `contact_hold_long_v1` 已到 40 轮上限，200,000 次更新。固定验证集所有轮次都是 0/2；第 39 轮按安全、到达阶段、合格保持时间排序最佳。21611 起抬 10.6 cm 后失握（起抬阶段无名指仅 2/60 帧有合格接触）；21612 最高 0.4 cm，在抓紧阶段超时（食指 6/193 帧）。两条均无连续 3 秒合格保持。

从上述最佳轮的权重、AdamW 状态与随机数状态继续。一次 Isaac 进程在 cuda:0 上并行跑四个独立桌面，各有一台 56 关节机器人和一颗 27 刚体自由魔方。步频 240 Hz、控制频率 30 Hz。各环境独立的初态、接触、终止和关节目标均按环境编号对应。每轮同步开始一批新 seed；提前终止的环境不再供样，下轮再重置。默认四条分别从 reset、抓紧、reset、起抬起由学生控制左侧 28 个关节，接管前由同一原版 teacher 执行。已接管后的安全状态可用于训练；失败回合、teacher 带入的状态都不能当作完整成功示范。若能走到转腕或保持，继续采到原 controller 的终止条件，最多 3500 步。到达控制器保持条件本身也不是完整成功证据。

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_gpu_parallel.sh --run-dir runs/contact_hold_parallel_v1 \
  --num-envs 4 --rounds 60 --updates 5000 --batch-size 32768 --patience 20
```

每轮训练 5000 次，最多新增 300,000 次更新；每 250 步存一个优化器断点。batch 中一半取三条已审查的成功示范，四分之一取原来的纠正数据，四分之一取累计的并行学生采样；有样本时后者的一半专取抓紧和起抬。物理步进和学生推理在 GPU，teacher IK 和接触归类仍用 CPU。此入口采用有 teacher 纠正的模仿学习，不运行 Lite3 的 PPO。验收仍在原单环境运行：每一轮对 21611/21612 做从 reset 起的完整验证；连续两轮 2/2 后才冻结最佳模型验收 51611–51613。连续 20 轮验证评分没提高会停；成功率不能用显存、训练损失、采样条数或单次抬升代替。

```bash
# 新终端，在同一个项目目录；中断后重启只需要最后一行。
bash run_contact_hold_gpu_parallel.sh --run-dir runs/contact_hold_parallel_v1 --status
watch -n 1 nvidia-smi
bash run_contact_hold_gpu_parallel.sh --run-dir runs/contact_hold_parallel_v1 --resume
```

当前轮的 `samples_*/report.json` 记录每个环境的阶段、终止原因和有效标签数，`validation/panel.json` 记录独立于采集的两个固定验证 seed、失败原因与保持秒数。`state.json` 的 `current` 是续训权重目录，`champion` 是闭环得分最高的权重目录。监控要区分并行物理、网络拟合、旧版单环境验收三个阶段；无需把显存占满。

本机短测 `runs/contact_hold_parallel_smoke_20261110_r1`：四个采集 seed 41000–41003 产生 1214 条标签，含 384 条抓紧、47 条起抬。41003 在起抬失握；另外三个在抓紧或准备阶段超时。跨环境接触为 0。采样共 335.6 秒；在其中约 105 秒每三秒采一次 GPU 读数，36 次的利用率为 89–97%（中位 95%），设备总显存占用为 522/8151 MiB。详细数据在该目录的 `gpu_during_collection.csv`。用 batch 32768 续训两次，从第 195000 步到 195002，checkpoint 成功重载。本轮用于检查管线与采样覆盖，不能证明策略已学会转腕保持。

短测原版完整验收仍为 0/2：21611 起抬后失握（全回合最大高度增量 3.7 mm），21612 抓紧超时（2.1 mm），两条都没有合格保持。旧第 39 轮仍为 champion。独立的双环境真机 PhysX 检查确认转腕阶段使用的手指刚度、阻尼只写进指定环境（`lane_gain_probe.log`）；这不是学生实际走通了转腕阶段。

**高显存配置。** 本机 8151 MiB RTX 5070 Laptop 实测，原并行 run 在四环境采样时设备占用 522 MiB，batch 32768 拟合时 568 MiB。用同一个 fit plan、模型和优化器，在独立临时目录做 3 次真实反向更新并重载断点，batch 262144、679936、712704 的设备峰值分别为 2818、6958、7282 MiB（含缓存和 CUDA 上下文）。最后一档活跃 PyTorch 张量峰值 5609 MiB；实际长训占用会随数据累计改变。短测 16 个物理环境、12 个控制步的设备峰值为 590 MiB。数据见 `runs/contact_hold_vram_probe_20261110_a/calibration.json`、`runs/contact_hold_16env_mem_check_20261110/short_probe.json`。

要切换，等原训练至少有一个完整验证轮次，原终端 Ctrl+C，执行：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_high_vram.sh --from-run runs/contact_hold_parallel_v1 \
  --run-dir runs/contact_hold_parallel_vram_v1 --num-envs 16 \
  --calibration runs/contact_hold_vram_probe_20261110_a/calibration.json
```

脚本核对已有物理验证、每轮模型和优化器、采样数据及显卡型号，继承最近**完整验证轮**，按源 run 的总轮数、每轮更新数继续；只改新分支的并行环境数和 batch 并重置停滞/连续达标计数。原目录保留，当前未完成轮次不计入继承。加 `--preview` 可在原任务运行中核验将继承的断点，不启动训练。新分支断线后用 `bash run_contact_hold_gpu_parallel.sh --run-dir runs/contact_hold_parallel_vram_v1 --resume`。更大的 batch 从同一批样本有放回抽取，每步计算量增加；仍以完整闭环成功率判断效果。采样与单环境验收不是大 batch 拟合，不会维持相同显存占用。

### 采样也使用大规模并行环境

`run_contact_hold_wide.sh` 只从最近**已完成原版双种子验证**的轮次建立新分支。源任务仍在运行时可预览；实际切换前需停止原终端。旧采集器、旧模型与原验收器的源码哈希保持不变。新增环境使用 100 万以上独立开发 seed；不同轮次使用互不相交的段，不进入最终留出集。

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_wide.sh --from-run runs/contact_hold_parallel_vram_v1 \
  --run-dir runs/contact_hold_wide_v1 --num-envs auto --rounds 60 --updates 5000
```

`auto` 读取本机 `runs/contact_hold_wide_qual_a/qualified.json`，核对显卡、探测代码、实际采样证据和内存余量。显存包含机器人、自由魔方、真实接触传感器和 PhysX 缓冲区。没有申请闲置张量凑占用。一次原型解析后由 PhysX 原生克隆环境；不同环境有独立碰撞 ID、目标、seed、接触和终止状态。按 240 Hz 读取两组批量接触张量，30 Hz 执行控制。每条环境根据自己的实际 reset 姿态计算 IK，CPU 工作者不启动 Isaac，也不复制 GPU 仿真。规划阶段和原版单环境验收不代表 GPU 物理步进吞吐。

每轮保留验证/训练、抓紧起抬/其他阶段四个互不混合的纠正数据分组，每组最多 8192 行；拟合时重放池每组最多 16384 行。这样积累几十轮不会把所有历史观测无上限放入显存。沿用源模型、优化器、batch、每轮原版验证及最终留出种子；阶段失败仍不算成功示范。每轮 `collection_NNNN/samples_*.resources.json` 记录实际资源读数。若剩余显存少于 512 MiB、系统可用内存少于 4 GiB，或驱动报告接触缓冲区不足，整批未通过的采样不准入，改用下一档已验证的较小环境数。其余错误显式停止。

```bash
# 另开终端查看状态和实际正在用的环境数；中断后使用同一运行目录。
bash run_contact_hold_wide.sh --run-dir runs/contact_hold_wide_v1 --status
bash run_contact_hold_wide.sh --run-dir runs/contact_hold_wide_v1 --resume
```

本机容量实验见 `runs/contact_hold_wide_capacity_a/trials.json`。下面是 CUDA 物理与冻结学生推理的 12 步短测（不含 teacher IK）；吞吐是所有环境合计的控制步/秒：

| 环境数 | 设备显存峰值 MiB | 环境步/秒 |
| ---: | ---: | ---: |
| 128 | 996 | 166 |
| 256 | 1462 | 248 |
| 512 | 2524 | 380 |
| 768 | 3386 | 440 |
| 1024 | 4966 | 467 |
| 1280 | 6158 | 469 |
| 1536 | 7364 | 467 |

128 环境的一次早期配置报告 `foundLostAggregatePairsCapacity` 不足，可能丢失碰撞，已排除；表中均为修正后的实际测量。不同显卡或修改场景后，可使用同一环境中的 `contact_hold_wide_capacity.py --output NEW --envs ...` 与 `contact_hold_wide_qualify.py --from-run SOURCE --capacity NEW --output OTHER_NEW` 重新测量并通过 `--qualification OTHER_NEW/qualified.json` 指定收据。

正式采样器的 1536 环境实测（`runs/contact_hold_wide_qual_a/qualified.json`）：8 个 CPU 工作者为不同物理初态完成初始 IK，用时 392.3 s。随后 12 个控制步（18,432 个环境步）用时 42.3 s，约 436 环境步/s，产生 7680 条非首末步、学生已接管的有效纠正行，跨环境接触 0。含初始化的峰值为 7370/8151 MiB，测试期间最低系统可用内存约 10.2 GiB。步进阶段 `n1536/gpu_during_steps.csv` 共有 38 次读数：设备占用 7368–7370 MiB，GPU 利用率中位数 79.5%、最高 100%。这是从 reset 出发的短测，不是 1536 环境完整抓持成功率。程序会在主终端显示 `PLAN_LANES`、`VECTOR_STEP`、`GPU_FIT`，分别对应 CPU 规划、物理步进与优化更新。

另用 8 条环境从 reset 一直跑到真实终止，得到 2875 条标签，含 1152 条抓紧和 31 条起抬，两个阶段的这些行都实际测到手指接触。6 条超时、2 条起抬失握，未将其作为成功示范；跨环境接触为 0。新重放池连同旧示范/纠正数据，用 batch 712704 从累计第 200000 步更新到 200002，优化器未重置，新断点成功重载，拟合峰值 7282 MiB。详见 `runs/contact_hold_wide_contact_smoke_a` 及 `docs/evidence/native_cube_sampling_1536_20261110` 的精简记录。

### 从 1536 环境的 PhysX 分配失败恢复

原 `runs/contact_hold_wide_v1/round_002` 在分配 415,576,064 字节时得到 CUDA Result 2，随后原生日志报告场景损坏。脚本写出了 16,384 行，但资源收据不合格，整批作废。控制器未识别该错误的英文措辞，因此也没有降并行度。最后一次通过数据、模型、优化器和闭环验证记录核验的是 `contact_hold_parallel_v1/round_001/fit`，累计 200,000 次更新，固定验证 0/2；不能把数据量、训练 loss 或开发集 teacher 查询当作腕门成功。

新版保留 27 个自由刚体的魔方、原机器人及关节顺序、240 Hz 物理/30 Hz 控制、原接触与安全门槛。按 64 个环境分组创建原生复制对象，减少创建时瞬时重叠的包围盒对。环境仍同时推进。GPU 批量计算复位 IK、轨迹时间尺度、接触冲量及 teacher 反馈（包括指尖和转腕修正）。首次遇到新 actor 时仍需从 PhysX 解析路径；CPU 还负责文件读写、冷启动与 API 要求的少量元数据。

收集过程中持续读原生日志，首次错误就停止 worker。不传递可能损坏的标签。确认显存不足时减小并行宽度，重跑未接受的种子，直至完成本轮全部 1536 个不同的开发种子；其余物理错误仍显式报错。已经接受的波在重启时复用。分波的合并按四个分层的原候选数进行抽样，避免把样本很少的一波人为放大。

速度基准：相同模型 `700733c3…`、相同开发种子 7000000–7001535、1536 环境、12 个控制步。参见 `runs/contact_hold_cuda_bench_a/comparison.json`。规划+teacher 初始化 392.318 → 5.644 秒；步进报告 42.275 → 35.594 秒（436 → 518 环境步/秒；旧计时含样本压缩写盘，新计时在压缩前结束）；整个进程 526.027 → 135.642 秒，板卡显存峰值 7370 → 5410 MiB。7680 行按 lane、step 对齐，phase/val/mask 完全相同，teacher 标签最大差 3.58e-7 rad。这里只测了 ready 阶段；不能将 69.5 倍的规划加速说成全训练加速。

独立证据：`runs/contact_hold_cuda_bench_a/isolated_ik.json`（1536 组 IK；含不可达姿态拒绝测试），`runs/cuda_scene_commission/g64_1536b.json`（同帧接触汇总 66.83 → 1.66 ms），`runs/cuda_teacher_trace_11601_v3.json`（冻结的实测轨迹回放，10 阶段、32 次增益写入、接受步一致，最大目标差 3.10e-6 rad）。

本机生产入口是 `run_contact_hold_cuda_long.sh`。为验证旧分配错误，先在 `runs/contact_hold_cuda_v1` 用原失败种子重采第 2 轮；该目录是采样验收记录。生产目录 `runs/contact_hold_cuda_long_v1` 使用这些完整、通过原生日志审计的 wave 收据，不从坏批次或测速用 12 步数据恢复。首次创建命令（已有目录应执行下一条续训命令）：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_cuda_long.sh --from-run runs/contact_hold_wide_v1 \
  --reuse-prepared runs/contact_hold_cuda_v1 --run-dir runs/contact_hold_cuda_long_v1 \
  --num-envs 1536 --rounds 60 --updates 5000 --prepare
```

这里的 `--prepare` 只验收并准备下一轮的拟合计划；移除此参数会继续拟合和闭环验证。已有生产目录，启动或中断后恢复：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_contact_hold_cuda_long.sh --run-dir runs/contact_hold_cuda_long_v1 --resume
```

另一个终端查看进展：`watch -n 5 python3 contact_hold_cuda_status.py runs/contact_hold_cuda_long_v1 --gpu`。全局 200000 次更新的模型与 AdamW 状态逐参数匹配。每 250 个拟合步保存一次，完成 5000 步后再运行原固定验证面板。连续 20 轮无改善可提前停止；不能保证跑完 60 轮或保证收敛。

第 2 轮实际验收（原失败种子 1004096–1005631）：一次启动全部 1536 环境，无降并行、无 PhysX 错误、无跨环境接触。全程 744 个控制步，物理采样 46.4 分钟，设备峰值 7010 MiB。1317 条到达夹紧，461 条到达抬升，均未到腕门；终止为 1071 次接触/稳定超时、461 次失握、3 次机器人碰桌、1 次硬限位。四层均有候选，总共保留 32768 条纠正标签。见 `runs/contact_hold_cuda_v1/round_002/actual_collection_summary.json`。夹紧/抬升存档的同工况反馈微基准（12 组连续状态铺至 1536 lanes，3 步，首步 1024 次指尖拟合）：CPU 7.266 秒，GPU 中位 0.02073 秒，约 350.5 倍；此处最大关节目标差为零，不含物理。见 `runs/cuda_finger_feedback_bench_v3/result.json`。正式续训目录为 `ready_to_fit 2`，另用 batch 712704 在隔离目录 `runs/contact_hold_cuda_fit_check_v1` 完成 200000→200002 两步，验证检查点重载，实测峰值 7314 MiB。


### 2026-11-11：按物理结果评价“拿稳”

用户关心能否拿稳，不限定四指、五指或特定指头。第 15 轮独立诊断保留原固定种子与冻结模型，并单独评价左手（含掌）实际支撑、离桌 ≥8 cm、完整安全、线速 <0.08 m/s、角速 <0.6 rad/s 连续 3 秒。运动中不因速度高直接判失，另报连续离桌支撑时间、掌内滑移及最后是否回桌。“曾稳持 3 秒”与“结束时仍稳持”分开；不把此诊断当成原转腕、倾转或求解任务成绩。

旧 10/20 帧对照继续。新增 matched_lift 保留原夹紧、在抬升起去掉手指数门；from_reset 从夹紧起使用真实左手总接触。均保留学生动作、手指反馈和安全约束，不写入正在运行的长训。两模式分别试 21611、21612。检查首个物理分歧；若旧判停前已经分歧，只作独立同种子结果，不能称作同一状态的后续。

历史记录：`runs/contact_hold_functional_history_20261111.json`，16 个轮次各两个固定种子，0/32 观察到连续稳持 3 秒；最高为第 5 轮 21611 的连续 1.67 秒稳持。原先按接触数提前截断的轨迹无法证明之后能否重新抓稳。
