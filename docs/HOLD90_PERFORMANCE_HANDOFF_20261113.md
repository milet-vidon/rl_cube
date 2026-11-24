# 90 mm 魔方训练、GPU 计时与 CubeLab 交接

更新日期 2026-11-15，时区 Asia/Shanghai。请在 `brainco` 的部署目录先读本页，再查实际进程、最新 worker 日志和结果文件。下面记录的 PID、步数和 GPU 读数是交接时的快照，机器重启后以当前状态为准。

## 当前任务与运行中的三个目录

远端部署项目为 `/home/brainco/rl_cube/github_project`。其 `revotron_cube_rl/runs` 下有三种不同用途的运行：

| 目录名 | 用途和状态来源 |
| --- | --- |
| `contact_hold90_long_v2` | 原始 90 mm 连续训练。第 7 轮已采样完成。之前看到的 `stage=fit` 和 `WAIT_GPU` 表示拟合排队。 |
| `contact_hold90_refill_3072_r5_20261113` | 生效中的首轮回填采样。1,536 物理槽，3,072 个逻辑回合，从原第 5 轮模型出发。启动时运行的是 `161ae94`，能提供槽位和回填进度。这个 worker 自身没有函数秒表。 |
| `contact_hold90_turn_probe_20261113` | 本次单次诊断，非训练。使用 1,536 个物理槽，原第 6 轮的全部 1,536 个采集种子再加 64 个种子；每条最多 840 步，覆盖历史上集中的举升转腕交接。先测真实采样，再比较固定输入下的路径求解分批宽度。12:58 已建立 run、持有自身 campaign 锁并等待 GPU。 |

首轮回填和诊断的父检查点相同：`runs/contact_hold90_long_v2/round_005/fit/best.pt`，SHA256 `3966ee6485b45892dd2145891d441fe71884fe7a15bf75e855026d971f22c418`。诊断沿用第 6 轮原计划、teacher、场景、握姿和安全阈值。用于定时的 840 步上限与标准训练 3,500 步不同，诊断数据不会进入训练集，也不产生新模型。正式求解批宽仍为 32。

## 为什么做这次修改

12:32 的 31 个一秒读数显示 GPU 利用率平均 57.7%，中位数 39%。独立 24 秒窗口中，GPU 低谷时采样主线程占用约 0.98 核，高段约 0.10 核。过第 720 步后，另两个窗口的平均利用率为 90.6% 和 97.5%。历史第 6 轮的 600–719 步出现 1,345 次可重建的举升交接，按原实现拆为 105 批路径 IK；720–839 步仅 16 次。这些观测支持重点检查转腕规划，同时还要测 PhysX 和回填。它们不能给出各函数实际占比。

## 源码位置和变更

- 已在用的回填版工作树：`/home/brainco/rl_cube/work/hold90_opt_src_20261113`，分支 `perf/hold90-refill-20261113`，提交 `161ae94`。实现索引重置、独立回合时钟、先筛选再回传，以及有效吞吐统计。之前已在 GPU 上用 2 槽执行 8 条轨迹、6 次重置，并通过原训练器更新和重载。结果在 `/home/brainco/rl_cube/work/hold90_refill_verify_20261113_a/status.json` 和 `cuda_bank.json`。
- 新计时工作树：`/home/brainco/rl_cube/work/hold90_timed_src_20261113`，分支 `perf/hold90-timing-dashboard-20261113`，代码提交 `3611649a7cdc1c6e0a43569bba34121dc6a2cba5`。实现文件在其 `revotron_cube_rl/throughput90`。这两个工作树都属于远端的 `rl-cube-source` Git 仓库；实际训练读取的资产、门禁和原模块来自 `github_project/revotron_cube_rl`。不要将带未提交工作的整个部署仓库重置成工作树。
- `contact_hold90_refill_sample.py` 增加实例级函数计时、交接组大小直方图、周期进度快照、可选的 `--capture-turn`。启用捕捉时记录 teacher 当时的原始 56 维 nominal、Core 姿态、原始组边界、槽位、种子和控制步；已保存的旧训练 NPZ 只有投影标签，不能代替这组输入。
- `contact_hold90_long.py` 把采集选项和源码 SHA 传给对应 worker；`run_hold90_timing_probe.py` 由原项目的两把 GPU/Isaac 锁串行调度诊断与离线比较。一次失败写入该诊断的 `state.json.last_error`。
- `hold90_turn_path_benchmark.py` 在同一台卡上用原 `TurnIK.path` 比较 32/64/128 三种宽度。诊断各配置重复计时 3 次，同时保存接受标志、有效段长度和轨迹误差、显存峰值、采样输入和源哈希。实际组边界下出现相同分块的配置跳过重复测量；只在显式的跨步灵敏度分析里拼接小组。
- 已经部署的前端包是 `/home/brainco/rl_cube/github_project/revotron_cube_rl/cube_dashboard`。修改 `performance.py`、`training.py`、`server.py`、`static/index.html`、`static/app.js` 和 `static/app.css`。它原本就是现场独立的未跟踪目录。原包、已部署候选、测试、校验表和部署回执在 `/home/brainco/rl_cube/work/cube_dashboard_perf_20261113_1251`。上述六个文件的完整改动在计时分支的 `docs/evidence/hold90-dashboard-20261113.patch`。已用部署前的原件在临时目录应用该补丁并核对六个文件的 SHA256。

## 页面和数据契约

### 2026-11-15 模型对比：同一验证种子有哪些进步与退步

[模型对比](http://127.0.0.1:8765/#compare) 取代了「结束原因明细」。旧书签 `#outcomes` 会打开新页面；训练首页与 [训练趋势](http://127.0.0.1:8765/#trends) 保留原有曲线。已打开的旧标签页需先按 Ctrl+R 整页刷新。

在同一实验内选参考轮和候选轮。指标包括三项同达、结束仍稳持、整颗转动和到达终点仍稳持。比较的是同一验证种子在两轮中的表现，可以分别筛出新增通过、由通过变未过、两轮都通过、两轮都未过。共同种子是比较的分母；只出现在一轮的种子不算失败。这里的“进退”描述这组仿真验证结果，不等于对全新初态的泛化，也不能单凭某个种子的一次差异认定原因。

验收样例：`contact_hold90_long_v2` R17 → R20，共同种子 32。“三项同达”从 29/32 变为 25/32：新增 1、退步 5、均通过 24、均未过 2；相同分母下净少 4。seed 20000004 两次均曾整颗转动、结束仍稳持，R20 又满足腕部终点条件，因此是新增通过。切换到“结束仍稳持”时应显示新增 0、退步 2、均通过 29、均未过 1，不应沿用上一指标的分类。

R21 在本次交付期间完成验证。相对 R17，32 个共同种子的整颗转动从 31 到 32（三项同达从 29 到 24）。这说明单项提高与联合作业完成率降低可以同时发生；点击「退步」可直接查看哪些种子、哪些条件发生了变化。

发布记录、上线前备份、Firefox 验收及可重放补丁在 `/home/brainco/rl_cube/work/cube_dashboard_compare_20261115`；计时工作树内的 `docs/evidence/hold90-dashboard-compare-20261115.patch` 接在训练趋势补丁之后应用。

### 2026-11-15 训练收敛与继续训练的判断

打开 [训练监控](http://127.0.0.1:8765/#training) 或 [训练收敛趋势](http://127.0.0.1:8765/#trends)，选择 `contact_hold90_long_v2`。部署后已打开的旧标签页需整页重新载入。看跨轮结果时，按同一轮对齐以下三类证据：

- **拟合**：所选检查点与轮末的留出示范动作误差。长训每轮重采样并滚动回放最近若干轮；不同轮次的留出样本会变化。MSE 降低表示对相应示范动作拟合更好，不能单独认定仿真任务达标。
- **参数**：选中 actor 的可训练权重相对于本轮实际父 actor 的变化，百分比为 `100 × ‖θ_selected − θ_parent‖₂ / ‖θ_parent‖₂`。不含优化器和固定缓冲区；不能从学习率或检查点哈希是否相同推算。没有可核实的结果就留空。相对变化没有普适的收敛阈值。
- **任务结果**：独立于梯度更新的学生闭环验证。曾拿稳、结束仍稳持和腕端仍稳持应分开看；「腕端仍稳持」要求同一回合同时满足两个条件。参与轮次间反复比较的固定验证种子，不能再作为新种子的泛化测试。未验收的轮次不计作 0 分。

可复核示例（固定截点 R20，并非永远使用该窗口）：R11–15 与 R16–20 的腕端仍稳持率均值 70.0% → 82.5%，选中检查点的平均留出 MSE 0.002095 → 0.001502，相对父模型的平均权重变化 4.85% → 3.54%。R16–20 单轮腕端仍稳持计数依次为 24、29、28、26、25（每轮 32 个）。R17 是这 20 轮中腕端仍稳持最多的轮次；原训练器选出的 R14 使用另一套排序标准。进步与波动可以同时存在，最后一轮也不一定是最适合部署的模型。

原 v2 是分阶段均衡采样的 DAgger 行为克隆，直接优化示范动作误差。结束时仍稳持和到达腕端不是该 run 的逐步奖励。若相邻窗口反复停滞，先按具体任务指标选候选；若要检验目标权重是否合适，应从同一父模型另开对照，保持数据、训练预算和评测方式可比，一次只改一个因素，再用未参与选择的种子验收。不要据趋势自动停止训练、修改原 run 或放宽门槛。

本版 `GET /api/quality` 额外返回每轮拟合、经验证的三项同达数和参数变化。权重仅在 CPU 上后台计算并缓存；首次打开时可能需等一次页面轮询。发布备份、可重放补丁、文件哈希与 Firefox 验收记录在 `/home/brainco/rl_cube/work/cube_dashboard_convergence_20261115`。计时工作树中的 `docs/evidence/hold90-dashboard-convergence-20261115.patch` 接在 11 月 14 日的质量页补丁之后。需回退时先比对回执，恢复本次涉及的文件并仅重载网页服务。

### 2026-11-14 训练质量页面（历史版本，结束原因页已替换）

侧边栏的「验证趋势」与「结束原因」分别用 `/#trends` 和 `/#outcomes` 打开；「训练监控」仍提供轮次记录表和当前运行日志。访问的是同一个 CubeLab 服务，切换页面时可继续查看同一个实验。部署后在浏览器重新载入一次。

- 「验证趋势」用于逐轮比较学生左手在闭环验证中的表现。曾拿稳、结束仍拿稳和到达腕部终点且仍拿稳是不同事件。后者要求同一条轨迹同时有 `end_reason=teacher_wrist_endpoint` 和 `ending_hold=true`。平均稳持时长取每个回合最长的连续稳持段，再对本轮求平均。
- 「结束原因」按所选验证轮次展示每类的回合数和占比。每个回合只记一个结束原因。到达腕部终点是一种正常结束方式；结束原因只说明记录里首先触发的终止条件，不能单凭分类推断根因。
- 新接口 `GET /api/quality?run=<名称>` 返回 `id` 和逐轮 `rounds`。验过的逐回合报告与本轮汇总一致时才发布 `endpoint_held` 和 `reason_counts`。尚在采样的轮次没有成绩，未验收的数据不填零。读档按实验隔离，已解析的逐回合记录有大小上限和缓存。
- 长训第 14 轮验收样例：32 个验证回合都曾拿稳，30 个结束时仍稳持，28 个到达腕部终点且仍稳持，平均最长稳持 33.72 秒。28 个回合在腕部终点结束，2 个被规划器拒绝，1 个倾转保持超时，1 个倾转时失去主要支撑。这里只衡量整颗魔方的拿持与转动。
- 实现补丁在计时分支 `docs/evidence/hold90-dashboard-quality-20261114.patch`，接在 11 月 13 日两个仪表盘补丁之后。远端归档 `/home/brainco/rl_cube/work/cube_dashboard_quality_20261114_1942` 包含修订前的 `original/`、候选 `cube_dashboard/`、测试日志、Firefox 的独立验收及部署回执。回退前核对当前文件和回执哈希，再恢复本次改动的页面与后端文件；此次由 `systemctl --user restart rl-cube-dashboard.service` 重载仪表盘，不涉及训练服务。

在远端桌面打开 `http://127.0.0.1:8765/#training`，在训练选择框中选对应 run，看训练曲线下方的「采样耗时」。从自己的电脑查看时，可使用现有远端桌面，或执行 `ssh -N -L 18765:127.0.0.1:8765 brainco@100.105.58.63` 后打开 `http://127.0.0.1:18765/#training`。服务仍只监听远端回环地址。

**部署前端后，重新载入已打开的标签页。** 在浏览器按 Ctrl+R 或 F5，然后确认「实验」仍选中要看的 run。「立即刷新」和可见标签页的 5 秒轮询只请求最新接口数据；只有浏览器重新载入才会执行新版本的页面脚本。可核对「采样耗时」区块、所选 run 和最近日志时间。静态脚本由服务按请求读取，这一步无需重启仪表盘或训练。

回填实验 `contact_hold90_refill_3072_r5_20261113` 的批次进度看「已结束回合 / 总回合」。3,500 是每条轨迹的控制步上限，结束计数包含各种结束原因。拿稳率请查独立闭环验证。采样器每 120 个批次控制步写一条进度，因此相邻多次页面轮询可能显示同一个步数。这个版本的采样器没有函数计时；判别转腕规划和物理推进谁更慢，需要带计时诊断产生的同区间快照。

转腕诊断按 `waiting_gpu`（等待 GPU）、`collection`（物理采样）、`benchmark`（转腕路径批宽对照）、`benchmark_complete`（批宽对照已完成）推进。在等待阶段，已启动的是排队进程，取得共享 GPU 资源并开始采样后才会出现函数计时快照。接手时同时核对 `status`；中断时系统保留出错阶段及 `last_error`。

| 来源 | 页面字段和含义 |
| --- | --- |
| `GET /api/hardware` | 复用设备级读数，约 5 秒缓存。`sampled_at` 为 UTC 采样时间；`scope=device`；`util_percent`、`used_mib` 和 `total_mib` 为整卡读数。 |
| `GET /api/training?run=<名称>` | `progress.step/active/envs/completed/assigned/refills/total_episodes` 解析当前 worker 日志。总回合数不能作物理槽数。界面保留老格式支持。|
| 当前 worker 输出目录 `perf_timing.json` | 原子替换，schema 为 `hold90_collection_perf_timing_v1`。通过 `performance.available/counters` 展示。`available=false`、`reason=no_snapshot` 表示尚无该 worker 快照。结束采集后页面可读取最后验收的 worker，并标注其轮次。 |
| 同目录 `perf_timing_history.jsonl` | 追加保存快照，适合取两个控制步之间的累计差值。常规每 120 步、首次交接、限频阶段变化和终态会产生记录。 |
| `round_001/collection_throughput.json` | 已验收 worker 的有效环境步/秒、进入 tilt_hold 的轨迹/小时和达到腕部端点的轨迹/小时。分母含初始化、执行及序列化，不含等待 GPU 的时间。原始报告另有纯学生分项。 |
| `round_001/turn_path_benchmark.json` | 页面在 `performance.turn_path_benchmark` 展示轮次、分批宽度、中位时长和数值一致性；完整配对结果、内存数据和种子留在原文件。 |

函数时间采用 `time.perf_counter()`，表示主机进入至函数返回的累计秒数，包含函数原本就会发生的 CUDA 等待。采样计时没有为每项函数插入 GPU 同步。父项 `teacher.action` 包括 `teacher.begin_turn`，后者又包括 `teacher.turn_path`；三者不能相加。新回合重置见 `refill.begin/finish/reset_ik`，共享 8 子步的 settle 已含在物理调用内。`physics.step` 包括内部接触读取；当前没有独立 GPU 内核时间。`snapshot.write` 的展示滞后一份快照。独立组件 benchmark 才在每个组件计时边界同步 GPU。

## 接手时先执行

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
R=runs/contact_hold90_turn_probe_20261113
cat "$R/state.json"
tail -n 12 /home/brainco/rl_cube/work/hold90_turn_probe_20261113.launch.log
curl -fsS http://127.0.0.1:8765/api/health
curl -fsS http://127.0.0.1:8765/api/hardware
curl -fsS 'http://127.0.0.1:8765/api/training?run=contact_hold90_turn_probe_20261113' |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["stage"],d["progress"],d["performance"])'
lslocks -o PID,MODE,PATH | grep -E 'contact_hold_gpu_training.lock|isaac_gui_evaluation.lock'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

按 `state.json.diagnostic_pid` 检查进程及命令行，勿只依据旧 PID 或状态文件启动重复任务。等待态不表示 GPU 空闲。在 `collection` 阶段，日志路径在 `state.json.log`；去掉 `.log` 后就是本次 worker 输出目录。超时、内存保护或物理/校验失败的首个原因在该 worker 日志、同名 `.resources.json` 或 `round_001/benchmark.log`。若需重新诊断，保留原目录，换一个新 run，并使用相同的锁。原计划和实际命令记录在 `round_001/commands.json`、`benchmark_command.json`（各阶段启动时生成）。

## 判读与移交

1. 先读最终 `report.json`、`perf_timing.json`、`collection_throughput.json`。看 600–720、720–840 步前后相邻快照差值。检查该段 `teacher.turn_path` 的累计时长差、调用次数差及层级；并列看物理、重置、初始/回填 IK 和 bank 时间。只有完成且有真实调用的区间能用于归因。
2. 在 `turn_path_benchmark.json.results` 中优先比较 `mode=preserve_actual_groups`。核对每个配置的 `actual_batch_sizes`、多次同步时长、内存、`accept_changes_seeds`、`retimed_length_changes` 和有效帧误差。若 64/128 的分区与 32 相同，则没有实际扩大批宽，不能报加速。跨步拼组另列为 `cross_tick_component_only`，不能直接换算成每个真实控制步的收益。
3. 若同一步扩大分组的时间更短、显存可接受且轨迹检验一致，用新候选做固定种子闭环对照，再决定是否调整训练默认值。若 IK 并非主要耗时，沿实测分项调查 PhysX、接触读取或索引重置。本轮只观察和测量，未修改碰撞、控制速率、阶段门槛或正式 IK 分组。
4. 已完成的验证：计时分支的 25 项 CPU 测试通过；前端候选的 16 项测试通过；Firefox 在样例和正式服务端口渲染通过；真实 serializer 的临时 CPU 快照通过部署中的 API 读取器，见 `producer_consumer_contract.json`。它验证接口契约，不是训练 GPU 的物理计时。12:58 核验正式接口：旧回填 run 返回 1080 步、1535 个活跃槽、256 个完成、255 次回填，函数计时为空；诊断 run 返回 `waiting_gpu`。之后请按实时 API 更新判断。

## 回退与现场文件

14 时的静态页面修订只更新了线上 `cube_dashboard/static/app.js`。对照 12:56 部署回执核查时，其他五个文件沿用原哈希；这个文件使用后续回执 `/home/brainco/rl_cube/work/cube_dashboard_status_20261113_1404/deployment.json`。同目录 `previous_app.js` 保存这次修订前的脚本。计时分支中的 `docs/evidence/hold90-dashboard-status-20261113.patch` 接在原六文件补丁之后应用。本次修订需要单独回退时，先核线上脚本与后续回执一致，再恢复该备份并重新载入浏览器标签页。

采样器回退：新 run 可以使用 `161ae94` 的旧工作树/入口；旧模型、旧运行和其他训练进程不需要迁移。已有 run 的源码签名与实现版本绑定，不能用另一个版本强行 resume。运行中的正式批次不要热替换。

页面回退：先核 `cube_dashboard_perf_20261113_1251/deployment.json` 中的 changed_files 和线上哈希，若已有后续改动，应逐文件合并。备份 `original/` 包含部署前静态资源、后端和配置；`original_manifest.sha256` 可复核。只还原回执中列出的 6 个文件，备份中原先不存在的新增文件仅在仍与回执哈希相等时才移出线上目录。确认没有执行中的 dashboard 控制/回放作业，核正在监听 8765 的 dashboard PID/exe/cwd，沿原解释器、cwd、环境重新启动单个 `python -B -m cube_dashboard.server`，再检查 `/api/health`、`/api/hardware`、当前 run 接口和 Firefox 页面。原 `deploy.py` 包含当时 PID 校验，是部署记录，不宜直接重跑。此操作不需要重启 Isaac 或训练器。

若要复跑离线比较，先确认前述诊断已结束且 GPU 槽空闲。按捕获报告的 SHA 核 `turn_fixture.npz`，沿相同 Python、CUDA、URDF、校准、actor 和源码运行 `hold90_turn_path_benchmark.py --fixture <npz> --fixture-sha256 <sha> --output <新JSON> --widths 32 64 128 --repeats 3 --concat-small`。常规启动请复用 `run_hold90_timing_probe.py` 的锁与流程。物理结论限于 90 mm 仿真、左手拿持与整块倾转。这次诊断不检验右手单层转动或视觉复原。
