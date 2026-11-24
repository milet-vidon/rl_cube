# R27 自然换姿与右手拧层：V3 诊断修正版

2026-11-18。此版本尚未运行真实 PhysX；用户要求只提供运行命令，不由助手启动训练。它是待物理验证的控制器修正，不能称为已解决自碰或已训练成功的模型。

V2 的远端已完成证据位于 `/home/brainco/rl_cube/github_project/revotron_cube_rl/runs/r27_natural_face90_v2`。128 条训练回合有 105 次 `robot_self_contact`：原抓取 1 次、outside 25 次、hover 27 次、engage 30 次、sweep 22 次。旧日志没有碰撞双方的部件名，也没有逐帧完整关节姿态；仅靠这些文件不能确认具体碰撞对。离线理想跟踪路径与 URDF 凸包检查也不能替代实际碰撞反馈。

V3 修改以下控制环节：

- 在右手原待机位置逐步预成形手指，实际关节到位连续 10 帧且等待至少 45 帧后才伸向魔方。V2 在伸手途中同时改变指形，指腹 IK 的末端几何随之改变。
- 右臂用独立运动学参考、轻量跟踪及速度阻尼，不再反复把实际跟踪滞后叠加进新目标；参考相对实测关节的超前限制为 0.06 rad。
- 在外侧路径点先对齐完整掌姿，再进入贴面阶段，减少贴近魔方时的突然转腕。指形变化限制约 0.33 rad/s。
- 被动记录原生接触对，保留原碰撞冲量判据与所有原有门槛。`collision_events.jsonl` 包含双方 body 路径、冲量、控制阶段、种子、56 维实测姿态与命令，以及碰撞前 12 个控制帧。逐物理子步累加的配对冲量必须与原始自碰总冲量一致，否则显式报错。

R27 权重、V2 源码及历史结果均保留。仍是 R27 神经抓取后接六参数反馈控制器搜索，目标仍是接触驱动的正向 15° 单层转动并撤手保持；不是 PPO 重训、90° 转面或实机能力。V3 使用新的训练/验证种子段，保留共享独立测试种子登记与原晋级条件。

在远端 5090 运行，先做 8 个历史种子的短诊断：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_natural_face90_v3_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v3_probe_$(date +%Y%m%d_%H%M%S)" \
  --smoke --lanes 8
```

结束时终端会打印诊断汇总，目录下也有 `diagnosis.json`。这里的 `natural_observed` 包含后来发生自碰的回合，不等于安全成功率；`smoke.json` 的评分仍将不安全回合的中间进度归零。具体碰撞见各已接受 attempt 的 `collision_events.jsonl`。短诊断进程正常退出只表示记录完整，不能视为任务成功。

先确认短诊断 `unsafe=0`、大多数回合完成自然换姿且能触碰目标层，再运行长训：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_natural_face90_v3_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v3_long_$(date +%Y%m%d_%H%M%S)" \
  --rounds 40 --population 8 --seeds-per-candidate 4 \
  --validation-seeds 8 --lanes 32
```

若短诊断仍自碰，先读 `diagnosis.json` 的 `self_collision_pairs` 和事件中的前序姿态，继续修对应部件路径，不应原样重复长训。中断恢复需要填写该次实际生成的目录，保持全部参数不变再加 `--resume`；不要用新的时间戳恢复，也不要恢复 V2。源码或配置改变后必须另建实验目录。

离线验证命令：

```bash
/home/brainco/rl_cube/.conda/rl_cube_6_2/bin/python -B -m unittest \
  test_natural_face90_v3 test_natural_face90_metrics -q
```

覆盖接触归属、子步累计、碰撞证据哈希、手型到位门槛、阻塞时参考超前限制、真实 URDF 运动学可达性、原关节/阶段/转层计分及训练早停契约。它们不证明实际手指接触或转层成功。`--check --run-dir /tmp/unused` 仅验证冻结输入，不启动 Isaac。
