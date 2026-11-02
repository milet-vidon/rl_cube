# 项目交接（2026-11-02）

## 目标与当前结论

目标是双臂 Revo3/Tianji 从抓取、换抓发展到实际单层转动，再接视觉与复原。目前最成熟的是锁层自由刚体魔方上的传统接触控制。没有完成神经视觉抓取、机器人单面转层、连续复原或实机部署。

| 分支 | 已记录结果 | 证据入口 |
|---|---|---|
| 安全 staged teacher | 独立 5/5，保持 1 秒 | `revotron_cube_rl/pipeline_status.json` |
| 状态 BC v4 | 独立 1/5；状态 PPO 未晋级 | `runs/state_selected_independent_test/result.json` |
| 侧夹 BC + 接触/到位监督 | 独立 6/6，保持 3 秒；不是纯 PPO | `SIDE_GRASP_TRAINING.txt`、`runs/side_bc_waypoints_v2/independent_test.json` |
| 对掌 teacher v20 | 独立 7301–7305：5/5；抬升 18.54–18.94 cm | `OPPOSITION_TEACHER_RESULT.txt`、`opposition_gate_certificate.json` |
| 整块倾斜 + 右手换抓 v17 | 开发 8201 完整回合通过；独立记录 8301 失败、8302 通过 | `runs/tilt_regrasp_full_v17_shift/result.json`、`runs/tilt_regrasp_independent_v17/result.json` |
| 动态魔方 | 被动检查通过；机器人转层未验收 | `pipeline_status.json`、`dynamic_cube.py` |

表中未加前缀的路径相对 `revotron_cube_rl/`。以上是已有实验记录，本次发布没有重新进行 PhysX 训练验收。打包时原工作区仍在运行 v17 独立实验（包含待完成的 8303）和 v18 候选恢复实验。本快照仅收录 v17 已写入的 8301/8302，1/2 是部分进度，不是最终整批成功率。v18 不属于已验收基线。

## 重要历史纠正

- 旧文档宣称的外部推板 +Z/+X 转层通过，后续发现 kinematic Core 与世界坐标转角误判，已撤销，不可作为机器人转层基线。
- 旧 Linux 迁移文件里的“未运行 Linux GPU”已过时：6.2 部署已有真实 RGB smoke、teacher、PPO 更新与重载记录。
- `pipeline_status.json` 更新到侧夹阶段；对掌和倾斜需要继续读各自 result/certificate，不能只看该总表。
- resume 回合只作调试，不能作为从 reset 起完整抓取→倾斜→换抓成功；`accepted` 还必须结合 `full_episode_verified` 和审计项。

## 代码地图

| 模块 | 职责 |
|---|---|
| `environment.py`, `assets.py`, `isaac_compat.py` | 场景、56 DOF 接口、Lab 3 / PhysX 兼容 |
| `joint_safety.py`, `acceptance.py`, `rewards.py` | 软硬限位、安全与抓取验收 |
| `traditional_controller.py`, `sweep_controller.py` | 传统抓取 / staged teacher |
| `state_learning.py`, `state_policy.py` | 状态 BC/PPO |
| `side_waypoints.py`, `side_policy_evaluate.py` | 侧夹关键帧 BC 与闭环监督 |
| `opposition_teacher.py`, `opposition_experiment.py` | 对掌主抓与右手支撑 |
| `tilt_regrasp.py`, `tilt_experiment.py`, `tilt_audit.py` | 整块倾斜、换抓与实测审计 |
| `dynamic_cube.py`, `robot_hand_turn_stage.py` | 动态层模型与机器人接触转层研究入口 |
| `stage_gate.py`, `opposition_gate.py`, `tilt_gate.py` | 阶段门禁、源码/资产/轨迹证书 |
| `policy.py`, `train.py`, `collect_dagger.py`, `rsl_compat.py` | 视觉/PPO/DAgger 与官方 RSL 适配 |

## 不可破坏的约束

56 维动作 = 左右各 7 臂 + 21 手。物理 240 Hz、命令 30 Hz。视觉 actor 为 RGB + 224 本体觉，不可泄漏 critic 真值；状态 teacher/状态策略另行标记。

双臂 J1/J3/J5 ±178°，J2 ±120°，J4 [-145°,60°]，J6 ±60°，J7 ±90°，再与硬限位求交。每步目标变化上限 0.04 rad、臂速度上限缩放 0.5。对掌分支还存在右肩 J2 不低于 -100° 的任务姿态约束。

抬升至少 8 cm、双侧真实接触、稳定保持；不同分支保持门槛为 1 秒或 3 秒，不能互换。碰桌、自碰撞、软/硬限位、掉落等必须单独审计。对掌分支 PD 阻尼 8，历史基线 1，不能混淆配置。

## 可复现性缺口

1. gate JSON 含 `/home/brainco/rl_cube` 绝对路径；换机器后可能找不到证据，甚至在原机误读旧工作区。必须先修复根路径解析并保持哈希校验，不能直接把门禁标为通过。
2. 历史训练集/视频未全量发布；仓库保留关键证书引用、两个选定 checkpoint、倾斜完整轨迹和小型结果。见 ARTIFACTS.md。
3. 原环境完整版本快照放在 `docs/environment/`，其中绝对路径与 editable 安装是记录，不应直接当作可移植 requirements。
4. `work/tilt_handover` 的探索脚本不属于发布源码；未验收候选不可替代 v17 已记录版本。
5. 新工作机先离线测试、资产检查，再 runtime/render smoke，最后选择单个 teacher 回合。普通 GitHub runner 无法证明 GPU/PhysX 行为正确。

## 下一步顺序

优先修复证据路径可移植性和汇总状态一致性；然后复现 v17 的 8301 失败、分析与 8302 差异，冻结修复后登记全新 seed。换抓稳定后再构建真正可动层的状态单面任务。视觉训练继续遵守阶段门禁，不应直接追加 PPO 迭代。

可直接开 Issue 的任务和验收条件见 NEXT_ISSUES.md。接手者每次 PR 应把新结论及证据路径写回本文件，并保留失败历史。
