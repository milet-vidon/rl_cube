# 项目交接（2026-11-02）

## 目标与当前结论

用户最新要求：左手肘朝外；当前演示朝内，尚未满足。掌心翻转通过与自然肘姿验收必须分开。最新证据见 [翻转修正说明](PALM_DIRECTION_UPDATE.md)。

目标是双臂 Revo3/Tianji 从抓取、换抓发展到实际单层转动，再接视觉与复原。目前最成熟的是锁层自由刚体魔方上的传统接触控制。没有完成神经视觉抓取、机器人单面转层、连续复原或实机部署。

| 分支 | 已记录结果 | 证据入口 |
|---|---|---|
| 安全 staged teacher | 独立 5/5，保持 1 秒 | `revotron_cube_rl/pipeline_status.json` |
| 状态 BC v4 | 独立 1/5；状态 PPO 未晋级 | `runs/state_selected_independent_test/result.json` |
| 侧夹 BC + 接触/到位监督 | 独立 6/6，保持 3 秒；不是纯 PPO | `SIDE_GRASP_TRAINING.txt`、`runs/side_bc_waypoints_v2/independent_test.json` |
| 对掌 teacher v20 | 独立 7301–7305：5/5；抬升 18.54–18.94 cm | `OPPOSITION_TEACHER_RESULT.txt`、`opposition_gate_certificate.json` |
| 整块倾斜 + 右手换抓 v17 | 开发 8201 完整回合通过；独立记录 8301 失败、8302 通过 | `runs/tilt_regrasp_full_v17_shift/result.json`、`runs/tilt_regrasp_independent_v17/result.json` |
| 掌心方向修正 v5（左手翻转/保持） | 开发 3/3、独立 9501–9503 为 3/3；左肘外展、右手接抓未验收 | `palm_direction_gate_certificate.json`、`PALM_DIRECTION.txt` |
| 动态魔方 | 被动检查通过；机器人转层未验收 | `pipeline_status.json`、`dynamic_cube.py` |

表中未加前缀的路径相对 `revotron_cube_rl/`。v17 是初次发布保留的历史部分进度，不是当前基线；旧 v19 的方向/交叉姿态被用户否定，其证书已撤销当前任务的通过状态。v5 的六个完整 PhysX 回合及两个接抓失败回合已收录。本次同步使用已有实测证据，没有重新训练或重跑物理仿真。左肘外展不在已有证书验收范围内。

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
| `palm_handover.py`, `palm_handover_experiment.py`, `palm_handover_audit.py` | 反向翻转、保持反馈、掌心方向与不交叉审计；接抓路径未验收 |
| `palm_direction_gate.py`, `palm_handover_replay.py` | 左手翻转证书和实测回放；不证明肘外展或右手接抓 |
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
4. 探索脚本不属于发布源码。旧 v17/v19 倾斜结果不能替代最新姿态要求；当前 v5 的通过范围仅为左手反向翻转和保持。
5. 新工作机先离线测试、资产检查，再 runtime/render smoke，最后选择单个 teacher 回合。普通 GitHub runner 无法证明 GPU/PhysX 行为正确。

## 下一步顺序

优先修复左肘外展构型并保持已认可的左手抓取、掌心方向与安全要求；随后处理右手腕部及相邻手指碰撞，完成从 reset 开始的完整接抓验收。证据路径可移植性同步处理，避免克隆后误读原工作区。已使用的 9301、9402、9403、9501–9503 均不得再次计为新独立种子。状态单面任务、视觉蒸馏和 PPO 继续等待前置阶段验收。

可直接开 Issue 的任务和验收条件见 NEXT_ISSUES.md。接手者每次 PR 应把新结论及证据路径写回本文件，并保留失败历史。
