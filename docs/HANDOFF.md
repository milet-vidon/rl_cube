# 源关节被动阻尼对照（2026-11-03）

v11 已完成：五指合格保持 4.033 秒，但实际层转角只有 2.305°，15° 保持未通过。v12 被动阻尼回合已完整完成：核心角速度降至门槛内、右手五指全接触，但左无名指在右手闭合后失触，未转层。74 项相关离线测试通过；v13 降低右指刚度并加入有界转角反馈，完整回合正在运行。未启动新的转层策略训练。见 [实现、边界与复现](PASSIVE_JOINT_DAMPING_UPDATE.md)。下文保留历史状态。

# 转层动力学与规划修正更新（2026-11-03）

v5 右手五指保持 4.03 秒，实际层转角 5.45°，仍未达到 15°；源惯量 v6 在初次抬升失抓；64 次位置迭代 v7 左手居中通过，但右手被原碰撞预检拒绝。间隙修正后的 v8 路径通过并建立五指接触，但核心角速度超限，未执行转层；v9 内部摩擦 (.4,.3) 导致层偏移 5.43°，前置验收失败；v10 (.6,.5) 在第 666 步失抓。已恢复默认内部摩擦，v11 仅降低右指刚度至 10，完整回合正在运行。没有新的自由旋转模型或训练成功。详见 [本轮结果、物理审计与复现](TURN_DYNAMICS_UPDATE.md)。下文保留历史状态。

# 当前优先级：先可靠转动，再训练与视觉复原（2026-11-03）

按用户最新要求，先完成目标层转动、换面与重抓，再训练这些操作技能，最后接入视觉识别和公式求解。尚未开始的 v10 评估已取消，保留其 checkpoint；当前物理转层实验继续。见 [实施顺序与接口](CUBE_SOLVE_ROADMAP.md)。

# Teacher 接续训练更新（2026-11-03）

右手五指保持通过 3.5 秒，并实际转动目标层约 5.7°，尚未通过 15° 转层验收。左手 BC v7 能举起但在腕部重构时掉落；v10 已完成 10 万次训练，等待闭环对照。新增腕肘保持候选仍未通过，右手单独材质实验正在运行。见 [接续优化与训练记录](TEACHER_CONTINUED_UPDATE.md)。

# Teacher 联合优化与闭环训练更新（2026-11-03）

右手已推进到真实四指接触，无名指未触及，未执行转层；左手 BC v5 完整闭环仍未抓起魔方。已安排无名指修正回合及仅接纳合格示范的多种子训练队列。见 [本轮优化记录](TEACHER_OPTIMIZATION_UPDATE.md)。

# 状态 teacher 初始化更新（2026-11-03）

左手 BC 初始化完成并保存 checkpoint，右手接抓搜索仍失败，闭环开发评估已排队。没有右手转层或 PPO 成功。见 [训练与开发记录](TEACHER_BOOTSTRAP_UPDATE.md)。下文为此前阶段记录。

# J5 外旋、J6 内收居中更新（2026-11-03）

用户要求将居中动作改为 J5 外旋、J6 向内收。新增 j56 规划和实测关节协调验收，J7 到位弯折由约 64° 降到 35°；肩肘仍配合位移。首轮食指失触，固定补偿又出现角速度越限；改为有界食指接触反馈后，开发 seed 9501 的完整回合通过，五指同窗保持 8.03 秒。详情见 [J5/J6 居中记录](J56_CENTER_UPDATE.md)。

默认 `center-teacher` 使用 `--center-style j56`；`legacy` 保留旧路径。没有新增独立或 PPO 成功，之前自由层前置腕部 9801–9803 的 0/3 未修复，右手接抓/转层仍未放行。下文保留历史记录。

# 左手居中与五指接抓更新（2026-11-03）

用户要求先将左手持物移到中线，再以右手拇指对四指接抓转层。居中开发 v2 已完成移动并保持 4.03 秒；新独立 9801–9803 的完整流程为 **0/3**，均在原腕部调整前置门失败、未进入居中。右手离线碰撞路径虽得到候选，但掌心朝向未通过原门槛，没有执行转层或解锁 PPO。

当前详细状态、运行入口、首次碰躯干失败和独立失败证据见 [居中与五指接抓记录](CENTERED_OPPOSITION_UPDATE.md)。`center-teacher` 为新增居中入口；`face-turn` 改为五指开发流程；旧单指探测改名 `face-push-probe`。下文为历史记录，其“当前单指入口”描述不再代表默认行为。

# 自由层接触转面开发（2026-11-03）

按用户新要求开始“左手保持、右手推转”的单指开发路径。自由核心魔方上的左手抓持已在开发回合通过，但两次15°实际接触推转均未带动整层；单面90°未验收。降低内部摩擦的实验暴露层偏移/抓持前置门失败，资产审计确认原模型被动关节参数未完整迁移。详见 [本轮转层开发与失败证据](SINGLE_FACE_CONTACT_UPDATE.md)。 后续修复见 [Issue #5](https://github.com/milet-vidon/rl_cube/issues/5)。

`face-turn` 为15°边缘探测入口，仍检查原有前置证书、当前回合抓持、机器人安全和九块相对核心转角。5个完整开发尝试、1个中断尝试全部使用已见 seed 9501，没有新独立成功率或PPO。右手五指接抓 #2 未被本单指路径替代。下一步先校准内部被动力学与接触力控制，不扩大动作或解锁训练。下文保留历史状态。

# 最新澄清与腕部修正（2026-11-03）

用户明确此前指的是腕部方向，要求“手腕弯折换到另一侧，掌心仍朝右手”。默认入口现为 `wrist_direction_experiment.py`：反转 J6 侧弯，保持肘外展与掌心方向。新独立 9701–9705 为 5/5，五指接触与姿态同窗保持 3 秒，无自碰、碰桌、掉落或限位命中。旧 9601–9605 仅证明旧腕姿的肘外展，未混入新结果。

详见 [腕部修正与验收](WRIST_DIRECTION_UPDATE.md)。持物点向左上方移动，#2 的右手接抓需要重新规划，尚未验收；没有新增 PPO/BC。下文按时间保留历史定义，当前行为以上述腕部修正为准。

# 当前交接更新（2026-11-03）

以 [本轮实施与验收记录](ISSUES_123_UPDATE.md) 为当前入口：#1 左肘外展实测独立 9601–9605 为 5/5；#3 解析和隔离离线验证通过；#2 右手接抓仍未验收。默认 tilt-teacher 已改为 elbow_handover_experiment.py，--regrasp 是 safe_handover_experiment.py 开发路径。

新文件：elbow_handover.py / elbow_audit.py / elbow_gate.py / elbow_out_protocol_v1.json / elbow_out_gate_certificate.json；evidence_paths.py / verify_portable_evidence.py / validate_relocated_evidence.py；safe_handover.py / safe_collision_plan.py / safe_path_audit.py / wrist_collision_schema.py。

9601–9605 已经用于独立验收，后续不能再称新 seed。9501 为开发/回归。右手接抓前置未通过，不继续盲目增加 PPO。下文是 11 月 2 日原交接，保留历史语境，其中“左肘未修复”“证据解析未迁移”不再描述当前实现。

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
