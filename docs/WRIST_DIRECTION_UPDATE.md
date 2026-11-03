# 腕部弯折换侧，掌心方向保持（2026-11-03）

用户澄清此前所说“肘方向反了”实际指腕部，选择“手腕弯折换到另一侧，掌心仍朝右手”。本版反转 J6 对应的侧弯方向，保留肘外展和掌面方向；没有翻转整只手，J7 屈伸没有反转。

## 实现与范围

`wrist_direction.py` 继承已通过的肘外展 teacher，保留桌面抓取与整块 -90° 倾斜，随后用 6 秒限速轨迹调整肩臂冗余构型与 J6。所有关节仍经软/硬限位、速度限制、安全层和 PD；没有绑定物体、写回位姿或关闭碰撞。

固定原持物点时，本次求解未能同时满足反向 J6 与肘外展约束，因此通过位置偏移代价寻找有界工作空间解。开发回合持物点约向左移动 9.7 cm、向上 13 cm，掌部姿态约束不变；具体实测偏移逐种子保存。右手仍停在原位置，尚未重新对准该新持物点，不能把“掌心方向保持”解释为完整双手对掌接抓已通过。

`wrist_audit.py` 从 PhysX 实测肘/腕位置、掌坐标系与关节角重算侧弯，不信任预存的 `measured_wrist` 摘要。要求 J6≥5°、掌坐标系下前臂 Y/Z 夹角≤−5°、腕在肘内侧≥5 cm，并在同一个≥3秒窗口内满足五指接触、肘外展、掌向机器人右侧、抬升、低速、整块角度及离桌要求。

## 实测结果

开发 9501 为 1/1；冻结源码与协议后，新独立 9701–9705 为 **5/5**。首个批次在 9704 的第 1080 步收到终止（退出码143），未形成完整轨迹，不计成功；保存原日志，源码/参数未变，另起批次从 reset 完整重跑 9704，再运行 9705。这些 seed 已见，后续不能再次称为新独立测试。旧肘外展 9601–9605 的 5/5 没有混入本版结果。

| seed | J6 前→后 (°) | 实测侧弯前→后 (°) | 手腕在肘内侧 (cm) | 同时合格保持 (s) | 掌方向变化 (°) | 最大抬升 (cm) |
|---|---:|---:|---:|---:|---:|---:|
| 9501 (development) | -17.95 → 12.11 | 26.63 → -15.32 | 10.24 | 3.033 | 0.142 | 36.88 |
| 9701 (independent) | -15.50 → 12.13 | 21.77 → -14.38 | 8.83 | 3.033 | 0.121 | 36.22 |
| 9702 (independent) | -8.19 → 12.14 | 12.79 → -15.37 | 10.00 | 3.033 | 0.097 | 32.21 |
| 9703 (independent) | -10.77 → 12.14 | 15.92 → -14.81 | 9.33 | 3.033 | 0.106 | 33.68 |
| 9704 (independent) | -25.45 → 12.10 | 34.39 → -14.22 | 9.12 | 3.033 | 0.184 | 40.53 |
| 9705 (independent) | -8.35 → 12.14 | 13.00 → -15.28 | 9.56 | 3.033 | 0.091 | 32.90 |

上述六个完整 PhysX 回合自碰与机器人-桌面冲量均为 0 N·s，无软/硬限位命中、掉落或抬升后重新接触桌面。每个回合都通过同窗 3 秒保持审计。独立验收总墙钟耗时（含中断与重启）902.4 秒；单环境、CPU PhysX、RTX 5070 Laptop 8 GB（开发录像使用 GPU 渲染），Isaac Sim 6.2 / Lab 3.0。61 个离线测试通过；开发前凸几何预检仅作路径筛查，不代替 PhysX 证据。

没有新 BC/PPO 训练，因此没有新增神经网络 checkpoint。传统 teacher 的实现/配置以 SHA256 冻结；原资产与软限位版本不变。整块倾斜不是魔方单层转动。右手完整接抓、右手单独持物、机器人转层和实机均未验收。

## 证据与运行

- 协议：`revotron_cube_rl/wrist_direction_protocol_v1.json`，SHA256 `7a9d04f76118e45996b5008d00228d5000c53f5eb0f5841dffa9eff0cb16b1ba`。
- 证书：`revotron_cube_rl/wrist_direction_gate_certificate.json`，逐回合包含源码、配置、资产与轨迹哈希。
- 完整证据：`runs/wrist_direction_dev_v1_20261103`、`runs/wrist_direction_independent_v1_20261103`、`runs/wrist_direction_independent_remaining_v1_20261103`。
- [开发实测完整录像](evidence/wrist_direction_v1/wrist_demo.mp4)、[腕部换侧片段](evidence/wrist_direction_v1/wrist_change_clip.mp4)、[正面](evidence/wrist_direction_v1/wrist_front.png)、[侧面](evidence/wrist_direction_v1/wrist_side.png)、[实测曲线](evidence/wrist_direction_v1/wrist_metrics.png)。片段来自完整视频 40–52.4 秒，没有合成画面，也不作为验收依据。
- [逐 seed 指标](evidence/wrist_direction_v1/wrist_metrics.json)、[离线测试](evidence/wrist_direction_v1/offline_tests_61.txt)。

```bash
source activate.sh
# 9501 是开发回归，不能算新独立测试；name 必须全新。
bash revotron_cube_rl/run_linux.sh tilt-teacher --name wrist_regression_unique --seeds 9501 --record
bash revotron_cube_rl/run_linux.sh wrist-check --out /tmp/wrist_reaudit.json
```

默认 `tilt-teacher` / `wrist-teacher` 使用新腕部实现。`elbow-regression` 保留旧腕部构型的肘外展回归。`regrasp-development` 明确是旧腕部姿态下的未通过右手研究分支；新腕部入口拒绝 `--regrasp`，不会悄悄回退旧姿态。

下一步：按新持物点重规划右手腕部、相邻手指及闭合全路径，完成 #2 后再考虑蒸馏或 PPO；保留旧失败记录。#3 的原证书字节和跨工作区解析继续保留。
