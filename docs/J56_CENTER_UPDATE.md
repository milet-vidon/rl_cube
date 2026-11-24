# J5 外旋、J6 内收居中候选（2026-11-03）

按用户反馈，将居中时的左臂协调方式改为 J5 外旋、J6 向内收，并减小 J7 弯折。肩肘继续配合完成横向位移；不把所有关节固定后仅转腕部。自由层魔方完整开发回合 `center_j56_dev_v3_20261103`（已见 seed 9501）通过，合格五指保持 8.03 秒。

## 新旧到位姿态

| 关节 | 上版居中实测 | 新版居中实测 |
|---|---:|---:|
| J5 | -89.6° | -53.3° |
| J6 | 8.2° | 29.8° |
| J7 | 63.7° | 35.0° |

新版相对居中起点的 J5/J6 实测变化为 +24.1° / +17.9°，J7 弯折减小 11.2°。以上正负号仅描述本机器人当前左臂构型，不作为其他机器人通用的“内外”定义。

J5 轴沿前臂，J6 为腕部弯折轴。两者主要改变腕后的姿态；固定其余关节时，腕心仍距中线约 32.8 cm，而腕心到持物中心约 25.0 cm，不能单靠这两个转动到达中线。仅调整 J5/J6 的本次 IK 尝试还剩约 13.3 cm 目标位置误差。因此新方案保留肩肘配合，并继续检查肘外展、掌心朝右与手腕在肘内侧。

## 三次实际物理回合

- v1：臂姿与居中位置达到要求，没有自碰、碰桌、掉落或限位命中；食指在魔方边缘失触，合格五指保持为 0 秒，记为失败。
- v2：食指固定 MPR −0.012 rad、PIP +0.003 rad 后五指接触全程恢复，但食指冲量明显增大，角速度偶尔超过 0.20 rad/s，最长合格保持仅 0.93 秒，记为失败。
- v3：取消固定 PIP 闭合，以有幅度上限的 MPR 接触反馈微调。每步最大 0.00005 rad，目标接触冲量区间 0.0005–0.002 Ns，累计补偿限制在 [−0.012, 0] rad；高冲量优先回退。到位观察延长到约 8 秒，连续 3 秒验收门槛不变。最终结果：通过，保持 8.03 秒，失败字段为空。

v3 自碰冲量 0.000000 Ns、机器人碰桌冲量 0.000000 Ns；软限位命中 0 步、硬限位命中 0 步、掉落 0 步。控制中魔方位姿写入次数为 0。食指补偿与手臂目标均经过原目标增量/速度限制、安全交集和 PD；没有修改抓持通过门槛、绑定魔方或关闭碰撞。

本轮仅做姿态开发验证，没有新增独立测试或 PPO。先前 9801–9803 在原腕部前置阶段失败的 0/3 仍有效，未在此轮修复。右手接抓与单层转动尚未通过；新左腕姿态也需要重新进行右手联合规划。不能用本次开发演示宣称完整流程已经稳定。

## 实现和使用

新增 `center_j56_plan.py`、`center_j56_audit.py`、`center_j56_protocol_v3.json`。默认 `center-teacher` 采用 j56 风格，旧路径可显式用 `--center-style legacy`。v1/v2 协议和三个回合的原始源码快照保留。

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
./run_linux.sh center-teacher --center-style j56 --name UNIQUE_RUN --seed 9501 --record
```

运行环境：Isaac Sim 6.2 / Isaac Lab 3.0，RTX 5070 Laptop GPU，1 个环境；CPU PhysX、GPU 渲染，控制 30 Hz / 物理 240 Hz。v3 完整回合用时 461 秒。无神经网络训练和 checkpoint，资产、源码、配置及软限位版本见各回合清单。

相关离线测试命令：`python3 -m unittest test_center_j56_audit test_centered_transport_audit test_opposed_grasp_audit test_layer_turn_metrics test_wrist_audit -q`。

36 个相关离线测试、1205 次新路径与食指补偿范围预检、完整 PhysX 回合、配置/资产/协议/源码哈希和冻结 27 文件校验均单独记录。证据见 [新旧对照指标](evidence/j56_center_20261103/center_j56_dev_v3_20261103_comparison.json)、[新姿态录像](evidence/j56_center_20261103/j56_center_demo.mp4)、[同步对比录像](evidence/j56_center_20261103/centering_comparison.mp4)；录像为实际 1 倍速度，来自开发 seed 9501。压缩完整轨迹和视频使用 Git LFS。
