# 左手居中与右手五指接抓开发（2026-11-03）

用户要求：左手拿稳后移到身体中间；右手采用与左手相同的拇指对四指抓法，再通过真实接触转层。

当前完成的是**左手居中开发演示**，不是稳定的完整双手转层系统。开发 v2 将自由层魔方从左侧 Y=0.287324 m 移到 Y=0.000246 m、身体前方 X=0.499965 m，五指真实接触与姿态同窗保持 4.03 秒。冻结版本后测试新种子 9801–9803，完整流程 **0/3**；三次都被原腕部调整前置门拦住，均未执行居中。不能把开发演示作为独立成功率，也不能沿用锁层魔方的腕部 5/5 来证明自由层模型已稳定。

## 实测回合

| 回合 | seed / 分组 | 结果 | 失败或通过范围 |
|---|---|---|---|
| center_dev_v1 | 9501 / 开发 | 失败 | 已接近中线，Link5_L 与 world 躯干接触，累计自碰冲量 0.149389 Ns，立即停止 |
| center_dev_v2 | 9501 / 开发 | 居中段通过 | 左手抓稳、移动、保持 4.03 秒；无自碰、碰桌、掉落、软硬限位命中 |
| centered_left_independent_9801 | 9801 / 新独立 | 失败，居中未执行 | `Wrist-side reversal IK violates palm pose envelope` |
| centered_left_independent_9802 | 9802 / 新独立 | 失败，居中未执行 | 同上 |
| centered_left_independent_9803 | 9803 / 新独立 | 失败，居中未执行 | 同上 |

三次独立回合都无自碰、碰桌、掉落或限位命中；但没通过前置抓持/腕姿同窗门，因此不能宣布安全完成任务。其原始 `push.failure=not_requested` 是 `--center-only` 的报告标签缺陷；顶层失败原因正确，原始结果保留。当前代码已修正该标签，未重写历史证据。

五个 PhysX 回合累计运行约 23.1 分钟，单环境、RTX 5070 Laptop 8 GB、CPU PhysX、Isaac Sim 6.2 / Lab 3；开发录像使用 GPU 渲染。没有新增神经网络训练或 checkpoint；`checkpoint=null` 和原因均记录在证据中。

## 居中实现与碰撞修复

新增 `centered_transport_plan.py`、`centered_opposition_runtime.py`、`centered_transport_audit.py`：在原抓稳/腕部前置门通过后，仅规划左臂七关节，左手指目标与右臂保持；通过肩、上臂共同改变持物位置。实际执行仍经过既有软硬限位、目标增量/速度限制、PD 与 PhysX，不绑定魔方、不写入运行中的魔方位姿。

目标位于机器人根坐标 `[0.50, 0, 0.15]` m；保持原 J6 方向、掌心朝右、肘外展、手腕在肘内侧，并增加掌坐标系下前臂 Z 分量 >0.3 的显式复核。中心位置、姿态、五指接触、稳定速度、完整层结构与安全条件必须在同一保持窗口满足。

旧 URDF 离线模型没有覆盖 USD 中独立的 waist_pitch 躯干碰撞外形。加入 USD 原始凸包后，仍无法复现已见的 Link5/world 接触；因此新增 `torso_collision.py`，从实际 USD 提取固定碰撞体并为 waist_pitch 加保守包围盒。已见失败姿态现在会被拒绝，新路径通过 241 次检查。**这不证明离线网格与 PhysX 烹制后的所有形状完全一致**，实测接触停止条件仍保留。

27 个冻结的已验收控制源码和原证书没有修改。独立测试保存的源码与配置为运行时快照；之后仅补充了当前驱动器的居中失败标签、顶层结果字段和独立的右手开发模块，未修改测试过的居中控制算法。

## 右手进展与明确限制

新增 `opposed_grasp_plan.py`、`opposed_grasp_runtime.py`、`opposed_grasp_audit.py`，目标是拇指与其余四指都真实接触指定 nZ 层，先保持 3 秒，再尝试接触转层。右腕/前臂碰魔方、近端手指碰触、错误层接触、缺少任一目标手指，均不能作为合格五指抓持。

离线研究中，直接镜像和角度偏置暴露出腕部过近、手指落点不可达、相邻手指间距不足。静态张开与闭合两端通过后，直线闭指中间仍出现约 2.92 mm 间距，低于保留的 3 mm 阈值。加入闭指中间 MPR 展开量优化后，v11 的碰撞路径预检通过；但右掌朝左余弦约 0.45，未达到原有 0.65 门槛。最终实现显式拒绝该姿态。进一步调整掌姿的 v13 候选出现右臂 IK 或无名指落点不可达，未执行物理动作。

**右手五指抓持和转层均未通过实际物理验收；本轮没有右手转层演示。** v11 原始 `passed=true` 只代表当时的碰撞路径子检查，不代表完整姿态门通过；与最终拒绝结果一起保存，不能用于放行。旧单指探测保留为诊断入口，不是当前要求的抓法。

## 入口

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"

# 居中开发演示；每次使用新的 run 名称
./run_linux.sh center-teacher --name UNIQUE_CENTER_RUN --seed 9501 --record

# 开发中的完整五指流程；前置或右手预检失败会停止
./run_linux.sh face-turn --name UNIQUE_OPPOSED_RUN --seed 9501 --record

# 旧单指接触诊断，显式选择
./run_linux.sh face-push-probe --name UNIQUE_INDEX_PROBE --seed 9501
```

9801–9803 已经用于这次独立评估，后续调参后重跑只能算已见种子。当前不放行 PPO、90°单面完成声明或两面动作。下一项应修复自由层魔方上的腕部保持反馈与可达性，再联合规划满足掌心方向的双手姿态。

## 验证与证据

24 个相关离线测试通过；已见躯干碰撞姿态负向回归、新居中路径预检、冻结源码哈希和模块编译检查另行记录。没有以离线通过替代 PhysX 验收。

- [结构化汇总](evidence/centered_opposition_20261103/summary.json)
- [开发演示视频](evidence/centered_opposition_20261103/left_center_demo.mp4)：19.6 秒、实际 1 倍速，包含移动及稳定保持；来自开发 seed 9501。
- [最终画面](evidence/centered_opposition_20261103/center_hold.png)
- 每回合目录：运行时源码、配置、结果、资产信息、压缩完整轨迹、原产物清单和归档清单。新独立回合另存实际机器人 USD 入口及依赖层哈希。
- [右手离线研究与拒绝证据](evidence/centered_opposition_20261103/right_hand_preflight/)
- Git LFS 保存大轨迹和录像。原始回合仍位于本机 `revotron_cube_rl/runs/`，不覆盖失败记录。

关联 [PR #6](https://github.com/milet-vidon/rl_cube/pull/6)、[Issue #2](https://github.com/milet-vidon/rl_cube/issues/2)、[Issue #5](https://github.com/milet-vidon/rl_cube/issues/5)。
