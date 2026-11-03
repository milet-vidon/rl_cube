# 左手状态 teacher 初始化（2026-11-03）

接续 PR #6，从已验收的 `center_j56_dev_v3_20261103` 完整开发轨迹开始 BC 初始化。新增 `teacher_bootstrap.py` 和单独的闭环开发评估入口 `evaluate_teacher_bootstrap.py`。

## 已完成的训练

- 6000 次 Adam 更新，1708 个训练样本、420 个同回合留出时间块样本。
- 输入：动作前的 56 关节实测位置、魔方位置/四元数和控制时间；输出：按原顺序排列的 28 维左臂/左手绝对目标。
- 数据使用前核对原 manifest 的轨迹、配置和结果哈希，并拒绝不完整、未通过居中、接触安全失败或写回魔方位姿的回合。
- 日志中的关节状态是动作执行后状态，因此使用上一行状态配当前行动目标，避免动作后状态泄漏。
- 最佳留出目标 RMSE：0.00568784 rad（约 0.326°）；checkpoint 重载输出检查通过。这是离线拟合误差，不是抓持成功率。
- checkpoint：`revotron_cube_rl/runs/teacher_left_bc_bootstrap_20261103_1905/best.pt`；发布副本在本目录 `evidence/teacher_bootstrap_20261103/best.pt`，由 Git LFS 管理。
- checkpoint SHA256：`edb6bf4d466e57a67cac14673aa83517b837a206046c9efcf9553a555f731733`。

仅使用一条已见开发 seed 9501 的示范，没有独立成功率。该模型没有学过右手转层，不解锁状态单面 PPO、视觉训练或任何原有阶段门禁。

## 本轮右手接抓开发

从最新 J5/J6 居中轨迹的最终实测状态重新规划，原候选在右中指/无名指 DIP 间距约 2.466 mm 处被原 3 mm 门槛拒绝。保留门槛，尝试增大优化避碰余量与五种掌位偏移；随后尝试三组四指横向分布，共 15 个第二轮候选。两轮均未找到通过候选，失败包括指尖目标不可达和几何拒绝。

所有请求、原候选结果、两轮搜索源码和失败结果保存于 `evidence/teacher_bootstrap_20261103/`。未将失败候选接入实际右手控制，也未降低碰撞或接触门槛。

## 实际物理验证与运行

本轮启动了完整传统 teacher 回归 `teacher_continue_baseline_20261103_1905`（开发 seed 9501，录像）。左手 BC 闭环开发评估 `teacher_left_bc_eval_20261103_9901` 在它退出后串行启动，保持同时只有一个 Isaac 进程。评估采用学习到的左手输出和原示范的右手时间序列，因此是混合控制开发检查，不能称为纯双手神经策略或独立成功率。

运行中日志位于服务器 `/home/brainco/rl_cube/github_project/work/teacher_continuation_20261103/`，分别为 `baseline.log`、`left_bc.log`、`left_bc_eval.log`。最终状态应读取对应 run 的 `result.json` 或 `error.json`；启动队列不能视为评估通过。后续必须先完成右手可达、五指抓持及真实转层验收，再收集对应状态示范。

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
python teacher_bootstrap.py --source runs/center_j56_dev_v3_20261103 --output runs/UNIQUE_BC_RUN --updates 6000
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
python evaluate_teacher_bootstrap.py --checkpoint runs/UNIQUE_BC_RUN/best.pt --source runs/center_j56_dev_v3_20261103 --output runs/UNIQUE_EVAL_RUN --seed 9901
```

验证：原有 36 项相关离线测试通过；新增 4 项数据对齐、哈希变更、失败回合拒绝及非有限目标拒绝测试通过；三个新 Python 文件语法检查通过。56 维环境动作接口、速度/限位、安全与阶段门禁没有修改。
