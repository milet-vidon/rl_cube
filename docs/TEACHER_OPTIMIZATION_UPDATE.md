# Teacher 联合优化与闭环训练（2026-11-03）

本轮接续 `a0b4973`，优先处理真实回合中的失败，而非以离线拟合误差作为成功。

## 已定位的问题

首版左手 BC 在 seed 9901 的第 457 步闭指阶段碰桌，累计该步机器人/桌冲量 0.2694 Ns，立即停止。状态扰动训练 v2 在第 261 步请求超出软限位，实际关节仍在限位内、无碰桌/自碰。两者均不晋级。

加入 tanh 有界输出的 v3/v4 训练后，离线目标拟合变差；因此增加可选择的 BoxOutput 网络层，在训练和推理中按相同安全交集约束输出，同时保持原标准化。没有修改环境的限位、目标速度、碰撞检查或成功门槛。v3/v4 结果保留，没有用于成功宣称。

v5 使用 0.02 rad 关节状态扰动、相应位姿扰动和原示范的控制时间，训练 40000 次。最佳同回合时间块目标 RMSE 0.00667349 rad。2128 个额外扰动状态的输出范围检查、旧 checkpoint 加载兼容、checkpoint 重载验证和 40 项相关离线测试通过。只有一条训练示范，尚不能推导独立成功率。

## 右手规划改进

旧规划先固定镜像掌姿、再依次拟合每根手指，两个约束相互冲突：中指/无名指最小间距约 2.466 mm，小于原 3 mm 门槛；右掌朝左手余弦为 0.451，小于原 0.65 门槛。

新增 `opposed_joint_fit.py`，联合优化右臂与五指，目标包含掌向、指尖目标区域、非交叉构型、手指间距和固定碰撞体间距。优化结束后仍执行原指尖可达、掌姿、闭指路径、全路径碰撞及转层路径检查。没有放宽这些检查。

同一最新居中实测请求上，新候选右掌朝左余弦达到 0.69791，左掌朝右余弦为 0.86342，完整 1497 次路径检查通过；原回合的 1205 次左手居中预检仍保留。离线通过不是实际接抓或转层成功。

入口增加显式 `--opposition-planner joint`，默认 baseline 保留。联合规划器及优化器均纳入物理回合的源码归档和哈希记录。

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
./run_linux.sh face-turn --opposition-planner joint --center-style j56 --name UNIQUE_JOINT_RUN --seed 9501
python teacher_bootstrap.py --source runs/center_j56_dev_v3_20261103 --output runs/UNIQUE_BC_RUN --updates 40000 --state-noise .02 --box-output
```

## 已完成的真实物理结果

`teacher_joint_opposition_v1_20261103` 完整运行 3250 步。左手居中保持通过，右手实际完成接近、闭指并保持拇指/食指/中指/小指的目标层接触，掌姿合格，没有自碰、碰桌或限位命中。无名指未触及，因此五指同窗保持为 0 秒，原门禁拒绝转层。实际无名指锚点局部 y=0.028968 m，距边长 57 mm 魔方的表面约 0.47 mm；后续开发候选对无名指目标内移 1 mm，但仍执行原路径/接触检查。

v5 模型完整评估 `teacher_left_bc_box_v5_eval_9901` 运行 2129 步，没有此前碰桌和限位失败，但未抓起魔方、居中保持 0 秒，结论为 `center_hold_gate_not_met`。这说明单条示范训练不能证明泛化，不晋级。

## 持续运行的后续任务

- `teacher_joint_opposition_ring_v2_20261103`：仅在修正候选完整预检通过后，执行从 reset 开始的右手接抓/转层开发回合。
- 串行采集开发 seeds 9901、9502、9503 的完整左手居中示范。41 项相关离线测试通过。
- 仅选入完整、无安全失败且通过原抓持/居中验收和证据哈希检查的新增示范；没有合格新增示范就明确停止，不以失败回合训练。
- 有合格新增示范时，以 `--extra-source` 合并示范，使用 `--applied-targets` 对齐实际施加目标、BoxOutput、0.005 rad 扰动继续训练 100000 次，保存 `teacher_left_bc_multiseed_v6_20261103`，再串行运行 9901 开发闭环评估。重复源拒绝，新回合均使用唯一名称，训练脚本自动随 run 归档。

所有物理任务按队列串行执行。结果以各 run 的 result.json/error.json 为准，日志和队列脚本保存在服务器 `github_project/work/teacher_optimization_20261103/`。已完成结果及 checkpoint 见 `docs/evidence/teacher_optimization_20261103/`。所有 seed 均按开发使用，9901 已见，不能再次称为新独立测试。当前没有完成单层转动或双手神经策略成功。

无名指 1 mm 内移候选完整预检已通过，第二轮真实回合已启动；其结果仍需等待实际接触验收。
