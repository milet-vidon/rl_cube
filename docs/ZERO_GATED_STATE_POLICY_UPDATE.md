# V7 状态策略训练与闭环结果（2026-11-04）

V7 以实际接触状态学习“保持精确基线”或有界残差。三个候选已完成 GPU 行为克隆训练；三个真实闭环均在 752 步的 primary_lift 阶段被原 contact_topology_or_settle_timeout 拒绝。长轮次候选离线误差更低，但实际闭环仍在同一门槛失败。尚未得到可靠转层、跨初始状态鲁棒策略或视觉复原模型。

## 训练定义与已完成检查

全部训练使用五段已审核开发轨迹，共 5,425 帧；物理种子为已见 9501。第二候选的 20261105 只改变训练初始化，不构成新的独立物理测试。模型为冻结 V4 基线加 284 维实际观测的 GRU，28 个左关节分别学习 exact-baseline/continuous-correction 分支。严格零标签来自原 float32 动作投影逐位相等；非零差异超过 1e-6 rad 才进入确定修正类别，其余模糊标签仅不进入分类损失，原连续目标全部保留。

原关节顺序、投影、软限位、速度限制、碰撞和接触门槛保持不变。左手位置动作从 reset 起由网络执行；原 teacher 仍负责公开阶段、右手动作以及声明的 K20 增益事件。因此不能称为全机器人纯神经策略或 PPO。

| 候选 | epochs / best epoch | CPU 分类 FP / FN | CPU 动作 RMSE (rad) | 真实闭环 |
|---|---:|---:|---:|---|
| V7 seed20261104 | 5000 / 5000 | 9 / 12 | 7.11948e-5 | 752 步失败，最大抬升 0.187806 m |
| V7 seed20261105 | 5000 / 5000 | 16 / 6 | 6.37745e-5 | 752 步失败，最大抬升 0.187794 m |
| V7 long seed20261104 | 10000 / 9600 | 4 / 0 | 2.29933e-5 | 752 步失败，最大抬升 0.187905 m |

三个检查点均经独立 5,425 帧 CPU 顺序重放，指标与保存结果一致；四段应保持零修正轨迹在第 499 步逐位等于冻结基线。安装到 revotron_cube_rl 的 6 个新模块与 6 个测试保持原冻结字节，79 项相关 unittest 通过。离线拟合改善不证明实际操作成功。

## 已定位的闭环限制

首 V7 的 752 帧实际 actor 重放逐位一致，0..544 步的实际命令、关节、魔方位姿和接触与成功 V4 开发对照一致。第 545 步开始分歧。原 FingerArrival 在重新接触后保持上一实际目标，但 V7 的修正围绕持续移动的 V4 基准；从第 546 步开始，精确保持需要约 0.009449 rad 的残差，超过当前 0.006 rad 的指关节范围。

全回合有 297 个 active 原专家投影标签超出该表示范围，最大投影缺口 0.00431049 rad。因此继续延长同一结构训练不能消除这一限制。下一步先严格审核新的实际失败状态，再验证包含精确 HOLD(previous) 分支与有界增量动作的表示；原标签和原回合失败标记不改写。

原实际数据仍标记为非合格 RL 成功回合；加入监督训练需要另建可复核的资格记录，不通过更改原 result.json 授权。

## 复现与证据

工作目录为 /home/brainco/rl_cube/github_project/revotron_cube_rl。先 source /home/brainco/rl_cube/activate.sh，再显式设置 REVOTRON_USD_PATH 为 github_project/revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda、ISAAC_MUJOCO_CUBE_ROOT 为 github_project/isaac_mujoco_cube。新回合必须使用唯一目录，单机只运行一个 Isaac 进程。

源文件 zero_gated_residual_policy_v7.py、eval_prefix_dataset_v7.py、inspect_residual_dataset_v7.py、train_zero_gated_residual_v7.py、state_zero_gated_eval_runtime_v7.py、evaluate_state_zero_gated_v7.py 已安装。训练入口要求显式 hash 绑定的五段数据资格和独立 root review；这些具体参数、运行命令、全部优化日志、源快照与检查点保存在训练归档中。训练资格仍不等同于物理任务验收。

- docs/evidence/zero_gated_v7_training_20261104：首 5,000 epoch 完整训练。
- docs/evidence/zero_gated_v7_seed20261105_training_20261104：第二初始化完整训练。
- docs/evidence/zero_gated_v7_long_training_20261104：10,000 epoch 完整训练。
- work/teacher_optimization_20261103/v7_actual_contact_feedback_diagnosis_20261104.json：297 个不可达标签及接触保持诊断，SHA256 24f37c3fce2e7bf2e9b3257fb3c77e2f43c445c4f88232eb2cf87f79e907b4b2。
- work/teacher_optimization_20261103/v7_first_actual_actor_replay_20261104.json：首实际 actor 重放，SHA256 31888c2e0096f29a83c89bcdd5a584f73b19482af4517735015a7d98edc4204a。

第二候选首次启动因命令使用不存在的 best.pt 而在仿真前退出；原空回合目录和日志保留。真实检查点为 model.pt，成功启动的回合名为 left_zero_gated_residual_v7_seed20261105_eval_9501_20261104_v2。

## Native 转层支线

native V20 已完成 3,019 步，在 right_close 被原 left_grasp_lost_during_opposition 拒绝，未开始转层。将食指 MPR 下界从 -0.012 扩至经几何预检的 -0.023，短暂延长接触但没有避免后续失触。原 FullTask 终止、质量/惯量/质心与 66 个被动关节参数保留，控制期魔方位姿写入为零。完整运行与失败证据在 docs/evidence/native_index_extended_v20_20261104。

独立箱体接触观测探针仍在校准，尚未集成机器人控制。已有失败和初始化问题分别保留；不能把探针结果当作机器人接触转层或力矩控制成功。
