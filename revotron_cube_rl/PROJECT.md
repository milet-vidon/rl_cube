# 双臂 Revo3 + Tianji 魔方 RL 仿真项目

## 项目目标

在 Isaac Sim/Isaac Lab 中使用双臂 Revo3 与两只 Tianji/Revo3 手，通过胸部相机完成：

1. 视觉定位魔方；
2. 双手稳定抓起并保持；
3. 双手接触转动指定魔方层；
4. 连续跨面操作；
5. 读取色块并完成三阶魔方复原；
6. 经过 sim-to-real 标定后迁移到真实机器人。

## 固定接口

- 动作空间：56 维，左右各 `7 个机械臂关节 + 21 个手部关节`。
- 相机：胸部 RGB，当前仿真分辨率 128×128。
- 控制：Isaac PhysX 240 Hz，目标更新 30 Hz，隐式 PD 执行。
- actor：RGB/视觉特征 + 224 维本体觉；不能读取魔方真值。
- critic：训练时可读取特权状态，但不能泄露给 actor。
- 每个实验必须保存配置哈希、数据集哈希、seed、trajectory、result.json 和验收指标。

## 当前验收边界

- 传统 56 关节抓取 teacher：已通过。
- 状态感知 RGB teacher 抓取：seed=1945 已通过。
- 状态感知 RGB teacher 可视化回放：seed=1945 已完成，已保存胸部/全景 GIF 和成功帧。
- free-core 外部接触转层：CPU PhysX 的 `+Z 90° → +X 90°` 已通过。
- Revo3/Tianji 双手实际转层：尚未通过。
- 转层 RL：尚未开始有效训练。
- 54 色块视觉识别、连续换抓和三阶复原：尚未接入。

## 后续迭代顺序

1. 优化 `robot_hand_turn_stage.py` 的 free-core + Revo3 接触场景，使双手接触 probe 在有界时间内完成；
2. 将外部推板转层替换为 Revo3/Tianji 双手接触；
3. 固化转层动作接口、层角度和接触安全门槛；
4. 采集抓起→预抓位→转 90°→释放/换抓 teacher 数据；
5. 做时序/RBF/检索蒸馏，再做小步长 PPO；
6. 用未参与采集的 seed 做独立验证；
7. 通过后再加入随机打乱、跨面和复原规划；
8. 最后进行相机、关节、PD、延迟和摩擦标定，再做 sim-to-real。

## 唯一状态来源

- `current_status.json`：机器可读状态；
- `STATUS.txt`：人类可读训练记录；
- `active_training_process.json`：当前/最近一次训练进程；
- `FRAMEWORK_CHAIN.md`：系统链路和 sim-to-real 边界；
- `runs/<name>/result.json`：单次实验的唯一结果记录。

后续训练、模型修改和实验记录都在本项目目录及本项目主线程中进行，不再把它们分散到源对话。


## 2026-11-02 用户软限位与 PPO round1

当前双臂 14 个机械臂关节使用用户指定的 Revo3 软限位（左右臂相同）：J1/J3/J5 ±178°、J2 ±120°、J4 [-145°,+60°]、J6 ±60°、J7 ±90°。运行时还与 URDF/PhysX 硬限位求交，J4 下限因此被微小浮点裕量截到硬限位；控制层保留每步 0.04 rad 目标变化上限，机械臂速度上限缩放为 0.5。限位审计和运行时硬限位一致性均通过。

在该安全契约下重新训练了 `runs/user_softlimit_warmstart`，再做真实胸部 RGB + PhysX 的 `runs/user_softlimit_ppo_round1`（40 个 PPO 迭代、2560 步、4 个 episode），没有成功抓取。独立闭环 `runs/user_softlimit_ppo_round1_eval` 为 0/1，魔方未抬离桌面，故不能把本轮策略称为已学会。当前唯一通过视觉抓取门槛的仍是 state-aware kNN teacher；下一轮应优先补充抓取阶段/接触访问状态或采用 teacher-guided residual，而不是放宽软限位。


## 阶段感知 actor 的闭环边界

显式把 episode phase 拼到 RGB+本体觉后，离线 holdout action MSE 明显下降，但 `phase_rgb_warmstart_eval` 仍为 0/1，且出现约 1.455 N·s 的机器人-桌面冲量。因此这个模型不进入 PPO 主线；下一步使用成功 kNN teacher 的在线查询/残差蒸馏，让训练样本覆盖真实接触分布，再逐步释放策略自由度。

## 当前训练结论（2026-11-02）

- 用户软限位已固定并通过审计：双臂 14 个关节均受限，J4 下限与硬限位求交。
- 新限位下普通 RGB PPO round1：2560 步、4 episode、0/4；独立闭环 0/1。
- 显式 phase RGB actor：离线 holdout MSE 0.000699，但独立闭环 0/1，并产生 1.455 N·s 机器人-桌面冲量，未进入 PPO。
- teacher-guided residual PPO：残差 0.1 时 0/2，残差 0.01 时 0/1；说明成功 teacher 接触轨迹对偏差敏感。当前下一阶段是接触/phase 门控的 teacher 权重退火，不再盲目增加 PPO 轮数。

## 分阶段抓取策略审计（2026-11-02）

项目新增 `StagedDownwardController`，将抓取拆成 `raise → orient → lower → settle → close → lift → hold`。在 orient/lower/settle 阶段，手指采用预抓姿态，等 7 轴末端误差和速度稳定后才驱动拇指与四指闭合。平行模式在 seed=17、预抓 35% 时可以抬升 10.8 cm 并保持 30 步，但机器人-桌面冲量为 55.64 N·s，未通过安全门；预抓 50%/65% 未能抬起。斜向 12° 目标在当前用户软限位与接触点下 IK 不可行，位置误差约 6.8 mm。因而 DAgger 默认仍保留已验证的 legacy downward teacher，分阶段 teacher 只作为待优化候选。
