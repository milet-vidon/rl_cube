# 转层动力学与接触对照（2026-11-03）

目标仍是先实现左手稳定拿持、右手可靠转层，再训练可组合技能，最后接视觉状态识别与公式求解。下列均为 seed 9501 的开发实验，不是新独立评估。尚未得到合格 15° 示范，不能据此宣称自由旋转或启动该能力的成功示范训练。

## 完整物理结果

| 实验 | 唯一主要改动 | 实测结果 | 结论 |
|---|---|---|---|
| ring v2 | 联合手臂/手指规划、无名指目标内收 1 mm | 右手五指合格保持 3.50 s，目标层最终 5.687° | 15° 保持未通过 |
| compliance v5 | 以 right traction v4 为基准，右手 21 指关节刚度 30→20 | 左手居中通过，右手五指合格保持 4.033 s；九块最终转角 5.264–5.887°，中位 5.452°；目标层最大角速度 11.875 rad/s | 3511 步完整回合，转层保持 0 s |
| source core v6 | 以 v5 为基准，仅按源 MJCF 球体推导 Core 惯量 | PhysX 惯量读取校验通过，但 563 步在初次抬升丢失主抓；最大抬升 6.36 mm，未进入右手控制 | 不能采用此配置或声称动力学已修复 |
| precision v7 | 以 v5 为基准，位置求解迭代下限 32→64 | 左手居中保持 8.033 s；右手预检发现食指与目标层预测距离 -2.025 mm，超过原允许穿入 2 mm | 2082 步，右手未执行，不能比较转层表现 |
| swept clearance v8 | 以 v7 为基准，加入闭合及转动扫掠间隙优化 | 1514 次路径检查通过，五指接触比例 100%；121 帧保持中 82 帧核心角速度 ≥0.20 rad/s，最长合格保持 0.133 s | 3202 步完整回合，未执行转层 |
| matched friction v9 | 内部摩擦 (.4,.3)，匹配外部接触系数，其余同 v8 | 抓取后最终最大层偏移 5.433°，超过 5° 装配门槛 | 1826 步完整回合，前置抓取未验收，未进入右手控制 |
| matched mid v10 | 内部摩擦 (.6,.5)，其余同 v9 | 初次抬升后丢失左手主抓 | 666 步完整回合，未进入右手控制 |

所有回合的控制中魔方位姿写入均为 0。默认 Core 惯量未改；新选项仅供可复现开发对照。





## 接触与速度诊断

`audit_turn_dynamics.py` 读取完整轨迹，不修改验收逻辑，输出轨迹 SHA256、逐阶段接触比例和速度统计。

v5 转动阶段：拇指/中指接触比例 100%，食指 45.7%、无名指 54.3%、小指 30.5%；最终保持阶段后三指接触比例均为 0%。与 v2 相同的失触模式仍存在，增加右手摩擦并降低刚度尚未解决转动中滑移。

在 v5 最终保持阶段，Core 端点角速度中位数约 0.254 rad/s，30 Hz 姿态差分平均角速度约 0.00150 rad/s；nZ 中心块分别约 3.733 与 0.00159 rad/s。二者一个是端点速度、一个是控制间隔平均值，不能仅据差异断言速度读取错误；可能涉及高频运动、求解或采样。原验收继续使用原始速度，未用差分或滤波替换速度门槛。

## 源模型与运行时审计

初始化属性探针确认 Core 质量约 0.002537037 kg，实际对角惯量约 1.014815e-5 kg·m²。源 MJCF 的 Core 为半径 0.01 m 的球体；保持运行时质量，用 I=2/5·m·r² 得到 1.014815e-7 kg·m²，相差 100 倍。新增 `--source-core-inertia` 在首次 PhysX reset 前显式设置惯量，随后读取实际惯量核验；不更改质量、碰撞、重力或动作，也不修改冻结的环境源文件。

这只迁移 Core 惯量，**不等于完整迁移源动力学**。源关节 armature=0.0001、damping=0.0005、frictionloss=0.001 尚未等价迁移。当前魔方是 27 个独立刚体通过关节连接；PhysX 的 armature/关节摩擦参数仅对 articulation 内关节生效，不能只给普通关节写同名属性就声称修复。参见 [PhysxJointAPI](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_joint_a_p_i.html) 与 [PhysxJointAxisAPI](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/joints/physx_joint_schema.html)。

运行时 TGS 已启用每次迭代外力，stabilization 为 false；没有重复添加已生效的开关。新增 `--position-iterations 64` 保持 240 Hz 物理、30 Hz 控制、TGS 和 4 次速度迭代，仅将场景位置迭代下限从 32 改为 64。`solver_verification.json` 验证实际场景属性；该选项用于数值收敛对照，不是放宽任务条件。

## 接触材料对照设计

已导出场景确认：默认机器人材料 (.5,.5)、魔方 (.9,.8)、桌面 (.8,.7)，原合成为 average。左手—魔方的有效系数为 (.7,.65)，桌面—魔方为 (.85,.75)。右手单独材料仍采用 max 合成的 (1.2,1.0)。

新增可选匹配材料试验：降低魔方内部摩擦，同时将左手和桌面改为上述有效系数、max 合成，以保持这两类外部接触的库仑摩擦系数。代码先核验原材质、默认值和左手绑定，再验证新绑定；不满足假设就失败。它不保证轨迹相同，不是实体材料标定，也没有改变接触、碰撞、限位或速度验收。内部摩擦仍会影响抓取时的层形变，需要从 reset 重新验证整个回合。

## 复现与证据

环境准备：

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
```

v5 配置公共部分：

```bash
./run_linux.sh face-turn --opposition-planner joint --ring-inset .001 \
  --center-style j56 --right-hand-friction 1.2,1.0 --right-finger-stiffness 20 \
  --name UNIQUE_NAME --seed 9501
```

v6 另加 `--source-core-inertia`；精度对照另加 `--position-iterations 64`，不同时添加源惯量选项。每次使用新名称，禁止同时运行多个 Isaac 进程。

原始结果、压缩轨迹、源码快照和运行配置位于 `docs/evidence/teacher_optimization_20261103/`，对应目录与 run 同名。总 `manifest.json` 记录证据 SHA256。两个初始化属性探针仅证明读取到的物理属性，不证明完整操作成功；完整行为结论来自完整回合。

## 本轮规划修正

`--cube-clearance-refinement` 为联合抓型优化增加目标层与静止两层的碰撞距离残差。目标层 DIP 接触优化目标为不深于 1.5 mm，其他部位间距至少 2.5 mm，均比原最终路径门槛严格 0.5 mm；最终仍由原来的 2 mm 门槛与完整路径检查决定是否执行。该改动解决优化目标未直接覆盖魔方表面几何的问题，不通过放宽碰撞阈值接受 v7 被拒绝的构型。

当前 64 项相关离线测试通过；新增间隙模块另有 3 项主机测试，物理成功另行判定。

首次仅约束闭合抓型的预检在转动第 70 步被拒绝：中指与静止层间距 1.992 mm，仍小于原 2 mm 门槛。随后加入转动 1/3、2/3、终点的手指扫掠间隙残差，完整 IK/碰撞路径通过 1514 次检查（111.71 s）。该三点约束用于优化；验收仍覆盖全部插值路径，并非只检查三个姿态。`teacher_joint_swept_clearance_v8_20261103` 已完成完整 PhysX 回合，使用 v7 的 64 次位置迭代和新间隙选项，不改变惯量或内部摩擦。五指全部接触，但核心角速度限制阻止转层，见上表。

匹配材料入口：在上述公共命令后增加 `--matched-internal-friction .4,.3`。尚无此选项的完整物理成功结论。

## 当前单次对照

`continue_contact_comparison.py` 已验证 v8 完整结果、原始 manifest 和当前源码哈希，串行启动 `teacher_joint_matched_friction_v9_20261103`，仅将内部摩擦改为 (.4,.3)，外部接触系数按前述规则匹配。初始化日志确认左右手各 77 个碰撞体的材料绑定，v9 完整回合已结束：层偏移超过装配门槛，前置抓取未验收。控制脚本已完成，只安排这一项对照，不会启动 BC/PPO，不会把失败轨迹加入成功示范；若 v8 因碰撞/前置失败、源码变化或证据损坏，则不会启动。

配对系数计算依据 [PhysX 材料合成规则](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html)。

基于 v8 的保持抖动和 v9 的装配偏移，另启动 `teacher_joint_matched_mid_v10_20261103`：内部摩擦 (.6,.5)，左右手与桌面的配对系数、64 次位置迭代、右指刚度 20、间隙优化及全部门槛均保持。该回合已结束，666 步时 `lost_primary_grasp`，日志 `work/teacher_optimization_20261103/matched_mid_v10.log`。命令在 v9 参数基础上将 `--matched-internal-friction` 改为 `.6,.5`，使用上述唯一新名称。尚未报告成功或生成新的操作策略 checkpoint。

## 最新运行

两种较低内部摩擦均未通过前置验收，恢复 v8 已通过左手拿持的默认内部摩擦 (.9,.8)，另启动 `teacher_joint_soft_fingers_v11_20261103`，仅把右手指部刚度从 20 降至 10；保留右手摩擦 (1.2,1.0)、64 次位置迭代、扫掠间隙优化和原全部门槛。PID 347163，日志 `work/teacher_optimization_20261103/soft_fingers_v11.log`。这是传统控制开发回合；尚未生成可自由转动的神经策略。

已进一步核对源文件为 6 个 hinge、20 个 ball 关节，均继承各轴阻尼 0.0005。普通 USD SphericalJoint 不支持 drive；因此未尝试只写阻尼属性来假装迁移成功。后续需要单独验证可表达被动阻尼的约束模型，保留平移锁定、旋转自由度、连接坐标系及碰撞。纯被动阻尼不能提供目标转动；也不能替代缺失 armature/frictionloss 的完整迁移。参考 [Omni Physics 关节支持](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html)，审计笔记已归档。

最新 v11 已完成：3474 步，右手保持 4.033 秒，实际面转角 2.305°，转层保持 0 秒；后续被动阻尼修复与 v12 对照见 [专门更新](PASSIVE_JOINT_DAMPING_UPDATE.md)。
