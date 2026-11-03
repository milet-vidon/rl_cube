# 源关节线性阻尼近似（2026-11-03）

当前仍以“先实现可靠转层，再训练操作策略，最后接视觉复原”为顺序。最新完成的 v11 右手五指合格保持 4.033 秒，实际目标层角度约 2.305°，未通过 15° 转层保持；不能把该回合用作成功转层示范。

## 改动及边界

源 MJCF 有 6 个 hinge 和 20 个 ball 关节，damping 三数为速度多项式的线性、二次、三次系数，并非 xyz 各轴。当前只施加第一个线性项 0.0005 N·m/(rad/s)，高阶项未迁移。本机 MuJoCo 3.12 编译原 XML 验证了 66 个 DOF 的 dof_damping=0.0005 和 dof_dampingpoly=[0.0005,0.0005]；早期源码和归档中的“各轴/各向同性”解释不准确，历史证据原字节保留。原 USD 中的普通球关节不支持 drive，直接附加阻尼属性不会实现所需效果。

新增默认关闭的 `--source-joint-damping`：

- 首次 PhysX reset 前，检查源 XML 哈希、继承关系、关节数量/类型和 USD 连接关系；拒绝未知类型、有限角球关节、已有 drive/自定义 D6 limits、非法多项式阻尼系数（线性项须正，其余非负）。
- 将 20 个无限角球关节表示为平移锁定、三轴旋转自由的 D6；保留原连接刚体、关节坐标系、碰撞开关和其余属性。6 个 hinge 保持原类型。
- 66 个角向轴采用 force 型纯阻尼：刚度 0、目标速度 0、目标位置 0。阻尼换算为约 8.726646e-6 N·m/(deg/s)。不指令魔方转角，不向控制循环添加魔方位姿写回、外部力或转动目标。
- 不修改 Core 惯量、质量、重力、手部增益或摩擦，除非命令另行指定其他独立选项。当前完整对照使用 v8 的指部刚度 20、位置迭代 64、默认内部摩擦及扫掠间隙优化，仅增加被动阻尼。

这是**部分动力学迁移**，尚未迁移非线性阻尼、source armature 和 frictionloss，不能声称等价复现整个源物理模型。D6 与原球关节的数值求解行为也可能不同，必须重新验证完整操作；大角度、多次换面和实际机器人参数仍未校准。

API 依据：[Omni Physics 关节支持与 D6 约束](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html)、[OpenUSD DriveAPI 的单位](https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html)。阻尼只抵抗相对速度；目标转层仍必须来自机器人手部接触。

## 验证

本轮 69 项相关离线测试通过。新增 5 项使用真实 USD 资产的内存会话层，检查单位/数量、非法源参数拒绝、连接/坐标系/碰撞属性保持、非零主动目标拒绝和初始化工厂恢复；不写回原资产。

`passive_joint_verification.json` 检查已声明参数，不把 USD 属性读取当作物理效果已通过。`teacher_joint_passive_damping_v12_20261103` 已启动完整 PhysX 回合，初始化验证 66 个轴，未见关节/drive 配置错误。v12 已完成 3175 步：左手居中通过，右手五指在保持阶段 121/121 帧全部接触；核心角速度中位数 0.0877 rad/s、最大 0.1525 rad/s，121 帧均低于原 0.20 门槛。但是左手无名指在右手闭合接触后失触，保持阶段 121/121 帧缺失，合格保持 0 秒，因此未执行转层。没有新增神经策略 checkpoint。

复现（使用新的 run 名称）：

```bash
source /home/brainco/rl_cube/activate.sh
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
export REVOTRON_USD_PATH="$PWD/../revotron_revo3/source/revotron_system/usd/revotron_bimanual_revo3_wrist_mesh_v1.usda"
export ISAAC_MUJOCO_CUBE_ROOT="$PWD/../isaac_mujoco_cube"
./run_linux.sh face-turn --opposition-planner joint --ring-inset .001 \
  --center-style j56 --right-hand-friction 1.2,1.0 --right-finger-stiffness 20 \
  --position-iterations 64 --cube-clearance-refinement --source-joint-damping \
  --name UNIQUE_NAME --seed 9501
```

该 seed 已见，仅为开发对照。单机同时一个 Isaac 进程；冻结的原证书、安全门槛、56 DOF、关节限位和速度限制保留。


## 有界转角反馈与下一轮

v11 首次失去右小指接触时，实测掌部转角 1.213°、目标层 0.184°，领先约 1.029°。这只是已发生轨迹的诊断，不是反馈控制必然成功的证据。

新增默认关闭的 `--turn-feedback-pacing`：规划输出每个已检查目标对应的掌部转角；只有五指仍接触，且下一目标角度不大于实测层角加 0.75°，才推进到下一个关节目标。否则重复最后一个已检查目标，不生成新校正姿态。连续停滞超过 90 个控制步或总额外时间超过 360 步时显式失败，不跳到最终保持或标为成功。仍保留原 3 秒保持、角度、原始速度、接触、碰撞和关节限制验收。该限制控制计划推进，不保证真实掌部跟踪误差恒为零。

本轮 74 项相关离线测试通过（含 5 项反馈推进测试：落后时暂停/恢复、失触停止推进、连续停滞、总预算、无效输入）。

`teacher_joint_paced_passive_v13_20261103` 已完成 3175 步。保留源被动阻尼、将右手指部刚度 20→10 后，右手保持阶段五指接触仍为 121/121，核心角速度最大 0.1278 rad/s，所有帧低于 0.20 门槛；但左无名指仍有 121/121 帧失触，合格保持 0 秒。未进入转层，反馈推进实际调用 0 步，不能据此评价转角反馈是否有效。该结果也说明单纯把右指刚度从 20 降到 10 没有解决这一保持问题。复现时在上面的命令中改为 `--right-finger-stiffness 10` 并加 `--turn-feedback-pacing`，使用唯一新名称。

`teacher_joint_paced_baseline_v14_20261103` 已启动，PID 495093，日志 `work/teacher_optimization_20261103/paced_baseline_v14.log`。恢复 v11 能完成 4.033 秒五指保持的未迁移源阻尼配置，仅增加 `--turn-feedback-pacing`，隔离检验时间推进对转动滑脱的影响。与 v13 相比取消 `--source-joint-damping`；其余参数相同。仍为 seed 9501 开发对照。完成后先检查 result.json 与反馈实际调用步数，再决定是否具备训练用成功示范；当前没有成功转层或新自由旋转模型结论。


## 多项式语义核验

[MuJoCo 官方 joint damping 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-damping)明确三数是线性/二次/三次速度系数。新增解析测试接受不相等的多项式系数并记录未迁移项；对既有 XML 施加的 USD 线性阻尼仍为约8.726646e-6，没有改变现有物理数值。编译和来源证据见 source_mujoco_damping_semantics_audit.json。
