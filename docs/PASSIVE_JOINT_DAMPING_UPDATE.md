# 源关节被动阻尼迁移（2026-11-03）

当前仍以“先实现可靠转层，再训练操作策略，最后接视觉复原”为顺序。最新完成的 v11 右手五指合格保持 4.033 秒，实际目标层角度约 2.305°，未通过 15° 转层保持；不能把该回合用作成功转层示范。

## 改动及边界

源 MJCF 有 6 个 hinge 和 20 个 ball 关节，继承各轴阻尼 0.0005 N·m/(rad/s)。原 USD 中的普通球关节不支持 drive，直接附加阻尼属性不会实现所需效果。

新增默认关闭的 `--source-joint-damping`：

- 首次 PhysX reset 前，检查源 XML 哈希、继承关系、关节数量/类型和 USD 连接关系；拒绝未知类型、有限角球关节、已有 drive/自定义 D6 limits、非正或各向异性的阻尼。
- 将 20 个无限角球关节表示为平移锁定、三轴旋转自由的 D6；保留原连接刚体、关节坐标系、碰撞开关和其余属性。6 个 hinge 保持原类型。
- 66 个角向轴采用 force 型纯阻尼：刚度 0、目标速度 0、目标位置 0。阻尼换算为约 8.726646e-6 N·m/(deg/s)。不指令魔方转角，不向控制循环添加魔方位姿写回、外部力或转动目标。
- 不修改 Core 惯量、质量、重力、手部增益或摩擦，除非命令另行指定其他独立选项。当前完整对照使用 v8 的指部刚度 20、位置迭代 64、默认内部摩擦及扫掠间隙优化，仅增加被动阻尼。

这是**部分动力学迁移**，尚未迁移 source armature 和 frictionloss，不能声称等价复现整个源物理模型。D6 与原球关节的数值求解行为也可能不同，必须重新验证完整操作；大角度、多次换面和实际机器人参数仍未校准。

API 依据：[Omni Physics 关节支持与 D6 约束](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html)、[OpenUSD DriveAPI 的单位](https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html)。阻尼只抵抗相对速度；目标转层仍必须来自机器人手部接触。

## 验证

本轮 69 项相关离线测试通过。新增 5 项使用真实 USD 资产的内存会话层，检查单位/数量、非法源参数拒绝、连接/坐标系/碰撞属性保持、非零主动目标拒绝和初始化工厂恢复；不写回原资产。

`passive_joint_verification.json` 检查已声明参数，不把 USD 属性读取当作物理效果已通过。`teacher_joint_passive_damping_v12_20261103` 已启动完整 PhysX 回合，初始化验证 66 个轴，未见关节/drive 配置错误。最终抓持和转层结果以 result.json 为准，当前不报告成功或新增神经策略 checkpoint。

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
