# V5：接近边界卡住与超时证据修正

V4 用户实测目录：`runs/r27_natural_face90_v4_probe_20261118_122716`。8 回合中，7 回合拿稳并自然换姿，0 次不安全终止，只有 1 回合触碰目标层，完整成功 0。5 回合在 outside 超时、1 回合在 hover 超时、1 回合在 sweep 超时，另 1 回合在原抓取阶段超时。六个接近失败回合均已完成向前绕行，后续停在约 25 mm 的避让边界，最后 100 帧间距逐值不变。

唯一进入 sweep 的回合有 412 个接触帧，平均目标层冲量约 0.00736 Ns，左手对目标层冲量为零，最终层角约 0.0292°。这些数据不能证明已学会单层拧转，也不能把后续转不动归因为左手压住目标层。

定位与验证：旧的基础运动学测试将右食指设为张开，未覆盖 R27 实际终点的弯曲手型。使用 seed 20000000、step 1545 的已录制 PhysX 刚体姿态重建关节后，V4 在离线理想跟踪中也接近超时。重建的右侧刚体与记录相比，位置最大残差 1.68 μm、角度最大残差约 0.000059°；这只是历史数据重建，不是新物理回合。

原控制器根据伸手前弯曲指尖的位置选择最近的面内方向，在该真实起点选中了靠机身的一侧。与 25 mm 前臂避让约束结合后发生冲突。还发现最终命令经过安全投影后，内部运动参考仍保留未执行的更新；这个反馈缺口已加入独立回归测试。V4 没有保存超时瞬间的关节/参考，所以不能仅从 V4 日志量化这两项在每个失败回合中的贡献。

V5 做三项修改：

1. 从 nZ 面朝机器人前方的一侧接近，方向转到 Core 局部坐标后固定，随持物姿态变化；不再依赖初始弯曲手指决定靠近哪一边。向前绕行和原 25 mm 避让约束保留。
2. 接近阶段以最终投影后的实际命令作为下一帧运动参考，避免反复提出已被截住的更新；不在约束求解后重复叠加跟踪滞后。原关节限位、速度、真实自碰终止及转层验收不变。
3. 新增 `controller_states.jsonl`：在交接、阶段变化、每 30 控制帧及所有终止时记录 56 维关节、命令、右臂参考、目标/实际点、目标误差、转弧命令和实测刚体位姿。无自碰的超时也有完整末态，文件哈希纳入验收。短测汇总新增 `terminal_phases` 和 `max_post_goal_error_mm`。

V5 尚未运行 PhysX。离线测试包括真实弯曲手型、三个姿态参数边界及延迟跟踪下的接近可达性，六个已见碰撞姿态仍被原避让约束排除；这些不证明实际成功。当前先验证安全接近能否恢复，接触后转不动仍是待解决问题。R27 权重及 V2/V3/V4 源码、结果均保留。

远端短测命令：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_natural_face90_v5_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v5_probe_$(date +%Y%m%d_%H%M%S)" \
  --smoke --lanes 8
```

这些仍是已见历史种子的开发回归。先看 `unsafe` 是否保持 0、`touched_target` 是否增加，结合 `terminal_phases` 判断是否通过接近阶段。若继续超时，查看对应 attempt 的 `controller_states.jsonl`，不要只增加阶段时长或原样扩大长训。`successes=0` 不能称为转层验收通过。

只有安全接近检查通过后才考虑长训：

```bash
bash run_natural_face90_v5_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v5_long_$(date +%Y%m%d_%H%M%S)" \
  --rounds 40 --population 8 --seeds-per-candidate 4 \
  --validation-seeds 8 --lanes 32
```

V5 使用独立源文件及新的 4400 万/6400 万训练、验证种子段；原独立测试登记和晋级规则不变。仍是 R27 神经抓取后接六参数反馈控制器搜索，目标是 15° 正向接触驱动单层转动与撤手保持。首次运行不用 `--resume`，不能恢复 V4。

离线验证：`/home/brainco/rl_cube/.conda/rl_cube_6_2/bin/python -B -m unittest test_natural_face90_v5 test_natural_face90_metrics -q`。远端 `bash run_natural_face90_v5_long.sh --check --run-dir /tmp/unused_natural_face90_v5` 只校验冻结输入，不启动 Isaac。助手未启动训练或仿真。
