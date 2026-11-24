# V4：右前臂与固定机身接近路径修正

用户运行的 V3 短测：`runs/r27_natural_face90_v3_probe_20261118_103614`。8 回合中，7 回合完成 R27 拿稳与自然换姿，6 回合在 outside 阶段出现真实原生接触 `Link5_R <-> world`。另 1 回合在原抓取阶段超时；唯一到达目标层的回合接触 418 帧，层角峰值仅 0.07632°，在 sweep 超时，完整成功仍是 0。

`world` 是 USD 中固定机器人刚体，包含基座、腰胸、肩座、颈和头部碰撞几何，不是地面。原生报告已确认右前臂 Link5_R 撞固定机身，但没有定位到 world 下的单个子碰撞几何。按关节角重建的 URDF 凸包距离未复现该接触；V3 也未保存物理刚体姿态，不能把某个躯干子网格猜成已经确认的根因。

V4 保留 R27、原物理资产、所有碰撞/限位/接触及转层验收，单独新增以下路径控制：

1. 右手在原位成形后，先向前移动至少 150 mm 并适当抬高，保持在右侧；实测到达后才横向接近魔方。
2. 以固定腰胸碰撞网格的外包盒及 Link5_R 网格的保守球串包络作为附加规划避让区。三份几何/URDF 哈希固定，离线检查确认包络覆盖这些网格顶点；该避让区覆盖本次六个实测碰撞姿态，但不是整个机器人的碰撞认证。
3. 通过右臂冗余自由度调整姿态，建立至少 25 mm 的包络间距。参考及最终命令都做检查；如果初始已接近附加包络，只允许间距不再恶化，不能直接跳到安全姿态。限速采用整体缩放，防止逐关节截断后反转避让方向。失败则保持/超时，不关闭碰撞继续推。
4. 保留碰撞对与 12 帧前序记录，新增 PhysX 实测 world/Link4_R/Link5_R/Link7_R 刚体位姿，便于核对关节运动学和实际几何的差异。逐帧日志新增 `right_forearm_clearance_m` 和 `right_detour_done`。

本版尚无实际 PhysX 结果。38 项离线测试通过，包括六个已见碰撞姿态回归、距离梯度、原网格包络检查、原关节/接触/计分契约，以及三个参数边界下的运动学到达和避让间距检查。离线理想跟踪不是物理成功；新避让约束也不证明其它碰撞对已经消除。接触后转不动的问题仍待安全接近后继续定位。

助手未启动训练或仿真。请先在远端执行短测：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_natural_face90_v4_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v4_probe_$(date +%Y%m%d_%H%M%S)" \
  --smoke --lanes 8
```

这 8 个历史种子是开发回归，不是新的独立测试。结束后查看终端汇总和 `diagnosis.json`：先确认 `unsafe=0`，且大多数回合完成换姿、能接近目标层。若仍自碰，查看 `collision_events.jsonl` 的实际刚体姿态；若无自碰但超时，结合 `right_detour_done`、`controller_phase` 和包络间距区分绕行卡住与拧层无进展。

只有上述接近检查通过后，才考虑长训：

```bash
bash run_natural_face90_v4_long.sh \
  --run-dir "$PWD/runs/r27_natural_face90_v4_long_$(date +%Y%m%d_%H%M%S)" \
  --rounds 40 --population 8 --seeds-per-candidate 4 \
  --validation-seeds 8 --lanes 32
```

V4 为独立文件，不覆盖 V3，不恢复 V3 的 run。训练/验证使用新的 4300 万/6300 万种子段，原独立测试登记与资格门槛保留。仍是 R27 神经抓取后接六参数反馈控制搜索，目标是 15° 接触驱动单层转动与撤手保持，不能称为 90° 转面或实机能力。

离线验证：在项目 `revotron_cube_rl` 目录运行 `/home/brainco/rl_cube/.conda/rl_cube_6_2/bin/python -B -m unittest test_natural_face90_v4 test_natural_face90_metrics -q`。远端 `bash run_natural_face90_v4_long.sh --check --run-dir /tmp/unused_natural_face90_v4` 仅校验冻结输入和几何，不启动 Isaac。
