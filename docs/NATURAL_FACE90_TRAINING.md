# R27 拿起 → 自然换姿 → 右手拧 nZ 单层（90 mm）

这条训练线用于 PhysX 仿真。R27 的神经网络和原始抓取场景冻结；学习的是接在它后面的六维、受限的反馈策略参数（交叉熵搜索）。每个候选参数向量执行真实回合。原模型的“整颗转动”是相对世界的姿态变化，并不是这里的转层角。

**在 5090 上运行**（也可在装有同套环境且空闲的本机运行）：

```bash
cd /home/brainco/rl_cube/github_project/revotron_cube_rl
bash run_natural_face90_long.sh --run-dir "$PWD/runs/r27_natural_face90_v2" \
  --rounds 40 --population 8 --seeds-per-candidate 4 --validation-seeds 8 --lanes 32
```

5090 上 32 路正好合并一轮的 8×4 个训练回合；若显存不足，在新的实验目录用 `--lanes 16` 分两波。第一次运行要求全新目录。意外中断后用相同参数追加 `--resume`；会检查源码和配置指纹，跳过有验收记录的波次。GPU 忙时会退出并提示重试，不停止已有任务。仅检查输入可运行 `bash run_natural_face90_long.sh --check --run-dir /tmp/unused`，不建实验目录、不启动 Isaac。单独诊断可用 `--smoke --run-dir "$PWD/runs/r27_natural_face90_smoke_v2"`；使用两个 R27 已知种子，不参与训练或泛化结论。

**两个阶段怎样验收。** 左掌相对 R27 终点向中线至少移动 18 mm，魔方也向中线至少移动 10 mm，掌姿实际改变 4–35°；需要双掌朝向和两肘位置合格、魔方受左手支撑，连续稳定 30 帧，右手才从待机位置接近。接着检查接触冲量归属：右指在 nZ 目标块上的实际接触期间，单层相对中心的有符号净转动必须达到 +15°，九块角度的最大分歧不超过 5°，另外十七块不能跟转超过 5°。右指撤退至少 20 mm、右手不再接触，左手保持支撑，持续 90 帧（3 秒），且从达标峰值回落不超过 2°，才算完整成功。这些是可测的仿真姿态条件，不是对“姿态像人”的主观评分。

**如何读结果。** 每轮八组策略在相同的四个新训练种子上比较，胜者和现任在另一组八个配对种子上验证。失败回合也进分母。每轮的 `selection.json` 保存两套成绩；`best_candidate.json` 是当前最佳候选，不等于合格策略。默认若连续四轮，所有候选的整组安全接触进度均不到目标的 20%（相当于平均 3°），状态记为 `no_safe_layer_progress` 并停止；若连续六轮验证组平均进度没有累计再提升 1°，也会停止。若验证至少 7/8 完整成功且无不安全回合，把参数冻结，第一次也是最后一次跑 32 个独立起点；至少 28/32 成功且无不安全回合才写 `qualified_policy.json`。独立测试失败即结束该轮实验，不能再用同一组测试种子调参。

**定位哪个动作没学会。** 在训练目录下看 `round_*/{train,validate}/wave_*/attempt_*/report.json` 的每条 `rows`：`prefix_endpoint_held` 是 R27 是否真实拿稳并到达腕部终点；`natural_ready` 是换姿是否满足观测门槛；`turn_started` 与 `right_contact` 用于区分够不到目标层和碰到后没拧动；`peak_signed_face_deg`、`final_face_deg` 和 `hold_fraction` 分别看单层实测峰值、末值和保持。具体原因看 `end_reason`，控制后半段逐帧测量在同一目录的 `post_measurements.jsonl`。回合持续在同一真实碰撞场景运行，换阶段不重置、没有直接驱动魔方关节。当前只训练正向 15° 小角度；不代表解魔方常用的 90° 转面或实物能力。
