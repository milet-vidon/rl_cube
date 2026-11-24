# 90 毫米魔方：采样、训练和闭环验证

在本仓库根目录运行。首次启动会新建目录；已有目录请用下方的续训命令。

```bash
bash run_contact_hold90_long.sh \
  --run-dir runs/contact_hold90_long_v2 \
  --num-envs 1536 --val-envs 32 \
  --rounds 100 --updates-per-round 2000 --batch-size 32768
```

复位时双腕比旧版各向后退 20 毫米，朝向不变，抓取和抬升目标不变。物理复位要求魔方各轴位置偏差不超过 15 毫米、没有双手或手腕对魔方的冲量。旧的 v1 失败记录保留在原目录。

起点是原第 15 轮的权重，第一轮使用新的优化器。每轮用 90 毫米物理魔方和已标定的张手宽度、抓取位置重新采样，用最近三轮的数据训练，再把候选模型放回仿真，在同一组独立的 32 个种子上从复位开始验证。左手由学生控制，阶段调度与右手保留脚本。每轮请求 1536 个采样环境；显存或主机内存剩余不足时，采样器尝试减小同批环境数，分批覆盖该轮。训练在第 100 轮结束；没有按 loss 提前停止。训练更新数不是拿稳次数。

```bash
# 另开终端查看所属阶段、日志路径和最后完成的轮数
bash run_contact_hold90_long.sh --run-dir runs/contact_hold90_long_v2 --status
# 仅在原进程退出后，从该目录的原设置和已保存进度继续
bash run_contact_hold90_long.sh --run-dir runs/contact_hold90_long_v2 --resume
```

每轮的 `round_XXX/validation_summary.json` 记录：`held` 是曾拿稳的环境数；`still_held_at_end` 是最后仍拿稳的环境数；`whole_turn` 是实现稳持并转动整颗魔方的环境数。拿稳使用物理量判定：连续 1 秒离桌至少 2 厘米、有左手接触、无桌面支撑、无危险终止，且魔方中心相对手掌的移动不超过 1 厘米。不要求四指。整颗转动要求稳持期间方向改变至少 15°，达到后继续稳持至少 0.5 秒。本训练没有单层拧动的动作示范或训练目标；不能用整颗转动推断会拧一层。

`state.json` 的 `best` 指向按闭环成绩选出的最佳轮，`current` 是最近完成轮次；轮内候选 checkpoint 按独立种子的拟合损失选择。若拟合损失下降，但多轮闭环仍然全部为零，先看 `round_XXX/fit/result.json` 的 `phase_rows` 以及采样报告里的 `end_reason`：若缺少抬升、转动阶段的示范，就应先修正 teacher 或采样策略。
