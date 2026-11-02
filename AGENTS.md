# 接手约定

先阅读 docs/HANDOFF.md、docs/NEXT_ISSUES.md，再核对 Issue 涉及的源码与 runs 证据。旧 PROJECT.md / FRAMEWORK_CHAIN.md / STATUS.txt 是累积历史，不能单独当作当前结论。

- 56 维动作顺序、用户软限位、安全交集、速度与接触门槛必须保留。禁止为提高成功率关闭碰撞、绑定魔方或重置魔方位姿。
- 区分 teacher、状态策略、视觉策略、带监督 BC 和纯 PPO；整块倾斜不是单层转动。
- 不绕过 stage_gate.py、opposition_gate.py 等门禁；证书缺失、哈希不符必须显式失败。
- 实验使用唯一 run 名称，单机同时只开一个 Isaac 进程。不得覆盖历史结果或 checkpoint。
- 冻结模型/配置后再测试；已经看过的 seed 不能再次宣称为新的独立测试。
- 修改后执行相关离线测试；涉及物理行为时另需真实 PhysX 回合。无 GPU 的环境只能报告静态/离线验证。
- 每个 PR 说明问题、改动、运行命令、结果、失败案例、未验证范围；证据保存 seed、配置/源码/资产哈希与指标。
- 新 runs 默认忽略，选择必要证据加入 Git；大文件使用 LFS，不提交环境、凭据和缓存。
