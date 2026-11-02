# 可直接创建的 Issues

## 1. P0：证书和实验路径跨工作区可移植

背景：gate JSON 保留原机绝对路径，克隆后可能失败或误读原工作区。
范围：stage_gate.py、opposition_gate.py、side_gate.py、tilt_gate.py 及证书解析。
验收：采用显式根目录或仓库相对路径；临时目录中的克隆可验证已打包证据；任意篡改/缺失证据仍失败；不得绕过校验或伪造历史证书。新增迁移映射与原始哈希溯源，覆盖跨路径及负向测试。

## 2. P1：定位 v17 倾斜换抓 seed 8301 失败

背景：独立记录中 8301 失败、8302 完整通过，不能宣称泛化稳定。
范围：tilt_regrasp.py、tilt_experiment.py、tilt_audit.py。
验收：先检查原始 result/trajectory 复现失败阶段；保留碰撞、限位、全回合/恢复回合区分；冻结方案后登记至少 5 个新独立 seed，报告每回合与总成功率、接触覆盖、倾斜误差、碰桌/自碰撞冲量。原 8101/8201/8301/8302 仅用于回归。成功目标为完整 reset→抓取→倾斜→右手换抓并保持 3 秒，未达到应明确报告失败。

## 3. P1：统一可核查的项目状态索引

背景：PROJECT/FRAMEWORK_CHAIN/pipeline_status 与最新对掌、换抓结果不一致。
验收：建立指向实际 result 与哈希的统一索引，区分 teacher/BC/PPO、锁层/动态层、完整/resume、development/independent；撤销旧转层结论；不删除失败记录、不扩大验收范围。

## 4. P2：建立真实动态魔方单面任务

前置：抓取/换抓与安全门稳定。
范围：dynamic_cube.py、robot_hand_turn_stage.py 与独立验收。
验收：自由核心、真实机器人接触，无外部推板代替手、无位姿写回/绑定；在魔方局部系计算指定层相对转角与其他层误差，保存接触和状态证据；先完成 state-only teacher，再讨论训练。通过前不解锁视觉阶段。

## 给 Codex 的通用 Issue 开头

请先读 AGENTS.md 和 docs/HANDOFF.md，只处理本 Issue 范围。核对现有证据后修改，先离线回归再按需 PhysX 验证。提交 PR 时列出复现命令、结果、失败案例与未验证范围；禁止降低安全门槛以满足验收。没有 GPU 时明确说明只完成离线部分。
