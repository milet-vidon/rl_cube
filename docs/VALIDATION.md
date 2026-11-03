# 2026-11-03 验证

- 56 个离线单元测试通过，包括新增 7 个路径解析测试和 5 个肘姿/同步保持负向测试、4 个完整路径检查回归测试。
- 新的完整 PhysX：左肘外展开发 9501 + 独立 9601–9605 均通过；录像回归 9501 通过。不是 BC/PPO 训练。
- 修正腕部 schema 后完整左手回归 9501 通过；右手接抓 v2 完整回合因预抓 IK 失败，v3 在闭合路径碰撞预检时拒绝执行；两者均未通过。另一个 v3 左手对准回归通过，不计右手接抓。
- `elbow-check` 从冻结源码及轨迹重算 6/6，独立 5/5；`evidence-check` 旧掌心方向证据离线重算 6/6，原证书字节不变。
- 跨目录复制与 Python 文件访问隔离验证共 18 项通过，没有读回旧工作区；含 14 项篡改/缺失/协议拒绝检查。
- `bash -n revotron_cube_rl/run_linux.sh` 与新模块编译通过。
- 当前范围和复现命令见 [实施记录](ISSUES_123_UPDATE.md)。右手交接仍未验收。

以下是历史验证记录。

# 本次同步验证（2026-11-02）

- 发布目录：`github_project`；解释器为原机 conda `rl_cube_6_2`。
- `python -m unittest discover -s revotron_cube_rl -p 'test_*.py'`：40 tests，全部通过。
- `python revotron_cube_rl/test_arm_safety.py`：通过。
- `python revotron_cube_rl/linux_migration_check.py`：通过，资产路径解析到发布目录。
- `bash -n revotron_cube_rl/run_linux.sh`：通过。
- 从发布目录核对 19 个冻结源文件、4 个资产、六个完整回合的协议/源码/配置/资产合同/轨迹哈希；从已存轨迹离线重算原有翻转验收，6/6 通过。读取路径全部由仓库内索引显式解析，不读取原工作区的 run。
- 以上是对已有 PhysX 证据的同步检查，没有新训练、没有重跑 Isaac。相关细节见 [publication_validation.json](evidence/palm_direction_v5/publication_validation.json)。
- 左肘外展与右手完整接抓未通过；本次检查没有扩大原证书验收范围。

以下保留初次发布记录。

# 发布验证（2026-11-02）

- Python：原机 `.conda/rl_cube_6_2/bin/python`。
- `python -m unittest discover -s revotron_cube_rl -p 'test_*.py'`：36 tests，OK，并输出 ARM_SAFETY_TESTS_PASSED。
- `python revotron_cube_rl/linux_migration_check.py`：通过，资产解析指向发布目录。
- `bash -n activate_6_2.sh install_6_2.sh`：通过。
- 发布文件常见 GitHub/AWS token 与 SSH 私钥格式扫描：未发现匹配；这不是完整安全审计。
- 本次未重新启动 Isaac、训练或运行硬件。

打包时原工作区仍有 v17 独立实验与 v18 候选恢复实验进程。已收录的 v17 result 只有 8301/8302 两条结果，8303 不计入分母；候选 v18 不属于已验收基线。本仓库不表示原工作区后续结果会自动同步。
