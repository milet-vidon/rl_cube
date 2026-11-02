# 发布验证（2026-11-02）

- Python：原机 `.conda/rl_cube_6_2/bin/python`。
- `python -m unittest discover -s revotron_cube_rl -p 'test_*.py'`：36 tests，OK，并输出 ARM_SAFETY_TESTS_PASSED。
- `python revotron_cube_rl/linux_migration_check.py`：通过，资产解析指向发布目录。
- `bash -n activate_6_2.sh install_6_2.sh`：通过。
- 发布文件常见 GitHub/AWS token 与 SSH 私钥格式扫描：未发现匹配；这不是完整安全审计。
- 本次未重新启动 Isaac、训练或运行硬件。

打包时原工作区仍有 v17 独立实验与 v18 候选恢复实验进程。已收录的 v17 result 只有 8301/8302 两条结果，8303 不计入分母；候选 v18 不属于已验收基线。本仓库不表示原工作区后续结果会自动同步。
