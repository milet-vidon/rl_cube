# Revo3 双臂魔方仿真研究 · 源码包

这个包提供 Isaac Sim / Isaac Lab 环境、56 关节安全限制、抓取与转层测量、teacher、状态策略、视觉 PPO，以及接触保持实验的研究源码。单层转动、整块魔方复原和实机泛化尚未得到验证。没有附带可直接部署的模型。

## 先运行离线测试

用 Python 3.12 和独立虚拟环境即可验证转角计算，不需要 GPU 或机器人资产：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy
python revotron_cube_rl/test_cube_state.py
```

测试验证整块旋转不会被计为单层转动，并覆盖正负 90° 和错轴动作。已安装兼容 Torch 时，另可运行 `python revotron_cube_rl/test_arm_safety.py` 检查关节限位。

## 使用模拟器

在 Linux/NVIDIA 机器上准备兼容的 Isaac Sim 6.2 / Isaac Lab 3.0 环境；附带的 `install_6_2.sh` 使用 Conda 和网络安装固定版本。也可以指定已有 Isaac Python 解释器。运行仿真前须通过授权渠道自行获取完整 Revo3 机器人资产（含其引用的网格）和 mujoco_cube 源资产；来源见 [第三方说明](THIRD_PARTY.md)。机器人 URDF 路径对部分规划工具是固定的：`revotron_revo3/source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf`。

```bash
export ISAACLAB_PYTHON=/path/to/isaac/python
export REVOTRON_USD_PATH=/path/to/revotron_bimanual_revo3_56dof.usd
# 此目录下必须有 upstream/mujoco_cube-main/{cube_3x3x3.xml,assets/sticker.png}
export ISAAC_MUJOCO_CUBE_ROOT=/path/to/isaac_mujoco_cube
bash revotron_cube_rl/run_linux.sh check
bash revotron_cube_rl/run_linux.sh env-check --runtime --render
bash revotron_cube_rl/run_linux.sh smoke --steps 64 --seed 7 --name fresh_smoke
```

一次只运行一个 Isaac 进程。为每次运行选择未使用的名称。先用 smoke 验证环境；训练、重放和评估可能额外需要对应版本的场景、父检查点与验收证据，缺失时不应放宽门禁。`contact_hold90_long.py` 等实验保留原始代码，依赖特定历史 run 和模型，不能仅靠本包复现实验结论。

## 包含什么

`revotron_cube_rl/` 包括项目源码与选定配置、测试；`tools/source-files.txt` 枚举每个输入文件。本包没有 Git 历史、机器人/魔方资产、第三方源码、环境、缓存、证书、轨迹、模型和研究机专用启动脚本。它不能代替完整实验归档。部分遗留工具包含原工作区路径标识，用于校验、迁移历史证据。

要从相同源码重新制作候选包，运行 `python3 tools/export_source.py work/releases/rl-cube-source.tar.gz`，输出路径必须不存在。打包不会修改 Git 索引，也不联网。

本包中项目自有代码按 [Apache License 2.0](LICENSE) 提供。外部依赖和未附带的机器人、魔方资产不在此授权范围内；见 [第三方说明](THIRD_PARTY.md)。
