# Revo3 双臂魔方仿真研究

基于 Isaac Sim 6.2 / Isaac Lab 3.0 的 56 关节双臂双手抓取、整块魔方重定向与换抓实验工程。

**尚未完成机器人单层转动、魔方复原或实机验证。** 本仓库包含传统 teacher、状态 BC、视觉/PPO 探索代码；不同方法的结果不能混用。

## 本次更新

[左掌朝右翻转修正与演示](docs/PALM_DIRECTION_UPDATE.md)：传统 teacher 完成左手抓起、整块反向翻转和保持，新独立 seed 9501–9503 为 3/3。**[左肘应朝外，当前朝内的姿态仍待修复](https://github.com/milet-vidon/rl_cube/issues/1)；[右手完整接抓也未验收](https://github.com/milet-vidon/rl_cube/issues/2)。** 不应将该 3/3 理解为完整交接、自然姿态或 PPO 成功。

## 接手入口

1. [交接文档](docs/HANDOFF.md)：现状、证据、代码地图、限制与下一步。
2. [建议 Issues](docs/NEXT_ISSUES.md)：可直接复制到 GitHub。
3. [Codex 工作约定](AGENTS.md)：验收与安全约束。
4. [历史部署记录](DEPLOYMENT.txt) 与 [环境版本快照](docs/environment/)。

## 环境和启动

基线：Ubuntu 24.04、Python 3.12、Isaac Sim 6.2.0.0、Isaac Lab 3.0.0、Torch 2.11.0+cu130、RSL-RL 5.5.1。原机 RTX 5070 Laptop 8 GB；单环境 CPU PhysX/CPU PPO，GPU 渲染。

```bash
git lfs install
git lfs pull
# 已安装兼容环境时，直接使用解释器：
export ISAACLAB_PYTHON=/path/to/isaac/python
bash revotron_cube_rl/run_linux.sh check
bash revotron_cube_rl/run_linux.sh env-check --runtime --render
bash revotron_cube_rl/run_linux.sh smoke --steps 64 --seed 7 --name fresh_smoke
```

新建隔离环境可执行 `bash install_6_2.sh`（需要 conda、网络和 NVIDIA 软件运行条件），然后 `source activate.sh`、`./run.sh check`。安装脚本保留原版本约束，并非跨平台安装保证。

```bash
# 无 Isaac 启动的离线回归；使用具备 numpy/torch/scipy 等依赖的解释器
python -m unittest discover -s revotron_cube_rl -p 'test_*.py'
python revotron_cube_rl/test_arm_safety.py
```

同一时间只运行一个 Isaac 进程。每个实验使用全新的 `--name`。训练门禁未通过时必须修复前置阶段，不能跳过。

## 仓库内容

- `revotron_cube_rl/`：环境、策略、规划、审计、配置和已选验收证据。
- `revotron_revo3/`：机器人 URDF/USD/网格与转换脚本。
- `isaac_mujoco_cube/`：魔方资产、上游快照与物理探针。
- `docs/`：交接、Issue 草稿、环境与发布清单。

大型资产、轨迹和权重由 Git LFS 保存。原工作区的环境、缓存、视频和大量历史训练数据未全部打包；参见 [制品说明](docs/ARTIFACTS.md)。源码和历史记录内仍有原机绝对路径，证书迁移是首个建议 Issue。

第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。未为自有代码或机器人资产擅自添加开源授权。
