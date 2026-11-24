# Revo3 双臂魔方仿真

用 56 关节的双臂双手机器人，在 Isaac Sim / Isaac Lab 中研究抓取、稳持、换抓和转动魔方。工程包含脚本控制器（teacher）、状态行为克隆（BC）和视觉 PPO 实验。整块魔方改变朝向和机器人拧动其中一层是两个独立的验收项目。

冻结的腕部姿态控制器在 5 个独立种子上通过了连续 3 秒的持握与姿态验收；这是脚本控制器的结果。尚未验证可靠的接触驱动单层转动、完整复原或实机运行。逐种子指标见[腕部验收](docs/WRIST_DIRECTION_UPDATE.md)，单层实验见[自由核心接触实验](docs/SINGLE_FACE_CONTACT_UPDATE.md)。

## 先试一个离线例子

在仓库根目录运行。这个测试只用 NumPy 和 SciPy：给整颗魔方施加旋转，层角应为零；只相对核心旋转一层，测得的角度应保留方向。它不需要 GPU、机器人资产、预训练权重或历史轨迹。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy
python -m unittest discover -s revotron_cube_rl -p 'test_cube_state.py'
```

## 在 Isaac 中运行

完整仿真需要兼容的 NVIDIA GPU 和驱动、Isaac 环境，以及合法取得的机器人 USD 和魔方资产。原项目使用 Ubuntu 24.04、Python 3.12、Isaac Sim 6.2、Isaac Lab 3.0。环境版本见[部署记录](DEPLOYMENT.txt)。以下命令用于具备相应资源的研究检出目录，轻量源码包不含模型资产或实验记录。

```bash
# 在研究检出目录按需取得资产，不必下载所有历史轨迹
git lfs install --local
git lfs pull --include='revotron_revo3/**,isaac_mujoco_cube/**'

# 已装好 conda 的 Linux 主机；安装需要网络和额外磁盘空间
bash install_6_2.sh
source activate.sh
bash revotron_cube_rl/run_linux.sh check
bash revotron_cube_rl/run_linux.sh env-check --runtime --render
bash revotron_cube_rl/run_linux.sh smoke --steps 64 --seed 7 --name "smoke_$(date +%Y%m%d_%H%M%S)"
```

如果已经有兼容环境，直接把 `ISAACLAB_PYTHON` 指向其中的 Python，再运行上面的三个检查和仿真命令。外置资产通过 `REVOTRON_USD_PATH`（USD 文件）和 `ISAAC_MUJOCO_CUBE_ROOT`（含上游 XML、纹理的目录）指定。同一台机器一次只启动一个 Isaac 进程；每个新实验使用唯一名称。

## 源码、资产与实验记录

| 路径 | 用途 |
| --- | --- |
| `revotron_cube_rl/` | 环境、训练和评估入口、配置、阶段验收与测试 |
| `revotron_revo3/`、`isaac_mujoco_cube/` | 研究检出目录中的机器人与魔方资产、来源记录和转换工具 |
| `revotron_cube_rl/runs/`、`docs/evidence/` | 本地运行输出与历史验收证据；部分旧记录已由 Git 或 LFS 跟踪 |
| `docs/` | [制品说明](docs/ARTIFACTS.md)、[研究交接](docs/HANDOFF.md)、[待解决问题](docs/NEXT_ISSUES.md) |

从维护者的工作区生成轻量源码包，用于审查或分发准备：

```bash
python3 tools/export_source.py work/releases/rl-cube-source.tar.gz
```

这个包不携带 Git 历史、Conda 环境或实验输出。只读现有研究仓库时也可在克隆命令前设置 `GIT_LFS_SKIP_SMUDGE=1`，避免初次检出下载 LFS 内容；普通 Git 对象仍会下载。需要复核某项历史结果时，再按[制品说明](docs/ARTIFACTS.md)取对应资产、权重和证据。

## 贡献与许可

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。改动控制器或策略前请阅读[项目约定](AGENTS.md)。提交实验结果时请注明方法、种子、配置、实际终止原因与验证范围。离线测试用于检查算法；物理行为需要 PhysX 回合验证，训练损失不能代替验收。

项目自有代码采用 [Apache-2.0](LICENSE)。该许可不授予机器人资产、模型权重或实验数据的再分发权；机器人资产的公开再分发授权尚未确认。第三方组件的来源和已知授权见 [THIRD_PARTY.md](THIRD_PARTY.md)。
