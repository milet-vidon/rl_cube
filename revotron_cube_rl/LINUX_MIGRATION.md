# Linux / Isaac 迁移包

本项目的 Linux 入口是 `run_linux.sh`。它只负责选择 Isaac Python、设置路径和
启动项目脚本，不会自动安装 Isaac Sim 或下载资产。这样可以避免在迁移时把
Windows 盘符、不同 Isaac 版本或错误的 USD 混进训练结果。

## 1. 目标目录布局

Linux 上建议保持与当前工作区相同的相邻目录结构：

```text
<workspace>/
  revotron_cube_rl/                         # 本项目目录
  revotron_revo3/source/revotron_system/
    usd/revotron_bimanual_revo3_56dof.usd
    urdf/revotron_bimanual_revo3_56dof.urdf
  isaac_mujoco_cube/upstream/mujoco_cube-main/
    cube_3x3x3.xml
    assets/sticker.png
```

如果目录不同，在 Linux shell 中设置：

```bash
export REVOTRON_USD_PATH=/data/assets/revotron_bimanual_revo3_56dof.usd
export ISAAC_MUJOCO_CUBE_ROOT=/data/assets/isaac_mujoco_cube
```

## 2. 运行时约定

可以使用以下任一种启动方式：

1. `ISAACLAB_LAUNCHER` 指向官方 Isaac Lab checkout 中的 `isaaclab.sh`，脚本
   会按官方约定调用 `isaaclab.sh -p <script>`；或
2. `ISAACLAB_PYTHON` 指向已经能导入 `isaaclab`、`isaacsim` 和 `pxr` 的 Python
   可执行文件（容器/conda 环境常用）。

不要用系统 Python 代替它运行 Isaac 场景；`python3` 只适合做静态检查或运行不
启动 Isaac 的离线脚本。

```bash
cd /data/work/revotron_cube_rl
# 方案 A：官方 Isaac Lab checkout
export ISAACLAB_LAUNCHER=/data/IsaacLab/isaaclab.sh
# 方案 B：直接使用已配置好的 Isaac Python（与方案 A 二选一）
# export ISAACLAB_PYTHON=/data/isaaclab/_isaac_sim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_USER_HOME="$HOME/.cache/isaaclab/userdata"
export OV_USER_HOME="$OMNI_USER_HOME"
export OMP_NUM_THREADS=4
```

具体 Isaac Sim/Isaac Lab 版本应与资产生成版本保持一致。当前工作区的资产合同
记录在 `asset_contract.json`，其中包含上游 commit 和 `20261025` 资产覆盖版本；
换版本后要重新做 smoke、软限位审计和传统 teacher 验收，不能直接复用旧 checkpoint
作为已验证模型。

## 3. 迁移后的检查顺序

先做不启动 Isaac 的静态检查：

```bash
python3 linux_migration_check.py
```

再检查 Linux Python、资产和依赖：

```bash
./run_linux.sh env-check
./run_linux.sh env-check --runtime
```

`--runtime` 会要求 Isaac 模块和 CUDA；如果只想查看完整 JSON 报告，可加
`--json`。本项目在 Windows 上只能验证前一个静态步骤，不能把静态通过当成 Linux
GPU 启动已经通过。

## 4. 首次 Isaac 启动和训练

先使用短 smoke，不要直接开长 PPO：

```bash
./run_linux.sh smoke --steps 64 --seed 7 --name linux_smoke_v3
```

确认 `runs/linux_smoke_v3` 生成后，再做传统 teacher 验收：

```bash
./run_linux.sh sweep --grasp downward --widths 0.096 \
  --seeds 17,31 --visual --name linux_traditional_visual
```

然后再按项目现有数据和配置做评估、DAgger 或 PPO：

```bash
./run_linux.sh evaluate --steps 600 --seed 131 \
  --name linux_bc_eval_131 \
  --checkpoint runs/visual_warmstart/checkpoint.pt --record

./run_linux.sh dagger --help
./run_linux.sh train --mode train --iterations 2 --steps 600 \
  --seed 137 --name linux_ppo_probe \
  --checkpoint runs/visual_warmstart/checkpoint.pt \
  --learning-rate 5e-5 --noise-std 0.05
```

这里的成功门槛仍然是双侧真实接触、离桌至少 8 cm、低速保持 30 个控制步、
无魔方-桌面接触；混合 teacher+policy 的成功不等同于纯策略成功。

## 5. 文件和权重迁移

只复制项目代码、配置、成功示范和明确选中的 checkpoint。`runs/` 很大时可按
实验目录选择性复制；每个 run 的 `result.json`、`config.json`、轨迹和 checkpoint
必须一起保留。迁移后先核对 `config_sha256` 与数据集哈希，再做独立 seed 评估。

```bash
rsync -a --info=progress2 revotron_cube_rl/ user@linux:/data/work/revotron_cube_rl/
rsync -a --info=progress2 revotron_revo3/ user@linux:/data/work/revotron_revo3/
rsync -a --info=progress2 isaac_mujoco_cube/ user@linux:/data/work/isaac_mujoco_cube/
```

不要把 Windows 下的 `E:\IsaacLab\...` 环境变量或临时缓存路径复制到 Linux。
`run_linux.sh` 和 `iterative_training.py` 会为 Linux 设置用户缓存目录；资产路径
仍建议显式设置并写入启动日志。

如果希望先在 Windows 侧生成一个可核对的压缩包，可以只打包项目和指定实验：

```powershell
python prepare_linux_bundle.py --output ..\revotron_cube_rl_linux.tar.gz `
  --run traditional_visual_verified --run state_phase_knn_user_softlimit_seed1945
```

需要把相邻资产也放进包时再加 `--include-assets`；脚本会同时生成
`.tar.gz.manifest.json`，记录每个文件的 SHA256。默认不会复制整个历史 `runs/`。

本次已生成两个可直接复制到 Linux 的示例包：`revotron_cube_rl_project.tar.gz`
（代码与配置）以及 `revotron_cube_rl_teacher_bundle.tar.gz`（另外包含
`state_phase_knn_user_softlimit_seed1945` 的教师回放和轨迹）。它们位于项目的
上一级目录；对应的 `.manifest.json` 是完整性清单。

## 6. 当前边界

- 本项目已经有 Windows 静态迁移检查和 Linux 启动入口。
- 当前工作区没有附带的 Linux GPU，因此 Linux Isaac Sim 实际启动、驱动兼容性和
  大规模并行环境尚未在本机验收。
- 双臂双手 56 关节、胸部相机、软限位和传统抓取 teacher 的接口保持不变；迁移后
  应先复现 smoke/teacher，再继续视觉策略或 PPO。
- 斜向抓取仍需重新选择接触点和 IK 目标；不能只旋转当前平行目标后直接训练。
