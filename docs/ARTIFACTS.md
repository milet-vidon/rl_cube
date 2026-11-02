# 打包范围

本次新增掌心方向修正 v5 的源码、协议和证书，六个完整物理回合（9301、9402、9403、9501、9502、9503），两个右手接抓失败回合（`palm_facing_dev_v1`、`palm_facing_resume_v4_spacing`），以及 seed 9501 的视频和最终姿态图。详见 [证据索引](evidence/palm_direction_v5/index.json) 与 [更新说明](PALM_DIRECTION_UPDATE.md)。

大型轨迹和演示视频使用 Git LFS。新 runs 仍默认忽略，本次逐项选择上述证据加入版本管理，没有上传 conda、缓存、凭据或全部工作目录。

原证书、run 源码快照和轨迹保持字节及哈希不变。原机绝对路径不保证可以直接在其他克隆中运行；证据索引只提供明确映射，运行时 gate 的路径迁移仍是待办。未收录的历史结果不能推定可复现，也不能生成替代数据。

`release_manifest.json` 记录已选源码、资产、结果与演示的大小和 SHA256，不含 Git 元数据和一般文档。LFS 下载后可验证文件完整性。

以下是初次发布范围，作为历史保留。


本仓库是原工作区的发布快照，原目录未移动。包括全部主项目源码/配置、机器人与魔方资产、历史 runs 中小于 2 MB 的 JSON、所有 gate JSON 直接引用的存在文件、state_bc_v4 与 side_bc_waypoints_v2 的 checkpoint，以及 v17 shift/independent 的 JSON 轨迹。

不包括 .conda、vendor、work 缓存、日志、绝大多数视频、旧 checkpoint、训练集和大型历史轨迹。原有结果里的路径不保证全部随仓库提供；缺失时应报告不可复现，不能生成替代证据。

`release_manifest.json` 记录本次选入源码/资产/结果的大小和 SHA256（不含 Git 元数据与文档）。LFS 下载后可按其验证文件完整性。LFS 提交的轨迹文件仍保持原始内容和哈希。
