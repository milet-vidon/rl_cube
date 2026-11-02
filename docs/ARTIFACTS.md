# 打包范围

本仓库是原工作区的发布快照，原目录未移动。包括全部主项目源码/配置、机器人与魔方资产、历史 runs 中小于 2 MB 的 JSON、所有 gate JSON 直接引用的存在文件、state_bc_v4 与 side_bc_waypoints_v2 的 checkpoint，以及 v17 shift/independent 的 JSON 轨迹。

不包括 .conda、vendor、work 缓存、日志、绝大多数视频、旧 checkpoint、训练集和大型历史轨迹。原有结果里的路径不保证全部随仓库提供；缺失时应报告不可复现，不能生成替代证据。

`release_manifest.json` 记录本次选入源码/资产/结果的大小和 SHA256（不含 Git 元数据与文档）。LFS 下载后可按其验证文件完整性。LFS 提交的轨迹文件仍保持原始内容和哈希。
