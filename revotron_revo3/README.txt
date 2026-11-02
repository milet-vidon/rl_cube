Revotron 双臂 + 左右 Revo3 模型
准备日期：2026-10-30
官方来源：https://github.com/BrainCoTech/brainco-description

先看外观：revotron_preview.png（正面 + 斜侧面）
图像直接由官方 URDF 网格和正向运动学生成，展示零关节姿态。
这是工程外观预览，不是 Isaac RTX 截图；屏幕材质在通用预览里做了简化。

模型文件
- source/revotron_system/urdf/revotron_bimanual_revo3.urdf
  官方原始模型：59 个活动关节（14 个臂关节 + 42 个手关节 + 3 个腰/头颈关节）。
- source/revotron_system/urdf/revotron_bimanual_revo3_56dof.urdf
  固定 waist_pitch_joint、neck_yaw_joint、head_pitch_joint 于零位，保留双臂双手56关节。
- source/revotron_system/usd/revotron_bimanual_revo3_56dof.usd
  用官方 convert_urdf_v4.py 在现有 Isaac Sim 5.1 / Isaac Lab 环境转换。
- view_scene.usda
  引用上述 USD 并添加外部预览相机与照明。外部预览相机不是胸部观测相机。
- revotron_bimanual_revo3.glb
  通用静态外观模型，可在支持 GLB 的三维查看器中旋转查看；不含机器人关节控制。
- joint_mapping_56.json
  建议动作顺序：左臂7、左手21、右臂7、右手21。按实际导入关节名匹配索引。

在 Isaac 中：File > Open 打开 view_scene.usda，或直接打开 source 下的 USD。
如未自动选中视角，可选择 /World/OverviewCamera；也可以在视口中框选机器人。
保持整个目录结构。不要只复制 USD：屏幕纹理仍使用相对路径。

验证与限制
- 全部387个引用网格与屏幕贴图已下载，原始文件哈希在 source/download_manifest.json。
- 官方转换器验证了2只手、2个掌碰撞配置、64个拇指凸包、164个手部接触碰撞体。
- 198个外观网格颜色与2个带贴图外观通过官方转换检查。
- 额外结构核查见 usd_audit.json。模型动力学运行、关节驱动、抓取和魔方任务尚未测试。
- 官方转换默认固定基座、合并固定关节、关闭刚体重力，驱动刚度/阻尼为0；训练前需要任务配置。
- 胸部相机、魔方和抓持姿态尚未加入；保留官方完整躯干，方便下一步布置相机。
- 上游网格与模型授权信息以 source/README.md 等上游说明为准。

复现：prepare_variant.py 生成56关节版本；官方 source/convert_urdf_v4.py 执行转换；
preview_model.py 和 render_preview.py 生成实际网格预览；verify_usd.py 检查USD与生成查看场景。
