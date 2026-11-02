import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cpu=json.loads((ROOT/'validated_cpu_result.json').read_text())
negative=json.loads((ROOT/'no_contact_result.json').read_text())
assert cpu['passed']
assert all(abs(r['mean_angle_deg'])<1 and r['fixture_contact_headers']==0 for r in negative['results'])
model=(ROOT/'cube_model.usda').read_text()
assert model.count('def PhysicsRevoluteJoint')==6
assert model.count('def PhysicsSphericalJoint')==20
assert 'DriveAPI' not in model and '/World/Fixture' not in model
assert 'physxScene:enableGPUDynamics = 0' in model
summary={'cpu_passed':True,'negative_control_passed':True,'model_joints':{'revolute':6,'spherical':20},'cube_drives':0,'runs':{}}
for file in sorted(ROOT.glob('*result.json')):
    d=json.loads(file.read_text())
    summary['runs'][file.name]={'passed':d['passed'],'gpu':d['gpu'],'physics_steps':d['physics_steps'],'phases':[{k:r[k] for k in ['axis','mean_angle_deg','max_position_error_m','max_orientation_error_deg','fixture_contact_headers','max_released_position_error_m','passed'] if k in r} for r in d['results']]}
    if 'termination' in d: summary['runs'][file.name]['termination']=d['termination']
summary['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'contact_test.py',ROOT/'cube_model.usda',ROOT/'cube_contact_scene.usda']}
(ROOT/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
text='''mujoco_cube → Isaac Sim 5.1 物理移植与接触转层验证
日期：2026-10-30（Asia/Shanghai）

已完成
- 下载并保留完整 upstream/mujoco_cube-main 源码及 Apache-2.0 许可证。
- 将原始分块网格、UV、贴图、6 个旋转关节与 20 个球关节移植到原生 USD/PhysX。
- 使用现有 E:\\IsaacLab\\env\\python.exe，无升级、无安装新依赖。
- CPU PhysX 下，连续 +Z 90°、+X 90° 外部接触台架测试通过。
- 两个动作在同一次物理模拟中执行，中间没有重置、改写魔方状态、改关节拓扑。
- 无接触对照通过：推板保持同样运动但禁用推板碰撞，魔方不转层。
- GPU 普通配置未通过；更小步长/更多迭代也没有通过连续两面测试。最后时长匹配的复测第一面失败后已停止，第二面未执行。

运行方式（PowerShell）
  & 'E:\\Codex\\2026-10-30\\w\\outputs\\isaac_mujoco_cube\\run_test.ps1' -Mode cpu
  & 'E:\\Codex\\2026-10-30\\w\\outputs\\isaac_mujoco_cube\\run_test.ps1' -Mode no_contact
  & 'E:\\Codex\\2026-10-30\\w\\outputs\\isaac_mujoco_cube\\run_test.ps1' -Mode gpu
  & 'E:\\Codex\\2026-10-30\\w\\outputs\\isaac_mujoco_cube\\run_test.ps1' -Mode gpu_refined
入口会依据 JSON 验收结果返回非零退出码，负对照则验收“没有转层”。

交付文件
- cube_model.usda：独立模型，默认 CPU 物理，核心固定，无魔方驱动器。
- cube_contact_scene.usda：额外包含四块外部推板的初始测试场景。
- contact_test.py：模型转换、动力学运行和测量脚本。打开 USD 只加载场景，不会自动执行推板运动。
- validated_cpu_result.json / no_contact_result.json：完整数值和块体轨迹。
- validation_summary.json：各次试验摘要及最终源文件 SHA256。
- contact_vs_no_contact.gif：用实际记录的位置/姿态绘制的示意回放；不是 RTX 相机图像，省略推板显示。
- SOURCE.txt：上游出处、下载哈希和移植差异。
- diagnostics/：早期调试数据，不作为最终通过依据；logs/ 保留原始验收日志。

验收阈值（各阶段分别检验）
- 选中该面的 9 块；8 个非中心块平均转角距 90° 小于 5°。
- 全部 26 块的位置相对目标最大误差小于 2 mm。
- 全部 26 块（含面中心）的姿态最大误差小于 5°。
- 推板撤离后最大位置误差仍小于 2 mm；存在实际推板接触报告。
- 负对照各面转角小于 1°，推板接触数为 0。

关键边界
- 固定核心台架，不是自由魔方，也没有接入 Tianji/Revo3 双手。
- 四块运动学推板形成几何夹持，靠法向接触推动；材料摩擦为 0。
- 保留原始碰撞网格的凸包，重力 9.81 m/s²；不宣称精确复现真实魔方摩擦或卡位。
- 使用普通刚体和关节，不是可直接交给 Isaac Lab ArticulationCfg 的训练环境。
- 尚未验证悬空夹持、多轮随机打乱、胸部相机渲染、训练吞吐或策略学习。
- 默认完整 SimulationApp 曾在 RTX scene database 初始化时崩溃；本测试用已有 Isaac Lab headless experience 完成物理步进。
- GPU 与 CPU 求解行为不同；必须查看下方实测结果，不可把 CPU 通过视为 GPU 并行训练已就绪。

实测摘要
'''
for name,run in summary['runs'].items():
    text+=f"\n{name}: GPU={run['gpu']}, requested_turns_passed={run['passed']}, physics_steps={run['physics_steps']}\n"
    for r in run['phases']:
        text+=f"  +{r['axis']}: mean={r['mean_angle_deg']:.6f} deg, max_position_error={r['max_position_error_m']*1000:.6f} mm, passed={r['passed']}\n"
text+='\n注意：no_contact 的 requested_turns_passed=False 是预期现象；其负对照验收已通过。\n'
(ROOT/'README.txt').write_text(text,encoding='utf-8')
print(json.dumps({'CPU':cpu['passed'],'negative_control':True,'runs':list(summary['runs'])}))
