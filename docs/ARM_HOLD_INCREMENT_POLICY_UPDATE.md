# V9：接触保持与手臂增量训练

V8实际闭环改善了无名指保持，但两回合均未通过腕部验收。首回合在食指刚恢复接触后的560步错误选择BASE，撤销约5.069mrad修正，561步失触。另有398个原专家投影目标超出V8的动作表示；第二回合有523个。全部区间缺口来自基线锚定的手臂，11个恢复手指的范围足够。原失败与原安全门保持不变。

V9保留冻结V4、284维实际状态、GRU、BASE/HOLD/UPDATE分类及公共阶段门。只把7个手臂UPDATE改为相对上一实际目标的增量，其范围0.025rad来自原投影器min(step_limit, velocity_limit×control_dt)。每份训练数据、模型loader及实际构建的评估环境均重新核验这个原范围。原soft limits、动作转换、速度投影、碰撞和阶段验收不变；11个恢复手指仍为上一目标±0.006rad，另外10个仍为基准±1e-6rad。

## 数据与覆盖

九段开发历史共9523 eligible帧、5032 active帧，物理种子仍是已见9501。新增的首V8失败史完整1842帧通过原逐转移安全审计，并逐bit重放实际284维输入及全部GRU动作。原tilt_hold_timeout、腕门False和训练回合不合格标记全部保留。第二V8在1725步之后的teacher查询抛异常，缺原unfinished query audit，因此暂不准入；不把它变成成功查询或补造标签。Native articulation数据未混入。

逐关节分支覆盖：区间缺口0、原float32动作转换与投影后的精确见证缺失0、公共门掩盖的真实修正0。浮点labels−previous可能略大于数学cap，因此候选见证同时检查cap-clipped delta；原标签不改。真实J4第1802步目标-2.2321274280548096rad已用有限网络输出逐bit重现。初次覆盖执行检测到审查期间源变化并拒绝，01错误日志保留；冻结后的02完整通过。

## 训练与验收

hidden64和hidden128各5000epochs，初始化seed都为20261104，用相同九史、AdamW、学习率3e-4和原回归/分类目标作容量对照。资格文件SHA为6283a3d6e1a1feccea809c5150eac49aaa5784299cfb48a8d8b396be07f07cf7，原training_authorized=False保留，另行root授权绑定各自配置。两容量的CPU/CUDA BASE/HOLD提案、投影和分类定义逐bit一致；预检不建优化器。

候选按7史第499步精确BASE、2史第545步无名指精确投影HOLD、首V8史第560步食指精确投影HOLD、全部类别错误、硬解码投影RMSE依序选择。560门先核原专家目标确为HOLD且不同于BASE，不重标标签。投影动作相等不意味着模型一定选择HOLD类，预测类仍单独记录。完整CPU顺序复核、冻结检查点预检和原门槛实际回合仍必须通过。

6个生产模块和11个测试模块按冻结字节安装，92项仓库测试通过、无skip。评估完整快照22项训练依赖。原每步单次actor历史推进、teacher查询、右手/阶段/增益事件、终止和清理顺序保留；新增初始化actual cap检查在actor建立前失败闭锁。

冻结候选证据保留在docs/evidence/arm_hold_increment_v9_candidate_20261104。首批hidden64和hidden128现均完成5000epochs、45000次更新；新增完整产物分别在docs/evidence/arm_hold_increment_v9_hidden64_seed20261104_training_20261104及对应hidden128目录，每份127文件已由不同于trainer作者的审查者独立核验。

| 容量 | 选中epoch | 九史分类错误 | 投影RMSE(rad) | 最大动作误差(rad) |
| --- | ---: | ---: | ---: | ---: |
| 64 | 4800 | 79 | 0.0001001232527 | 0.0049999952 |
| 128 | 4900 | 14 | 0.0000686465512 | 0.0025962591 |

两模型原499/545/560回归违反均0，完整CPU动作重放、原资格/授权/22依赖、真实checkpoint preflight全部通过。训练日志每轮shuffle、候选选择与Adam/RNG记录已检查。hidden64模型SHA为98460e40cfb66007612d597db8065502f18d48ec9b06c6d9039a9c6ecb3944fe；hidden128为9809d22ed3446cbc9887bf9db9d6a50d3ae708e02da2b0d0277d98617c90046d。

两模型按单Isaac次序等待闭环实测，不能把拟合指标当物理成功。追加hidden128/seed20261105和hidden256/seed20261104仍用原九史做种子/容量对照。尚未新增V9物理成功、独立泛化、接触转层或视觉求解结论。last_state保存Adam/RNG，但没有精确resume CLI或恢复训练证明。
