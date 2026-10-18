"""Write a design-only V3 plan; never alter V2 frozen code or experiment data."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1];W=R.parent
core_name='BCI_VLA框架_神经交互模块方案_V3.txt'
manual_name='BCI_VLA通用交互框架_逐步执行手册_V3.txt'
assert not (W/core_name).exists() and not (W/manual_name).exists(),'Preserve existing V3 plans'
def txt(p,s):p.write_bytes(b'\xef\xbb\xbf'+s.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
core='''脑机—VLA框架V3：显式神经交互模块设计书
日期：2026-10-16
状态：设计草案；尚未实现新网络、训练或验证新模型。当前授权只覆盖计划改写与只读核对。
主目录：D:\\Scene_Aware；现有代码：D:\\Scene_Aware\\bci_vla_framework。

一、当前事实与改线原因
V2步骤14C/15C完成了非真实完整闭环和软件输入×实际VLA工程适配；并未修改π0.5内部架构。
V2步骤16完成6272个合成episode、409项软件测试和30项完整性检查。M与同probe B9的动作及结局相同，M未选择probe；M2错误启动观察比例比B9高约3.8个百分点。
G2效用未通过，G3未验证，正式G4未通过。上述失败、数据和冻结源码必须保留，不能改名成为V3成果。
V3转向一个待检验问题：可训练模块是否能将场景候选物体与带来源的脑机交互证据对齐，帮助π0.5提出用户愿意确认的主任务，并在歧义时提出恰当澄清？
代码简单或网络复杂都不决定创新；新网络要有可复核的作用、额外信息价值及相同预算下的对照。

二、完整系统及研究范围
真实视觉前端（待联调）：摄像头→现有Florence/SAM候选物体、标签、位置/掩码。
早期受控开发：先使用LIBERO状态或固定标注得到候选对象，必须标记特权来源；不计为视觉识别能力。
脑机输入：物体选项对应的公开选择事件及其证据，保留不确定性、无响应和版本。
新模块：物体编码支路＋交互历史支路＋查询融合网络，输出内部证据token。
π0.5高层：当前画面/状态、公开目标上下文、证据token→主任务候选或澄清建议。
用户：选择或确认具体主任务，确认内容包括动作、目标物体及必要的目的地。
后续接续：同一高层VLM根据已确认主任务和新画面预测当前语义subtask；动作专家生成连续动作；独立状态判定结果。
当前首要研究对象是“主任务建议”，subtask先复用原论文思路作执行接续，不同时宣称新子任务规划理论。
物体选择不唯一决定用户意图。选中杯子不能直接推出倒水、清洗或收纳；场景只产生候选建议，用户确认决定采用哪一项。
首版证据token只进入高层语言推理。低层仍接收图像/状态和已确认subtask，不同时改造动作专家。

三、新模块的结构：场景与交互证据查询网络（暂定名称）
这是独立nn.Module和独立checkpoint中的一组可训练参数；不是一段拼接提示词的函数。
采用已有query/cross-attention工具，设计动机是对象数可变、选择存在歧义、历史有时间与来源；不以“查询Transformer”本身作为原创点。

3.1 物体支路
保留所有有效候选物体，而非先argmax后只保留一个。
每个物体包括：scene-scoped instance_id、冻结视觉编码器的ROI/区域特征、名称语义特征、归一化bbox、可用性mask。
同时绑定该对象对应的公开选择证据；有校准概率时附概率及校准版本，未校准raw score必须带score_kind且不能冒充概率。
缺失视觉特征、未知可信度、遮挡、未知对象都用显式mask/unknown表示，不用0代表“可信度为0”。
线性投影和类型编码将输入映射到d维object tokens。空间使用bbox位置，不把物体列表行号作为语义编号。
换物体排序时，证据、bbox、ID映射一起重排；网络输出的对象指针应等变，主任务语义应保持不变。

3.2 交互历史支路
输入最近H条公开事件：TASK选择、已知目标PROBE、CONFIRM、REJECT、CANCEL，以及提示内容、实际回答、有效性、相对时间和版本。
TASK回答描述用户任务选择；PROBE回答描述已知提示下的通道证据；CONFIRM描述对明确任务的接受/拒绝。使用不同类型embedding，不能混成同一种标签。
历史只含已发生信息。私有真实目标、注入时间表、未来确认和测试标注禁止作为特征。
事件编码后用1层时间Transformer处理顺序；当前证据也作为本支路末尾事件，避免把“历史有意义”仅建立在任意字符串拼接上。
没有已知目标probe且存在不可辨识混淆时，网络必须允许unknown/澄清，不能靠训练猜出隐藏真值。

3.3 查询融合核心
K个可学习query经两个融合块。
每块依次执行：query self-attention→query对物体tokens的cross-attention→query对历史tokens的cross-attention→FFN，各有归一化与残差。
两路分别mask，避免padding或无效事件参与注意力。空场景、空历史使用显式unknown token，禁止全mask softmax产生NaN。
查询token不预设具有“目标/故障”等可解释语义；是否形成相关表示，需要辅助监督和可复核实验，而非看注意力热图下结论。
该结构将可变数量物体/历史压缩为固定K个向量。

3.4 三个输出
输出A：K个融合向量，投影到实际VLM embedding维，作为soft evidence prefix tokens。
输出B：对当前候选物体及unknown的指针分布，用于验证视觉实例与选择证据的绑定；不直接决定执行许可。
输出C：需要澄清的预测。它只能建议继续交互，不能替代用户确认、授权或STOP。
两个辅助头是可检验的监督接口，不以多加两个头作为独立创新点。

3.5 待验证工程初值
d=256，注意力头数4，融合块数2，FFN宽度1024，K=8，历史H=8，首版每帧最多16个候选。
这是原型初值，不是通过实验得到的最优设置。实际参数量按所接视觉/语义维度和VLM维度计数，预计百万量级，不能提前宣传精确数值或加速倍数。
超过16个对象须明示截断规则和候选召回损失，不能偷偷丢掉被选物体或用隐藏目标选择保留对象。
通用性测试必须扩大对象数量、改变位置/顺序，并保留未见物体/场景；二选一原型不等于通用框架。

四、接入π0.5的实际位置
原论文统一VLM高层输出subtask文本，低层动作专家使用独立权重；同一模型不等于所有分支共用全部参数。
当前服务器pi05_libero链路调用sample_actions，尚未接通或验证原论文的高层文本生成能力。
因此首个模型门槛是核实当前checkpoint的文本输出头、tokenizer、训练来源和高层生成入口，不能凭π0.5名字承诺开箱即用。
高层推理中，在视觉/任务prompt前缀的指定边界插入K个inputs_embeds，正确更新padding、attention、position_ids和生成cache。
soft token不是自然语言token ID，不能直接塞进tokenizer字符串。
关闭模块必须走原始输入序列与原始位置/缓存路径；把K个全零token留在序列中不保证恢复原模型，因为注意力归一化和位置已变化。
插入、移除、替换物体或交互证据时清理相关高层KV cache，防止旧意图在缓存中残留。
第一阶段冻结视觉编码器、VLM及动作专家，只训练新模块。requires_grad=False冻结权重，但高层前向仍需向证据token反传梯度；禁止整个VLM包torch.no_grad导致模块无梯度。
如果现有checkpoint无法可靠生成所需文本，单列训练分支：在固定验证集上评估给高层VLM加LoRA的必要性；所有比较方法获得相同LoRA范围和数据/预算，不只给完整模型额外训练。
不得自动下载/替换checkpoint、宣称已有新权重，或把随机LM head输出当成高层能力。

五、主任务及subtask的边界
首版主任务限制为已核实动作能力内的两类，例如“拿起A”“将A放到B”；具体类别在能力审计后冻结，不预设模型能倒水、清洗或开门。
目标对象A必须来自当前候选实例；目的地B必须合法且在确认内容中明确。无目的地证据时提出选项或询问，不能默认最近容器就是用户目的地。
高层输出采用可解析记录：suggestion_id、scene/mapping版本、action、source_id、destination_id、自然语言说明、需要澄清项。
模型文本经过语法、物体存在性、能力和版本检查后才展示给用户；失败输出和幻觉计入评价，不能由解析器静默修成正确答案。
一次主任务确认可否覆盖多个subtask，是新的层级授权问题：必须显式定义父契约、允许的子动作/实例范围、终点条件和预算。
现有V2的单任务授权不能未经验证就自动扩成整条任务链授权。第一版可保持每个子任务走已有授权，先测清交互成本；减少确认次数的层级授权另做版本与负例验收。
STOP绕过神经推理直接到现有停止链路。目标/授权范围变化时撤销旧授权、清空动作队列、重新确认；普通画面更新不应无条件当成用户改目标。

六、训练数据和监督
6272个V2合成回合主要验证符号交互，不是带视觉/动作示范的新模块训练集，也不是V3测试集。
每个新样本需要：图像/场景实例、候选框/区域、公开选择与历史、合法任务候选、用户确认或独立人工目标标注，以及歧义/需澄清标注。
subtask接续需要带阶段边界和已确认主任务的语义子任务标注；连续动作训练另需对应机器人示范，不用失败推理轨迹冒充专家动作。
用模拟脑机噪声生成的历史标记synthetic；真人未采集就不能声明脑机识别或真实迁移有效。
训练/验证/测试按原始场景、底层episode和实例分组，然后才做噪声、改编号、换顺序、改写文本等增强；同源增强不得跨分区。
同一所选物体必须覆盖多个可能任务，防止“杯子=倒水”的类别捷径；无法从公开输入唯一确定的样本，标签应允许多个候选并要求澄清。
反事实组A：相同目标、不同输入映射/通道故障，在充分公开诊断证据后任务输出应一致。
反事实组B：通道稳定、用户明确改变目标，输出应跟随新证据，而非坚持旧任务。
同一公开输入对应不同隐藏目标且无额外证据的样本，不得强制模型区分；该组评估正确保留歧义与询问。

七、损失与训练步骤
主损失：合法主任务文本的teacher-forcing交叉熵；目标对象指针损失；人工/协议可辨识性支持的澄清监督。
反事实一致性和变更响应作为候选辅助项，权重只在训练/验证阶段选择，冻结后再开test。
所有辅助监督只在有可靠标签时计算，并保留缺失mask；未知不当负例，暂停不当错误目标。
不能用模型自己的高置信输出当真实脑机校准标签；执行成功/失败也不能用作“脑机选择正确/错误”的直接监督。
第一阶段模块训练不更新动作专家。需要LoRA时另存训练范围、参数数量、版本和对照公平条件。
训练应保存随机种子、精度、数据清单、优化器、步数、early-stop规则、实际参数量、峰值显存与完整日志。

八、必要对照和否证
B0：原π0.5语言接口，明确所获输入信息。
B1：把相同物体、选择与历史信息写成文本提示；这是信息量匹配的强基线。
B2：同样输入经池化MLP投影为K个prefix，不使用交叉注意力；匹配参数或另报参数/计算差异。
B3：标准查询连接器，保留相同输入与训练预算，移除本方案的交互来源处理/专用辅助监督；必须拆清结构与训练监督的收益。
Full：双路查询模块。
消融：去历史、将TASK/PROBE来源混合、选择分布改argmax、去辅助头、去反事实监督；主干训练范围及数据一致。
置乱通道历史/无效证据应能暴露模型依赖的真实信息；只依赖对象类别或固定位置的模型不算学会交互。
如果Full不稳定优于文本、MLP和标准查询连接器，不能声称本网络有独特方法优势。
如果额外通道信息无收益，应收缩为场景/对象交互融合；如果澄清头只复刻预设熵阈值，必须与该阈值基线比，不把它包装为新推理。
若收益仅来自更多训练数据、更多暂停或更多VLM微调，不能归因于新增模块结构。

九、评价与研究门槛
G-V3-0能力门槛：高层生成入口真实可用，来源与训练能力审计通过；否则停止下游能力宣称。
G-V3-1接口门槛：形状、mask、梯度、cache、关闭模块回归、物体重排、未知输入和确认/STOP检查通过。
G-V3-2离线门槛：合法候选覆盖、对象绑定、主任务接受、歧义澄清及校准有独立留出证据；先锁主指标及误执行/交互成本约束。
G-V3-3闭环门槛：LIBERO主任务独立终点成功、错误目标启动、澄清次数、完整耗时/失败恢复分别评价；正确文本不等于任务成功。
G-V3-4研究门槛：信息/预算匹配基线、公平消融、独立分组统计以及未见场景/故障验证支持新增贡献；不提前定“足够创新”。
无真实EEG/用户采集时，所有结论限定为合成输入与仿真；主观控制感、临床效果、真实跨BCI仍未验证。

十、已有方法与候选贡献
BLIP-2已用可训练Querying Transformer桥接冻结视觉编码器和语言模型；查询连接结构不作为原创。
Perceiver已通过潜在查询瓶颈处理可变多模态输入；不能将固定K查询宣称为首次。
π0.5已有高层语义预测和低层动作分支；两阶段推理及文本subtask也不作为新理论。
候选贡献只在充分证据后定位为：面向含噪选择、异源交互和可变对象的证据对齐机制，以及它对任务建议、澄清与纠正的实证增益。
当前没有穷尽最新文献，也没有V3实验结果，不宣称首次、普适或录用保证。
主要来源：
π0.5原论文，IV-A–D、V-E及技术附录：https://arxiv.org/abs/2504.16054
BLIP-2：https://arxiv.org/abs/2301.12597
Perceiver：https://arxiv.org/abs/2103.03206
Gaze2Act（已有交互线索条件化VLA，需进一步逐项比较）：https://arxiv.org/abs/2605.30282
INDI（已有意图表示进入动作解码器，不能靠加意图向量自动声称创新）：https://arxiv.org/abs/2608.23478

十一、本轮交付边界
本轮只生成V3设计书、逐步手册、机器可读设计与新进度模板；没有实现上述网络，没有训练或部署，没有新模型性能。
历史V2 progress.json保持步骤16和原G2/G3/G4状态。V3另用execution/v3_plan_progress.json，步骤号一律带V3-前缀。
根目录无版本手册也被历史冻结清单引用，保持原件。V3入口使用BCI_VLA当前计划入口.txt和带V3后缀的新手册；V2及旧源码/数据/结果保留。
'''
manual='''脑机—VLA框架：逐步执行手册V3（显式神经交互模块）
日期：2026-10-16；状态：设计计划，以下步骤均未执行。
先读：D:\\Scene_Aware\\BCI_VLA框架_神经交互模块方案_V3.txt。
当前源码基线：D:\\Scene_Aware\\bci_vla_framework；V2截至步骤16的研究负结果保持。

执行者规则
1 编号必须使用V3-01等前缀；不得把“执行步骤1”默认当作重跑V2旧采集器，先依据当前手册版本解析。
2 本次只写计划。后续执行各步时，以用户当次及既有授权为准；继续排除真实脑机与实体机器人，直到用户明确改变范围。
3 不执行git reset --hard、不覆盖旧输出、不改冻结V2来源、不把test用于调参、不显示凭据。
4 旧代码作只读基线，新实现建议放bci_vla/neural_interaction_v3、tests/test_neural_interaction_v3.py及execution/v3/；网络实现尚不存在。
5 每一步记录输入来源、动作、输出、命令/环境、通过范围、失败及下一步。计划文件存在不等于实现通过。
6 原执行器确认/授权/STOP不由神经网络绕过；原始选择、known-target probe、确认回应分别记录。
7 训练、远端部署、真实设备、外部模型调用均不得仅由本手册文字自动启动；执行相应阶段的实际用户指令才是工作授权。

V3-01 建立新版本与基线清单
输入：现有Git状态、V2步骤16报告、14C/15C代码、最新用户范围。
操作：
 a 运行git status --short和git log，确认当前用户变更，保留15C备份及16预注册提交。
 b 检查适用AGENTS.md，读取V2进度/验收和V3设计；记录新计划版本，不复位旧研究门槛。
 c 枚举新模块将调用的视觉特征、tokenizer、π0.5与适配器来源、版本/权重哈希、授权状态和能力缺项。
 d 只读核对服务器路径和运行资源；不输出凭据，不启动模型服务。
输出：execution/v3/01_inventory.json、01_scope.txt；更新v3_plan_progress.json。
验收：明确哪些是已实现、未实现、未校准、不可用；G2/G3/G4原状态未升级。
失败处理：路径或权重未知就记录未知，先做不依赖它的本地schema工作，不下载替代模型掩盖缺项。

V3-02 冻结公共接口与任务语法
输入：设计书3/5/6节、V3-01来源。
操作：
 a 定义scene_id/revision、实例ID、视觉来源、bbox、候选mask、选择证据类型/校准版本、历史event role/时间/版本。
 b 定义主任务候选记录、unknown与需澄清状态；source/destination引用同一scene实例。
 c 定义模块输入输出形状及合法批次；空物体/空历史、无响应、过期和映射改变都给出样例。
 d 定义私有训练/评估标签容器，禁止进入线上特征；合成数据必须标synthetic。
 e 限定首版最多16物体、最近8事件，写清截断和未知策略。
输出：configs/v3/interfaces.json、examples/v3/public_input.json、private_label_schema.json、docs/v3/02_interfaces.txt。
验收：ID/来源/时间混淆和非法任务应被拒绝；不能把未校准分数自动当概率。

V3-03 验证π0.5高层文本入口（能力门槛）
输入：已核实checkpoint、tokenizer、模型源码和高层任务样例。
操作：
 a 先读模型的文本输出头、embed_prefix、sample_actions和训练说明；区分原论文完整系统与现有LIBERO SFT实现。
 b 在隔离脚本实现最小文本teacher-forcing/解码入口，核对视觉/状态/文本输入、EOS、attention与cache。
 c 先用明确主任务的少量开发样例检查输出是否可解析；保存原始失败，不能用外部LLM代写成“π0.5生成”。
 d 报告checkpoint是否确有高层能力；若不足，明确需要高层训练/LoRA与相同预算基线，不能继续声称冻结主干即开箱可用。
输出：execution/v3/03_text_capability.json、docs/v3/03_capability_gate.txt、独立开发trace。
验收：证明真实模型路径可生成/训练所需文本，或给出清晰能力缺口；不得据模型名称判定通过。
失败处理：继续数据与模块CPU开发，但暂停任何高层任务成功声明；重新确认可训练来源和设计分支后再推进模型实验。

V3-04 建立小规模数据与分组划分
输入：已核实任务能力与公共schema。
操作：
 a 选两类物体、两类可执行主任务；每类物体覆盖不同意图，不能让类别决定目标。
 b 收集或整理场景图像、实例标注、选择证据/交互历史、主任务候选与确认、歧义标签；subtask标注另列。
 c 先按原始scene/episode/实例分组划train/validation/test，再做噪声/位置/顺序/语言增强。
 d 加入同目标通道变、通道稳目标变、公开证据完全别名需澄清三种组；私有计划单独存。
 e 保存数据来源/许可、标注规则、双人或独立复核方式、缺失mask与重复/增强血缘。
输出：data_manifest.json、split_manifest.json、annotation_rules.txt、不可变分区哈希。
验收：同源episode/增强不跨分区；历史不含未来确认；不能把V2的6272符号回合冒充视觉或动作示范。

V3-05 先建立信息量匹配基线
输入：开发/训练分区，不读取test结局。
操作：
 a 原π0.5文本接口基线；保留它实际获得的信息。
 b 相同场景/物体/选择/历史写成文本的强基线。
 c 相同输入通过池化MLP生成K个prefix的基线。
 d 标准query连接器基线，计划好与Full的输入、参数、LoRA、数据和步数匹配方式。
 e 记录候选覆盖、澄清及解析失败，不只展示正确文本。
输出：baseline_registry.json、每个方法独立源码/参数/输入表、开发结果。
验收：同一信息、同一任务定义和硬授权；无原LLM trace时记不可用，不伪造NOIR结果。

V3-06 实现新网络的CPU原型
输入：V3-02接口、固定随机张量与可复核小夹具。
操作：
 a 独立nn.Module实现物体投影、类型/空间编码和事件时间编码。
 b 实现1层历史编码、8个learnable queries、2个self/object/history cross-attention+FFN融合块。
 c 实现K个prefix输出、对象/unknown指针、澄清头；报告逐组件参数量。
 d 单测不同物体数/历史长、padding、全空、NaN输入、对象重排、batch隔离与invalid mask。
 e 检查所有合法路径梯度，禁止把private label/scene序号当特征。
输出：bci_vla/neural_interaction_v3/（建议名）、模块CPU测试、06_parameter_count.json。
验收：形状/掩码/重排/未知与梯度检查通过；这只证明实现正确，不是推理效果。

V3-07 接入高层VLM的embedding路径
输入：通过能力门槛的文本入口与新模块。
操作：
 a 在inputs_embeds中插入K个prefix，按真实config确定D_vlm，不硬编码假定隐藏维。
 b 同时更新padding/attention/position_ids、teacher-forcing标签偏移与KV cache。
 c 关闭模块走原输入序列；不要以“零token”替代关闭。固定随机数检查基线回归。
 d 冻结原模型权重但保留prefix反向梯度；测试无梯度/错dtype/错device能明确失败。
 e 第一版只让prefix参与高层文本；动作专家路径保持原样，避免一次改变多个因素。
输出：独立模型版本与07_integration_audit.json、off/on形状和梯度证据。
验收：未训练模块无能力声明；关闭回归、cache失效、标签对齐和反向传播均通过。

V3-08 训练开发版本
输入：train/validation、通过软件门槛的网络与模型。
操作：
 a 冻结训练范围、显存/运行时预算、种子、损失与优化器；保存样本分组哈希。
 b 先单batch验证loss与梯度，再小规模拟合开发夹具，排查代码错误；不以记住夹具宣布泛化。
 c 训练文本主损失与可靠标签上的辅助损失；候选反事实项在validation决定，记录所有尝试。
 d 若加入LoRA，基线和Full使用相同可训练主干范围；别只给Full更多容量。
 e 保存权重、完整失败/停止日志、选模规则、参数/计算/时间成本。
输出：训练配置、日志、checkpoint manifest、验证报告。
验收：数据血缘与训练范围可复核，模块真实被使用；未达到验证目标不打开test挑参数。

V3-09 冻结并执行离线独立留出
输入：选定开发版本、预先锁定主指标和原始test分组。
操作：
 a 冻结checkpoint、超参数、候选语法、澄清规则、基线、样本量与分组统计方法。
 b 同时评估对象绑定、合法主任务候选覆盖、主任务接受/更正、应澄清样本及完整交互成本。
 c 所有方法运行相同分区；保存raw文本、解析失败、拒绝、暂停，不只统计成功。
 d 做去历史/来源混合/argmax/去辅助头/去反事实的预声明消融。
 e 未见场景/对象排列/物体实例及未见噪声参数单列；不把随机增强行当独立被试。
输出：离线总表、配对区间、失败案例、G-V3-2报告。
验收：相对文本、MLP、普通query连接器的收益可定位，或诚实记未建立；相同不能写优越或等效。

V3-10 接续subtask与父子任务约束
输入：已确认主任务、当前画面/状态、高层生成能力和任务语法。
操作：
 a 让同一高层VLM在另一模式下生成当前语义subtask；先使用已有训练覆盖的粒度。
 b subtask绑定当前实例和主任务，不直接输出绕过适配器的关节控制。
 c 明确父任务授权范围；未实现层级授权前沿用逐子任务现有授权，并记录额外交互时间。
 d 编写换目标、超范围、缺目的地、旧版本、STOP以及旧cache的负例。
输出：subtask schema、父子契约设计与测试、失败回退流程。
验收：语言子任务可执行且在已确认范围内；高层能讲计划不等于机械臂会做。

V3-11 LIBERO完整闭环验证
输入：通过离线门槛的新模型、冻结执行适配器与独立终点谓词。
操作：
 a 先无动作核对模型/图像/状态/指令/动作语义；再运行隔离仿真服务，终止只清理本任务进程。
 b 执行选择→主任务提议→模拟确认→subtask→动作→独立终点；日志标明模拟BCI。
 c 测目标改变、通道翻转、来源混淆、未知/NO_RESPONSE、推理失败、STOP及旧队列清理。
 d 将任务成功与控制检查分别计数；执行失败不当作通道监督，所有视频/失败和完整耗时保留。
 e 不用官方另一任务成功字段替代当前主任务谓词。
输出：新版本闭环证据与G-V3-3报告，不覆盖14C/15C。
验收：完整链路与独立任务终点可复核；仍失败则保留，不以工程检查通过冒充成功率。

V3-12 接入真实视觉前端（仍不做真实脑机/实体动作）
输入：已核实Florence/SAM版本、非敏感场景图像和可用标注。
操作：
 a 读原视觉脚本和checkpoint；确认物体检测、分割、坐标、实例关联与跨帧ID的具体来源。
 b 使用静态/录制图像检查漏检、重排、遮挡、同类多实例和选择频率映射的版本绑定。
 c 单列真实视觉与特权模拟器来源，测试时不得把真实标注偷偷作为实际检测输出。
输出：视觉适配、ID/映射一致性记录和来源分层报告。
验收：感知误差与交互误差分开；未成功不能宣称已完成真实摄像头脑机闭环。

V3-13 公平对照、跨场景与证据验收
输入：所有冻结模型与数据/任务/统计方案。
操作：
 a 在共同视觉来源、输入信息、训练预算和授权下比较B0/B1/B2/B3/Full。
 b 按目标变化×通道变化区分错误归因，报告误启动、任务完成、暂停和用户完整交互成本。
 c 区分仅新数据、仅高层监督、仅连接器及完整设计收益；合成/视觉/实际VLA来源各自报告。
 d 主观量表、真人脑机和真实2×2仅在另有实际授权及条件时开展；否则明确缺失。
输出：G-V3-4报告、统计与可重复证据清单。
验收：达到预先定义的研究条件才支持相应贡献；无收益则停止扩充故事并定位信息是否冗余。

V3-14 更新论文、简历和交付
输入：G-V3各门槛的实际结果及V2负结果。
操作：
 a 准确写“修改高层输入与任务生成分支”，不写“重设计动作专家”或“首次通用脑机框架”。
 b 模块尚无收益时称研究原型；不能用V2的409测试/6272回合证明V3神经网络有效。
 c 绘制含来源、确认权限和高/低层边界的框架图；逐条绑定贡献与证据。
 d 保存新代码、配置、文档、权重清单和复现实验入口；Git备份排除凭据、真人原始数据和大权重。
输出：论文修改稿、claim-evidence映射、复现说明与版本化备份。
验收：每项结论有对应证据和范围，失败与缺项可查；投稿、正式发布和真实设备操作按用户实际要求执行。

本轮计划验收
只检查V3方案与手册是否具体、互相一致、保留V2事实，所有步骤初始状态均为未开始。
不运行上述网络/训练/仿真步骤，不把计划写完记为新研究门槛通过。
'''
txt(W/core_name,core);txt(W/manual_name,manual)
txt(W/'BCI_VLA当前计划入口.txt','当前新研究计划：V3设计草案（尚未实现/训练）。\n先读BCI_VLA框架_神经交互模块方案_V3.txt，再读BCI_VLA通用交互框架_逐步执行手册_V3.txt。\n执行编号使用V3-01等前缀。V3进度：bci_vla_framework/execution/v3_plan_progress.json。\n旧的无版本手册及V2手册为冻结历史，不能覆盖；历史V2步骤16及G2/G3/G4状态不变。\n')
spec=dict(version='V3-design-1',status='design_only_not_implemented',name='scene_and_interaction_evidence_query_network',
 architecture=dict(hidden_dim=256,heads=4,fusion_blocks=2,ffn_dim=1024,queries=8,history_events=8,max_objects=16,
 object_encoder='frozen ROI/semantic features + box + typed selection evidence -> projection',history_encoder='one temporal Transformer layer with role/time/version masks',
 fusion='query self-attention -> object cross-attention -> history cross-attention -> FFN',
 outputs=['K soft evidence prefix tokens','candidate-instance/unknown pointer','clarification proposal'],
 injection='pi05 high-level VLM inputs_embeds only in first prototype',action_expert_modified=False),
 defaults_are_unvalidated=True,new_weights_available=False,new_network_implemented=False,
 evidence_sources=['privileged LIBERO or annotated scene for early prototype','Florence/SAM front-end after separate verification','synthetic BCI until real calibration exists'],
 hierarchy=['object selection evidence','propose main-task candidates','explicit user confirmation','semantic subtask','action expert','independent endpoint'],
 backbone_text_generation_capability_verified=False,V2_research_gates_upgraded=False,
 baseline_requirements=['same-information text','pooled MLP prefix','standard query connector','Full matched training and supervision controls'],
 literature=['https://arxiv.org/abs/2504.16054','https://arxiv.org/abs/2301.12597','https://arxiv.org/abs/2103.03206'])
save(R/'configs/design_v3.json',spec)
titles=['建立新版本与基线清单','冻结公共接口与任务语法','核实π0.5高层文本能力','建立分组数据','信息匹配基线','CPU实现显式查询网络','高层embedding接入','训练开发版本','独立离线留出','subtask与层级约束','LIBERO闭环','真实视觉前端','公平研究验收','论文与交付']
save(R/'execution/v3_plan_progress.json',dict(plan_version='V3-design-1',current_stage='design_written_only',implementation_started=False,
 historical_progress_ref='execution/progress.json',V2_current_step='16',V2_gates=dict(G2='未通过',G3='未验证',G4='未通过'),
 steps=[dict(step_id=f'V3-{i:02}',title=t,status='未开始') for i,t in enumerate(titles,1)],
 authorization='rewrite concrete module and executable plan only; no training or deployment'))
notice='2026-10-16 V3设计计划入口：根目录BCI_VLA框架_神经交互模块方案_V3.txt与BCI_VLA通用交互框架_逐步执行手册_V3.txt。\nV3为显式双路查询网络设计，尚未实现/训练。新编号V3-01起，进度在execution/v3_plan_progress.json；历史V2步骤16及G2/G3/G4状态不变。\n'
for f in ['README.txt','execution/handoff.txt','execution/worklog.txt']:
 p=R/f;txt(p,notice+'\n'+p.read_text(encoding='utf-8-sig'))
files=[W/core_name,W/manual_name,W/'BCI_VLA当前计划入口.txt',R/'configs/design_v3.json',R/'execution/v3_plan_progress.json',Path(__file__).resolve()]
save(R/'execution/v3_design_manifest.json',dict(scope='planning_only_not_network_implementation',files=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
print(json.dumps(dict(result='WRITTEN',files=[str(p) for p in files[:5]],network_implemented=False,trained=False),ensure_ascii=True))
