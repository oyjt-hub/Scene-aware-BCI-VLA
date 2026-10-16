# 真实数据采集与意图标注规范

本文件定义人工标注过程。`examples/`或命令生成的合成数据仅用于代码冒烟测试，不能直接成为论文有效性数据。

## 1. 一例数据的边界

每个case包括公开观察 `observation` 与仅用于训练监督/评价的 `annotation`。模型、规则和LLM只能接收前者。case_id、scene_id、分组、子集标签、隐藏用户请求、标注者意见、终态和annotation均不得拼入提示或特征。

```json
{
  "case_id": "scene_001_context_01",
  "scene_id": "scene_001",
  "observation": {
    "objects": [
      {"id":"grapes_a","label":"grapes","position":[0.2,0.5],"attributes":[],"affordances":["pickable"],"detection_confidence":0.97},
      {"id":"bowl_a","label":"bowl","position":[0.7,0.5],"attributes":[],"affordances":["container"],"detection_confidence":0.98}
    ],
    "relations": [{"subject":"grapes_a","predicate":"left_of","object":"bowl_a","confidence":0.98}],
    "selection": {"object_id":"grapes_a","probabilities":{"grapes_a":0.95,"bowl_a":0.05}},
    "context": {"goal":"put_away","confirmed_action":null,"confirmed_destination_id":null,"destination_hint":"unspecified","history":[]}
  },
  "annotation": {
    "status":"ready",
    "intent":{"action":"place","source_id":"grapes_a","destination_id":"bowl_a"},
    "clarification_slot":null
  }
}
```

该示例的ready依赖于已公开的“put_away”及唯一适用容器，而不是因为标注者看到水果就猜应当收纳。历史必须是系统当时已知、已确认的信息。`goal=repeat_last`只用于用户明确选择“重复上次操作”的情况，复用上一条已确认动作和目的地，源为本次BCI选择；历史本身不构成重复授权。模型输入不应含未来的成功执行信息。

## 2. 物理场景与可观测内容

- 在采集时分配不可变scene_id，保留原始照片/视频和录制批次；同一摆放的所有上下文变体使用同一scene_id。
- 对同一原始录像、邻近帧、裁剪/增强和高度相近的重复摆放记录共同group_id；拆分时先按该组归并，避免仅换scene_id造成泄漏。
- 对象id在一个场景内唯一，只用于内部引用。相同类别多个实例必须分别标识。自然语言不得使用这些不透明id。
- 位置为图像坐标归一化xy；属性仅记录图像中能可靠辨认的颜色/形状等，不写“用户想要的”或推断出的用途。
- 自动感知版保留实际感知输出及错误；人工纠正版另存，不覆盖原文件。不要在看到某方法错误后只修它的输入。
- affordance取当前任务语法支持的pickable、container、foldable。记录来源与人工规则；它是外部提供的信息，不应称为模型自主发现。
- BCI object_id与后验使用当时真实输出。没有记录的置信度不要人为填成高置信度；受控人为扰动另标synthetic_perturbation。

## 3. 状态和意图的判定

两名标注者先只看公开observation，回答“系统凭现有信息是否足以提出一个确定任务”，然后再核对隐藏用户请求。隐藏请求只能检验真实意图，不得用来补足系统未得到的信息。

**ready**：公开信息明确支持一个有效的动作、源实例及必要目的实例，且目标可通过可观测描述唯一指认。所有ready都仍需要用户确认。

**clarify**：存在合理任务，但缺少选择源、动作或目的地信息；也包括低置信度BCI需要重新选择、同类实例不能唯一指认、多个同样适用的目的地。不要把“最常见动作”当唯一正确答案。若场景中有两个同样合理的碗且用户没指定，应标clarify，而非把隐藏请求中的碗设成系统必须猜中的标签。

**reject**：输入损坏、没有有效对象、明确任务与对象可供性冲突、明确指定对象不存在，或不在当前支持任务语法中且无法通过一次受支持澄清解决。记录具体原因。schema非法的数据原则上在数据质检阶段剔除/单列损坏输入测试，不混入普通任务准确率。

待澄清slot依序确定：source → action → destination。一次只问一个。若先消歧源后还会有目的地歧义，当前标签只标source，下一轮使用新的case并保持同一scene分组。数据存在多种同样合理的澄清顺序时，先修订统一规则再标注，不能按方法答案修改真值。

ready时intent为 `{action, source_id, destination_id}`，pick与fold的destination_id为null；place必须有明确有效容器。clarify/reject时intent为null；reject的clarification_slot为null。不要在clarify标签中写入隐藏意图，以免训练代码误用。

## 4. 同类实例与语言可定位性

可用实例描述依次考虑：独有可观测属性、可靠位置关系、与明确参照物的关系。两个碗若位置不同且图像清楚，可以使用“left bowl”；重叠、同位或缺少参照时不能硬写“bowl_2”。精确数值坐标不是对VLA可理解性的证明。

标注者应独立检查模板生成的指令是否唯一指向目标。若模板不能表达但场景可区分，应修复共享renderer后重新评价所有方法；若场景本身无法区分，标clarify。模板能力与下游VLA理解能力需要分别验证。

## 5. 一致性、裁决与冻结

先试标20例，比较状态、动作、源、目的和slot，记录分歧原因并修订规则；试标进入训练或单独留存，不进入独立盲测。正式数据由两人独立标注，均看不到任何方法的预测。保存初始标注，不用裁决结果覆盖。

报告裁决前状态一致率与Cohen's kappa（状态类别），并报告仅双方均标ready例的意图元组一致率和clarify slot一致率。类别分布严重不平衡时同时给混淆计数，避免只报kappa。所有分歧由第三人或预先指定仲裁者依据公开信息裁决，并写简短理由。

冻结后生成split清单与内容哈希。错误修订需记录日期/原因/受影响case，统一重跑全部方法；如果改标依赖测试输出，应在论文中披露，并额外保留原冻结集结果。不能删除Full失败而其他方法成功的困难样本。

## 6. 现场试验记录

每次记录参与者匿名id、任务id、scene/group、方法条件、顺序、系统版本、选择与置信度、澄清问答、确认/拒绝、提案与最终指令、VLA随机种子、起止时间、物理终态、停止/接管、失败原因和恢复耗时。BCI采集和人体实验数据按既有批准与参与者同意处理。

终态由预先定义标准和录像判断，不能以“指令生成正确”替代执行成功。失败归因允许多个标签，但指定首个导致任务失败的环节：感知、BCI、意图推理、语言指向、VLA、控制/通信、用户取消、超时。不要把所有失败都归入VLA以掩盖错误指令。
