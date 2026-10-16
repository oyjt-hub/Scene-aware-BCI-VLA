步骤07软件fixture，非场景检测/机器人执行/独立真实标注记录。
scene.json声明稳定ID和归一化图像坐标，不证明跟踪器身份真实。
goal_hypotheses.json的old/new仅为模拟表别名，分别明确绑定左/右碗假设；不是真实用户目标，也不按时间自动更新。
task_candidates生成的哈希goal_id与这些演示别名不会自动混用；后续08必须显式建立一致状态词表。
trajectory.json是手写终态算例，用来核同轨迹多目标和缺失恢复值，不能作为真实独立轨迹。
全部成本配置见configs/consequences，status=simulation_only且禁止live；真实能力与后果均未验收。
