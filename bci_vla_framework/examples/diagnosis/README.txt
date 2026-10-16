步骤09固定模型规划示例；simulation_only。
state.json来自08规范目标与相关别名先验，新增两个proposed契约和20秒假设预算。
时间2026-10-15 UTC是固定fixture，不能替换成本轮日期后当真实记录。
prepare_step09_assets.py只显式将07 old/new映射为left/right规范ID，保留全部原始损失，不拟合参数。
python -B execution/replay_step09_policy.py保存09_policy_trace.json；没有随机性能实验/真实事件/机器人动作。
详情docs/09_policy.txt；执行前需11真实契约与授权链，不能将模拟accept当用户确认。
