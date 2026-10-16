步骤08固定别名世界的软件与解析例，不是实验数据。
alias_events.json含手写TASK/PROBE请求响应；规范目标ID及问题绑定位于configs/belief/simulation_binary_filter.json。
fixed_state_transition.json以零变化率复现窗口固定的等价证明；非零时间转移例位于configs/belief/simulation_canonical_transition.json。
alias_expectations.json的old/new/goal-0/1映射仅展示显式对应，程序不会自动认这些别名。
NO_RESPONSE、unknown、none和unknown_channel不是可互换标签；未知状态缺似然时暂停，不能删除它们的正质量。
运行execution/replay_step08_alias.py生成审计trace，不读取测试真目标、不执行设备、不测试策略性能。
