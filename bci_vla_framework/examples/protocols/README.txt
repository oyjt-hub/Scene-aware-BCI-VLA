步骤05软件示例
全部是手写协议fixture，不是EEG采集、参与者实验、真实校准或实际执行记录。
task_query/channel_probe/confirm各有request与response，共6份；stop/cancel/execution_event共3份合法事件。
invalid_cases.json含8种非法结构及预期错误码；运行时旧版本/超时/重叠/乱序等另由单元测试构造。
public_records.json为7类公开模型声明。
示例时钟固定2026-10-15T00:00:00Z，须显式注入相应now和本机monotonic_now进行回放，不能用今天的真实时间把fixture当在线消息。
probe提示0与response选择1故意不同，测试来源分离，不代表系统已经诊断成功。
CONFIRM保持proposed契约，不自动执行；EXECUTION_EVENT只是结构示例，不代表动作发生。
测试命令在项目根目录执行：python -B -m unittest discover -v。
