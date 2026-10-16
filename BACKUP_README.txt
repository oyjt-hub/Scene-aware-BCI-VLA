BCI–VLA代码与实验记录备份
日期：2026-10-16；截止步骤15C。
远程：git@github.com:oyjt-hub/Scene-aware-BCI-VLA.git
分支：backup/framework-step15c-20261016
这是当前本地研究工作区的独立快照分支，未合并或改动远程main。

目录导航
bci_vla_framework/：当前主项目；代码、配置、测试、设计、冻结来源和非真实仿真证据。
bci_vla_framework/docs/14C_full_simulation_report.txt：步骤14完整非真实闭环。
bci_vla_framework/docs/15C_full_transfer_report.txt：步骤15完整非真实迁移矩阵。
bci_vla_framework/execution/progress.json：当前步骤、未完成事项及G2/G3边界。
context_intent/：早期场景上下文小模块原型；当前主线以bci_vla_framework为准。
根目录BCI_VLA*.txt：框架方案、逐步执行计划及历史版本，原文件名保留。
BACKUP_MANIFEST.json：此次纳入版本控制的文件相对路径、SHA256与字节数。

验证状态
步骤14C：398项软件测试、35项完整性检查通过。
步骤15C：401项软件测试、51项完整性检查通过；两种软件输入×两个实际VLA的LIBERO工程矩阵完成。
实际抓取仍有失败，所有仿真失败保留；真实脑机、真实2×2迁移、实体机器人未开展。G2效用未通过、G3未验证。

排除内容与恢复限制
账密文件、.server-inspect连接辅助目录、编辑器调试目录、论文原稿目录、缓存、虚拟环境、环境密钥及模型权重不上传。
真实脑机raw_stream.bin与processed_signal.npy只在原本地保存；模拟TCP夹具原始流仍作为软件证据保留。
context_intent沿用自己的.gitignore，生成的smoke_data和runs不纳入该模块代码备份。
历史冻结清单包含原Windows绝对路径，并有本地/服务器资产引用；恢复到其他目录后须按相对路径映射验证，不能直接修改原冻结清单。
真实原始流未上传，因此依赖它的历史完整性检查需恢复本地私有资产；这份Git备份不宣称包含所有外部运行环境与权重。
不移动或重写已冻结源码、日志、验收结果；当前目录结构便于原路径恢复和来源核查。

使用
将分支克隆到D:\Scene_Aware可保持现有本地引用路径；其他目录可先按BACKUP_MANIFEST.json验证文件内容。
先阅读execution/progress.json与最新14C/15C报告，再按pyproject.toml配置Python环境。
服务器相关部署器需原本地私有连接配置；不要直接重跑带防覆盖约束的旧部署/采集/终结器。
