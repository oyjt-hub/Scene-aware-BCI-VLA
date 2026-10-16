Scene-aware-BCI-VLA 服务器仓库检查报告

检查日期：2026年10月15日。
服务器仓库：`/home/oyjt/Scene-aware-BCI-VLA`。
分支：`main`；当前提交：`d224552`（2026-09-30）。检查时 `git status --short` 为空。

本次读取了目录、代码、配置和Git记录，并对Python源码做静态解析，没有启动训练、视觉服务、BCI或机械臂。以下行号均指检查时的服务器版本。未修改服务器文件。

**整体结构**

这是论文系统的集成仓库，包含项目脚本及两套附带的上游框架。共1021个Git跟踪文件，其中Grounded-SAM-2占826个，OpenPI占158个，scripts占30个，同名BCI子目录占5个。

```text
/home/oyjt/Scene-aware-BCI-VLA/
├── README.md
├── Scene-aware-BCI-VLA/          在线BCI界面、解码、预处理、触发盒SDK
│   ├── interface.py
│   ├── fbcca.py
│   ├── creat_raw_data.py
│   ├── triggerBox.py
│   └── README.md
├── scripts/
│   ├── inference.py             场景选择→Gemini指令→双臂执行的集成入口
│   ├── server_api.py            实际为视觉与BCI结果中转服务
│   ├── bci/                     EEG接收与另一份预处理/解码代码
│   ├── robot/                   多种推理、RTC、手柄/键盘和消融执行脚本
│   ├── training/                JAX/PyTorch训练、归一化统计、策略服务
│   ├── data_tools/              LeRobot/ROS转换、录制、日志统计
│   └── docker/                  容器相关配置
├── openpi/                      附带的π0/π0.5框架，已包含RTC改动
├── Grounded-SAM-2/              附带的Florence-2/SAM-2视觉框架及实验脚本
└── src/                         当前为空
```

**系统如何连接**

```text
机器人相机
  → POST /update_scene
  → Florence-2/SAM-2输出对象、掩膜和场景信息
  → Windows BCI客户端 GET /get_bci_scene
  → 掩膜闪烁、EEG接收、FBCCA选择目标
  → POST /submit_bci_result
  → 机器人客户端 GET /get_bci_result
  → Gemini根据对象标签和选中对象生成英文指令
  → obs["prompt"]
  → OpenPI WebSocket策略服务返回动作块
  → 双Piper控制与动作时序处理
```

该流程由多个进程和主机合作实现，并非一个文件启动全部功能。

**BCI部分的实际职责**

| 文件（相对仓库根目录） | 实际职责及关键位置 |
|---|---|
| `Scene-aware-BCI-VLA/interface.py` | Windows在线客户端；场景获取65–82行，掩膜闪烁138–242行，EEG解码250–273行，结果提交293行 |
| `Scene-aware-BCI-VLA/fbcca.py` | FBCCA；68–85行进行子带相关系数加权平方求和，93–110行构造参考信号 |
| `Scene-aware-BCI-VLA/creat_raw_data.py` | MNE九通道预处理：0.3–90Hz带通、50Hz陷波、降采样至250Hz，19–41行 |
| `Scene-aware-BCI-VLA/triggerBox.py` | Neuracle串口触发盒SDK；不是刺激绘制程序。`interface.py:108` 当前将triggerbox设为None |
| `scripts/bci/dataServer.py` | EEG的TCP接收线程、设备解析与环形缓冲；58–84、123–132、212–224行 |
| `scripts/bci/fbcca.py` | 另一份FBCCA，核心算法相同，主要差异为路径和绘图输出 |
| `scripts/bci/creat_raw_data.py` | 另一份预处理；返回ndarray，嵌套目录版返回MNE RawArray，不能直接互换 |
| `scripts/bci/BCIMAP.py` | 实际生成空白电极头型SVG/PNG，4–17行；不是对象目标映射逻辑 |

当前在线界面配置：60 FPS、4秒刺激；1000Hz采集、接收12通道，解码只取前9通道；刺激频率按目标数在2–10Hz均匀分配，客户端调用7子带FBCCA。该界面待机阶段以回车键启动，见`interface.py:127–129`；不能据此认定论文描述的眨眼启动已在这个版本接通。日志记录选择时间、解码耗时、频率、标签及样本数，没有在结果接口中传送解码置信度。

**上下文推理部分**

`scripts/inference.py:347–407` 的 `infer_intent_with_gemini()` 是实际实现，不是空函数。

- 输入是场景对象标签列表与选中对象，见584–587行。
- 364行创建`gemini-2.5-flash`，382–395行构造提示，396–400行调用模型并返回英文指令。
- 当前提示要求从星果、葡萄、樱桃放碗和折布等示例指令中选择；未见独立的候选意图分布或澄清闭环。
- 195行把`TASK_INSTRUCTION`加入`obs["prompt"]`，201行调用策略。
- `USE_LLM=False`在353–357行只返回对象名词，没有生成完整规则指令。
- 网络调用失败时，402–407行降级为`pick up the 对象`。

`scripts/robot/SSVEP_inference.py`也包含相应流程。此次检查未发现“DiT生成自然语言L替换Gemini”的模块。

**视觉部分**

`Grounded-SAM-2/`包含Florence-2/SAM-2、GroundingDINO及多种检测、分割和跟踪示例。`experiment.py`、`grounded_sam2_florence2_image_demo.py`等包含项目相关实验代码。

`scripts/server_api.py`实际导入`grounded_sam2_florence2_image_demo`，提供`/update_scene`、`/get_bci_scene`、`/submit_bci_result`、`/get_bci_result`等FastAPI路由，默认8000端口；它与`Grounded-SAM-2/server_api.py`内容相同。README将其描述为π0.5策略服务，与代码不符。

**机器人执行和RTC**

| 文件 | 用途 |
|---|---|
| `scripts/robot/RTC_inference.py` | 历史动作前缀、延迟估计和RTC策略请求；398–409行检查RTC v1及动作形状，416–422行有dry-run分支 |
| `scripts/robot/action_queue.py` | 带RTC替换逻辑的线程安全动作队列 |
| `scripts/robot/inference.py` | 常规多线程VLA执行 |
| `scripts/robot/yibu_inference.py` | 提前推理、延迟补偿与平滑换块版本 |
| `scripts/robot/raw.py` | 去掉平滑的执行变体 |
| `scripts/robot/old_inference.py` | 较旧执行版本 |
| `scripts/robot/experiment.py` | 另一套BCI/视觉交互与计时实验流程 |
| `scripts/robot/gamepad.py`、`keyborad.py` | 手柄、键盘直接操作 |

RTC改动确实在这个仓库的内嵌OpenPI中。检查时，内嵌的`pi0.py`、`policy.py`、`policy_config.py`、`websocket_policy_server.py`与服务器单独`/home/oyjt/openpi`中的对应文件SHA256一致。

`openpi/src/openpi/models/pi0.py:67`声明`rtc_supported=True`；219–229行在`sample_actions`增加历史动作块、延迟、前缀范围、引导权重参数；297–307行将动作前缀误差用于去噪引导。这属于动作采样过程的改动，作用不同于Gemini上下文推理，也不是自然语言输出头。`policy.py:57`当前仅在非PyTorch后端开启RTC支持。

**训练、数据与工具**

- `scripts/training/train.py`：JAX/Flax训练。
- `scripts/training/train_pytorch.py`：PyTorch、DDP训练及checkpoint恢复。
- `scripts/training/compute_norm_stats.py`：按配置计算并保存归一化统计。
- `scripts/training/serve_policy.py`：根据配置和checkpoint启动WebSocket策略服务；99–117行为主要服务入口。
- `scripts/data_tools/convert_to_lerobot.py`：目录、PKL、ROS bag转换。
- `scripts/data_tools/ros_to_lerobot.py`：ROS推理与交互式LeRobot录制，涉及机械臂控制，不能作为普通离线检查脚本直接启动。
- `scripts/data_tools/check.py`：CSV耗时统计。

仓库中未发现`.safetensors/.pt/.pth/.ckpt/.parquet/.h5/.bag`等权重或数据产物，`.gitignore`也明确排除了模型、数据与日志。训练配置引用仓库之外的模型、assets与数据路径；`uniform`配置指向`oy/uniform24_v21`，默认任务为`fold the cloth.`。所以代码已存在，不代表仅克隆仓库即可复现或运行。

**后续需要核对的问题**

1. BCI采样率处理：嵌套目录的`creat_raw_data.py:38`已降至250Hz，但`interface.py:255–257`又按1000Hz切片并每4点抽样，259行仍按250Hz生成参考信号。这是静态代码中的采样率错配，需要用实际部署版本及回放验证，不能直接据此推断已发表/已报告的实验数据结果。
2. BCI导入路径：`interface.py:14`直接导入`dataServer`，但该模块在`scripts/bci/`，不在同一目录，独立启动需要整理模块路径。视觉服务也有跨目录导入依赖。
3. 日志含义：`server_api.py:258–270`在收到BCI选择后即把当前trial的`success`置True并保存；这个字段不能未经核对就当作机器人最终任务成功率，其计时也不自动覆盖机器人执行。
4. 入口与端口：README把视觉服务误写为策略服务；`serve_policy.py`默认8000端口，而机器人客户端默认9000，需要显式统一，避免与视觉服务冲突。
5. 脚本版本分散：候选物体、固定cloth任务、服务器、摄像头和CAN路径分布在多个文件；需要先指定哪份代码对应哪组论文实验，再统一配置。
6. 静态语法：`scripts/robot/test_server.py:1`存在异常字符，`ast.parse`报告`unexpected indent`。这只确认该文件语法问题，不代表其他脚本运行已验证。
7. 现有LLM关闭分支仅输出名词，后续做论文消融时仍需要实现同输入、完整指令的规则基线。

当前最适合进一步查看的三个入口为：BCI客户端`Scene-aware-BCI-VLA/interface.py`、上下文推理`scripts/inference.py::infer_intent_with_gemini`、RTC动作采样`openpi/src/openpi/models/pi0.py::sample_actions`。新文本推理模块如继续通过自然语言接原VLA，应先在明确的上下文模块接口上比较，而不是混用现有多个执行脚本。
