# OpenPI RTC 双 Piper 实机部署

本文对应两台机器：

- 工控机：`172.19.2.142`，仓库 `/home/agilex/openpi`，运行双 Piper、三路相机客户端。
- 推理机：`172.19.5.252:9000`，仓库 `/home/oyjt/openpi`，运行 Pi0.5 策略服务。

目标 checkpoint 是推理机当前 OpenPI 配置里的 `uniform` Pi0.5，动作形状为 **100 步 × 14 维**。RTC 是推理时的动作前缀引导，不需要重新微调 checkpoint。这里把 RTC 的指数前缀权重和伪逆引导适配到了 OpenPI 的 JAX Pi0/Pi0.5 去噪采样流程；参考 [RTC 论文](https://arxiv.org/abs/2506.07339) 和 [Physical Intelligence RTC 实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)。

## 修改内容

- 服务端在每个去噪步中，根据上一动作块和预计推理延迟引导新动作块；只接受 `exp` 前缀权重方案。WebSocket 策略元数据公布 RTC v1 和动作形状。
- 客户端启动时先连接策略服务并检查 RTC v1、100×14 动作形状；服务端不支持时会在连接 Piper 前退出，不会静默退回普通分块推理。
- 客户端在新一轮请求中发送时间对齐后的上一动作块、推理延迟估计和前缀范围。延迟估计从 20 步初值开始，再按实际往返时间平滑更新。
- 工控机客户端直接发送 Aloha 推理变换要求的顶层 `images.cam_high/cam_left_wrist/cam_right_wrist` 与 `state`；Aloha 夹爪宽度使用分米，状态按毫米除以 100，动作按分米乘以 100000 转成 Piper 单位。服务端把上一动作块重新走一遍输入变换，确保 RTC 在模型归一化动作坐标里工作。
- 去掉了会改变 RTC 承诺动作前缀的客户端动作 EMA、跨块融合和夹爪死区；关节仍按原控制频率做硬件插值。首个动作块从索引 0 开始，之后的动作块按已经过去的控制步对齐。
- 服务端返回过期动作块或连续 30 个控制步收不到新动作时，客户端停止控制循环。
- 实机输出默认锁定；`OPENPI_DRY_RUN=1` 只检查服务端 WebSocket 与 RTC 元数据，不连接或使能机械臂。实际动作必须显式设置 `OPENPI_ENABLE_ROBOT=1`。

## 启动顺序

### 1. 推理机启动策略服务

等推理机上占用 GPU 的训练任务结束后，在 `172.19.5.252` 执行：

```bash
cd /home/oyjt/openpi
.venv/bin/python scripts/serve_policy.py --env ALOHA
```

当前仓库的 `ALOHA` 默认 checkpoint 配置指向 `/home/oyjt/workspace/checkpoints/uniform/uniform24_horizon100_0326/29999`，端口为 9000。服务启动后可在推理机上检查：

```bash
curl -fsS http://127.0.0.1:9000/healthz
```

### 2. 工控机做无硬件干运行

确认策略服务健康后，在 `172.19.2.142` 执行：

```bash
cd /home/agilex/openpi
OPENPI_DRY_RUN=1 .venv/bin/python scripts/RTC_inference.py
```

干运行成功应报告 RTC v1 和 `100×14`，随后退出。这个步骤不连接相机或 Piper，也不会执行模型动作。

### 3. 实机启动

先清空双臂工作区并确认急停可用。脚本使能机械臂后会先执行现有归位动作，再开始推理控制：

```bash
cd /home/agilex/openpi
OPENPI_ENABLE_ROBOT=1 .venv/bin/python scripts/RTC_inference.py
```

可以通过 `OPENPI_PREFETCH_STEPS`、`OPENPI_LATENCY_STEPS`、`OPENPI_RTC_INITIAL_DELAY_STEPS`、`OPENPI_RTC_PREFIX_EXTRA_STEPS` 和 `OPENPI_RTC_MAX_GUIDANCE_WEIGHT` 调整时序与引导强度。先从默认值运行受控低速任务，并观察日志中的往返时间、RTC 前缀延迟和动作块切换索引，再调参数。

## 当前部署状态与范围

推理机的 9000 端口检查时没有服务监听；当时机器上有一个占用 GPU 的四卡训练进程。为避免影响该任务，没有启动推理服务，也没有向机械臂发送动作。服务端代码和工控机客户端已经准备好；服务启动后应先完成上面的干运行。

目前 RTC 服务端只支持 JAX Pi0/Pi0.5。PyTorch checkpoint、BID、hard masking 和普通 naive 模式没有接入此客户端流程。