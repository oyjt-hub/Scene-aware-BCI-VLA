# 场景与BCI上下文意图模块

先阅读[模块设计](docs/MODULE_DESIGN_ZH.md)。当前交付为**设计和原型代码**；按用户最新要求，实验暂不开展。尚未训练，无可部署的神经网络权重，服务器原仓库未修改。

核心流程：对象与关系 → 小型关系注意力网络＋BCI/历史条件 → 结构化任务候选 → 补选或指令提议 → 受限英文模板 → 用户确认后交给原VLA。

| 文件 | 用途 |
|---|---|
| `scene_context/schema.py` | 公开输入与监督标注验证、场景划分防泄漏 |
| `scene_context/model.py` | 2层4头关系注意力、条件查询、联合候选评分、状态和补选头 |
| `scene_context/planning.py` | 有效任务语法、缺失信息判断、公共约束校验；保留原始提议 |
| `scene_context/rendering.py` | 将结构化任务转为有限任务英文，处理同类实例指代 |
| `scene_context/integration.py` | labels/boxes适配、一次性提议确认、取消和场景变化检查；无机器人调用 |
| `scene_context/training.py` | 已预留训练、验证集温度/阈值和权重保存；尚未执行 |
| `scene_context/baselines.py`、`llm.py`、`evaluation.py` | 预留规则/Gemini比较与评估；云调用默认关闭 |

依赖：Python≥3.10。输入、模板、规则及基础测试只需标准库；神经网络需要PyTorch≥2.2和NumPy。没有自动安装步骤，也不会在导入时连接任何服务。

从本目录检查代码：

```powershell
python -m unittest discover -v
python -m compileall -q scene_context
```

没有torch时模型单元测试会明确跳过。此检查是代码正确性验证，不是模型性能实验。

训练/比较命令已预留在`python -m scene_context.cli --help`，本阶段不运行。合成fixture及现有规则smoke报告只用于开发，不是论文证据。实验与标注文档保留为待执行草案：`docs/EXPERIMENTS_ZH.md`、`docs/ANNOTATION_ZH.md`。
