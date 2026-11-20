# AI Evaluation

**面向 AI 模型、智能体与自主研究系统的基准设计、可复现评测和统计比较环境。**

[English](README.md) · [开展一次研究](docs/workflows.md) · [评测方法](docs/research.md) · [接入 Agent](docs/connect.md)

一次评测需要把研究问题、任务总体、系统版本、执行协议和观测结果联系起来。本环境使这些关系明确可计算，支持研究模型、Harness、数据、工具或推理预算的变化究竟带来了什么。

研究范围包括语言与多模态模型、检索、代码生成、交互智能体、科学预测和自主科研。共享运行时处理结构化任务与观测，具体领域的执行和测量由相应引擎提供。Benchmark 是有明确任务总体与评分规则、可以持续修订的实验工具。

## 运行完整示例

需要 Python 3.11 或更新版本。核心仅依赖标准库，MCP 按需安装。

```bash
git clone https://github.com/EnvFoundry/AI-Evaluation.git
cd AI-Evaluation
python -m pip install -e '.[mcp]'
python examples/run_examples.py --output runs/example
```

示例比较两组人工编写的检索结果，并通过子进程执行数值参数恢复，生成逐项记录、统计比较和独立 HTML 报告。示例用于说明流程，不代表真实模型的性能。

```bash
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/baseline.jsonl \
  --system-name baseline --system-revision 1 --output runs/baseline
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/candidate.jsonl \
  --system-name candidate --system-revision 1 --output runs/candidate
ai-eval compare runs/baseline runs/candidate --metric recall --slice-by variant
ai-eval report runs/baseline --against runs/candidate \
  --metric recall --output runs/comparison.html
```

## 已实现的能力

| 层次 | 能力 |
| --- | --- |
| 研究定义 | Benchmark 与系统版本标识；样本、独立任务组、分层、重复次数、指标方向与权重 |
| 执行 | Python 函数、JSON 子进程、预测结果回放、已安装的目标插件；本地并发与断点续跑 |
| 评分 | 精确匹配、数值容差、集合与 Token F1、Recall@k、nDCG@k、Brier、Log loss；评分插件 |
| 比较 | 加权样本均值、按独立任务组配对 Bootstrap、完整样本策略、分层分析、二值 Pass@k、覆盖率与资源记录 |
| 评审校准 | 与参考标注的一致率、混淆矩阵、分类别 Precision/Recall、Cohen's kappa、分组不确定性和缺失标注覆盖率 |
| 外部结果 | Gym export-v1 与 Inspect JSON-v2 导入，保留原始文件、评分定义和未获得结果的情况 |
| 接口 | Python、CLI、MCP、仓库 Skill 与 Agent 插件清单；简短能力发现与分页样本读取 |
| 报告 | 可筛选、排序、下载原始分析 JSON 的独立 HTML，直接使用同一套统计结果 |

[接口约定](docs/contracts.md) 定义数据含义，[评分说明](docs/scoring.md) 给出指标的精确定义，[生态关系](docs/ecosystem.md) 区分现有接入、可复用上游与待实现扩展。

运行 `ai-eval calibrate-judge examples/judgments.jsonl` 可查看人工构造的标注比较示例。[评审指南](docs/judges.md) 解释结果含义与参考标注选择。[Gym 组合示例](docs/gym-integration.md) 实际执行任务、导入结果、比较策略，并可进一步生成 Data Engineering 数据版本。

## 科学问题决定评测设计

先明确改变了什么、固定了什么：模型检查点、提示词、检索、记忆、工具、环境、采样方式与资源预算。在收集重复尝试之前确定独立统计单位。执行失败和部分覆盖保留在结果中，避免在平均值里消失。

[研究方法](docs/research.md) 覆盖构念与真实场景的对应、分组不确定性、自动评审校准、数据污染、任务分布演化、自适应复用、迁移、质量—成本关系及长程科学成果。[架构设计](docs/design.md) 说明静态任务、交互轨迹与总体级指标如何协作。

进入新领域时，先写清楚科学主张、可观测结果、分析单位和对照协议，再选择执行引擎。领域系统已经提供合适的实验世界和测量方法时，直接复用；接入层负责保留它们的含义。

## 接入模型与 Agent

```bash
ai-eval capabilities
ai-eval connect --root /absolute/path/to/studies --output mcp.json
```

配置可交给支持 MCP 的宿主。函数或命令目标可以封装本地模型、云 API、检索系统或完整 Agent Harness；它接收任务输入与随机种子，参考答案留在评分侧。凭据与模型服务沿用目标系统自己的运行方式。

[连接指南](docs/connect.md) 包含命令 JSON、插件入口、MCP 和源码启动方式。[评测 Skill](skills/ai-evaluation/SKILL.md) 按任务引导 Agent 阅读必要方法与执行操作，避免一次加载整个仓库。

## 与其他研究环境协作

| 环境 | 职责 |
| --- | --- |
| [Data Engineering](https://github.com/EnvFoundry/Data-Engineering) | 数据构建、变换、来源与采样 |
| [Verifiable Agent Gym](https://github.com/EnvFoundry/Verifiable-Agent-Gym) | 可执行任务、交互环境、轨迹与验证结果 |
| **AI Evaluation** | 基准设计、测量、实验比较与解释 |
| [IOP](https://github.com/EnvFoundry/IOP) / [Optics-Photonics-AI-Lab](https://github.com/EnvFoundry/Optics-Photonics-AI-Lab) | 物理仪器操作与光学科研流程 |

## 开发

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python examples/run_examples.py
python -m build
```

贡献应带来有用的测量、可执行接入、研究方法或更清晰的实验。参见 [贡献说明](CONTRIBUTING.md)。当前执行引擎运行在本地；分布式调度、GPU 服务、领域模拟器和基准数据集由各自系统提供。

MIT 许可证。
