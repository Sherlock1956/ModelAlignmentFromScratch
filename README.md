# ModelAlignmentFromScratch

在[前面的项目](https://github.com/Sherlock1956/TransformerFromScratch)中我们已经实现了从0手写一个大模型并完成了模型的预训练，但是大模型在预训练之后，模型往往无法与人类偏好对齐。而一般的对齐方式有SFT，RL等方法。在本项目中，我们将介绍如何从0手写不同的模型对齐方法，包括监督微调(SFT)、专家迭代(EI)和组相对策略优化(GRPO)，并在一个千问系列的一个小模型上使用gsm8k进行对齐训练测试，该项目专注于将语言模型与人类偏好对齐，特别是在数学推理任务方面。

通过本项目你会学习到:
- 如何手写SFT，EI，RPO的损失函数，如何手写以上方法的训练流程的所有细节。
- 强化学习中的策略梯度方法如何应用在大模型微调中，不同的策略梯度方法的特点，如何选择和改进。
- 如何使用vllm和训练进行结合，使用vllm高效进行数据采样并用于模型训练。
- ...

同时，本项目探究了在小模型上应用GRPO会存在的一些问题，并尝试用不同的方法进行改进，提出一些自己的思考。

**注意**：请充分结合[本人的作业完成笔记](作业完成笔记.md)和[官方的作业指导书](cs336_spring2025_assignment5_alignment.pdf)使用本项目。

如果你觉得本项目对你有帮助，欢迎给出你的Github Star🌟，也欢迎对本人的代码批评指正，提出Github Issue / Github PR : )

## 项目概述

本项目的目标是实现和实验不同的对齐技术用于大型语言模型。它包括以下实现：

1. **监督微调(SFT)**：传统的监督学习方法，在输入-输出对上训练模型。
2. **专家迭代(EI)**：一种通过过滤高质量响应并重新训练来迭代改进策略的方法。
3. **组相对策略优化(GRPO)**：一种基于偏好的对齐方法，基于响应组内的相对奖励优化策略。

## 项目结构

```
.
├── cs336_alignment/          # 主要实现目录
│   ├── GRPO.py               # GRPO算法实现
│   ├── config.py             # 配置参数
│   ├── download_models.py    # 模型下载工具
│   ├── drgrpo_grader.py      # 用于评估响应的奖励函数
│   ├── expert_iteration.py   # 专家迭代实现
│   ├── math_baseline.py      # 基线评估方法
│   ├── sft.py                # 监督微调实现
│   └── utils.py              # 各种算法的工具函数
├── scripts/                  # 评估和部署脚本
├── tests/                    # 实现的单元测试
└── data/                     # 数据集目录
    └── gsm8k/                # GSM8K数学推理数据集
```

## 主要组件

### 监督微调(SFT)
[sft.py](cs336_alignment/sft.py) 文件实现了传统的监督微调方法，其中模型在来自GSM8K数据集的输入-输出对上进行训练。这作为与更高级对齐方法比较的基线。

### 专家迭代(EI)
[expert_iteration.py](cs336_alignment/expert_iteration.py) 文件实现了专家迭代算法，该算法：
- 使用当前策略生成响应
- 基于奖励函数过滤高质量响应
- 在这些高质量示例上重新训练策略
- 通过此过程迭代改进策略

### 组相对策略优化(GRPO)
[GRPO.py](cs336_alignment/GRPO.py) 文件实现了GRPO算法，该算法基于生成响应组内的相对奖励优化策略。该方法：
- 为每个提示生成多个响应
- 计算每个响应的奖励
- 在每组内标准化奖励
- 使用策略梯度方法优化策略


### 工具函数
[utils.py](cs336_alignment/utils.py) 文件包含在各种实现中使用的各种工具函数，包括：
- 分词和数据处理函数
- 损失计算函数
- 梯度裁剪和学习率调度
- 奖励标准化函数
- ...

### 奖励函数
[drgrpo_grader.py](cs336_alignment/drgrpo_grader.py) 文件包含用于评估模型响应的奖励函数，特别是针对数学推理任务。这些函数评估响应的格式和正确性。

## 快速开始

1. 使用`uv`安装依赖：
   ```bash
   uv sync --no-install-package flash-attn
   uv sync
   ```

2. 下载所需模型（如需要）：
   ```bash
   python cs336_alignment/download_models.py
   ```

3. 运行实现：
   ```bash
   python cs336_alignment/sft.py
   python cs336_alignment/GRPO.py
   python cs336_alignment/expert_iteration.py
   ```

## 测试

使用pytest运行测试：
```bash
uv run pytest
```

## 主要特性

- 多种对齐算法的实现（SFT、GRPO、专家迭代）
- 使用GSM8K数据集支持数学推理任务
- 全面的评估框架
- 模块化设计，便于扩展和实验
- 与vLLM集成以实现高效推理
- TensorBoard日志记录以监控训练进度

## 依赖项

主要依赖包括：
- PyTorch
- Transformers
- vLLM
- TensorBoard/TensorBoardX
- 各种工具库（tqdm等）

请参阅[pyproject.toml](/pyproject.toml)获取完整依赖列表。