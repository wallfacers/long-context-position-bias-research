# 论文索引

更新时间：2026-08-29。状态以论文 PDF 和公开会议信息为准；“预印本”不等于结果无效，但结论需要更谨慎复现。

## 核心现象与评测

| 年份 | 论文 | 状态 | 本项目用途 |
|---|---|---|---|
| 2023/2024 | [Lost in the Middle](https://www.alphaxiv.org/abs/2307.03172) | TACL 2024 | 位置控制基线、U 形定义 |
| 2023/2024 | [LongBench](https://www.alphaxiv.org/abs/2308.14508) | ACL 2024 | 真实多任务长上下文评测 |
| 2021 | [MMLU](https://www.alphaxiv.org/abs/2009.03300) | ICLR 2021 | 57 学科知识与推理回归 |
| 2023 | [IFEval](https://www.alphaxiv.org/abs/2311.07911) | arXiv 预印本 | 541 prompts 的可验证指令遵循评测 |
| 2024 | [RULER](https://www.alphaxiv.org/abs/2404.06654) | 公开论文 | 有效上下文长度、可配置合成任务 |
| 2025 | [NoLiMa](https://www.alphaxiv.org/abs/2502.05167) | ICML 2025 | 无字面匹配语义检索 |
| 2025 | [Positional Biases Shift as Inputs Approach Context Window Limits](https://www.alphaxiv.org/abs/2508.07479) | COLM 2025 | 相对窗口长度与曲线形态 |
| 2026 | [Self-Consistency Falls Short](https://www.alphaxiv.org/abs/2411.01101) | TACL 2026 | Self-Consistency 与系统位置误差 |
| 2026 | [Attention Basin](https://www.alphaxiv.org/abs/2508.05128) | ACL 2026 | 结构边界、块级注意力盆地 |
| 2026 | [Positional Failures in Long-Context LLMs](https://www.alphaxiv.org/abs/2605.23170) | 预印本 | 2026 API 模型、filler × 位置 × 长度 |

## 机制研究

| 年份 | 论文 | 状态 | 本项目用途 |
|---|---|---|---|
| 2023 | [Attention Sinks](https://www.alphaxiv.org/abs/2309.17453) | ICLR 2024 | 开头 Token 注意力吸附与流式稳定性 |
| 2024/2025 | [Positional hidden-channel intervention](https://www.alphaxiv.org/abs/2406.02536) | Findings ACL 2025 | 隐藏状态位置通道与训练免更新干预 |
| 2026 | [Lost at Birth](https://www.alphaxiv.org/abs/2603.10123) | 预印本 | 初始化时的结构性 U 形假设 |
| 2026 | [Shortcut Before Circuit](https://www.alphaxiv.org/abs/2608.24460) | 预印本 | 位置/重复共线时的机制不可识别性与跨 seed 复制边界 |

## 训练和数据方法

| 年份 | 论文 | 状态 | 本项目用途 |
|---|---|---|---|
| 2024 | [LongAlign](https://www.alphaxiv.org/abs/2401.18058) | Findings EMNLP 2024 | 长指令数据与训练效率基线 |
| 2024 | [Data Engineering for Scaling Language Models to 128K Context](https://www.alphaxiv.org/abs/2402.10171) | ICML 2024 | 持续预训练与长度上采样基线 |
| 2024 | [IN2 / FILM](https://www.alphaxiv.org/abs/2404.16811) | 公开论文 | 细粒度位置与多片段整合训练 |
| 2025 | [Pos2Distill](https://www.alphaxiv.org/abs/2508.15709) | EMNLP 2025 | 优势位置到弱势位置的蒸馏 |
| 2026 | [Shuffle the Context](https://www.alphaxiv.org/abs/2604.14339) | 预印本 | RoPE 索引扰动双视图与自蒸馏 |
| 2026 | [FocuSFT](https://www.alphaxiv.org/abs/2605.09932) | 预印本 | 双层 SFT 和上下文表示强化 |
| 2026 | [LongCrafter](https://www.alphaxiv.org/abs/2607.06160) | 预印本 | 证据图、引用和难度均衡数据合成 |

## 推理期和位置编码方法

| 年份 | 论文 | 状态 | 本项目用途 |
|---|---|---|---|
| 2024 | [Found in the Middle](https://www.alphaxiv.org/abs/2406.16008) | Findings ACL 2024 | 注意力偏差校准 |
| 2026 | [LPES](https://www.alphaxiv.org/abs/2606.27705) | 预印本 | 逐层 RoPE scaling 搜索 |
| 2026 | [Randomized YaRN](https://www.alphaxiv.org/abs/2606.23687) | 预印本 | 训练期随机位置索引与长度课程 |

## 阅读优先级

### 第一优先级：建立问题定义

1. Lost in the Middle
2. NoLiMa
3. Positional Biases Shift as Inputs Approach Context Window Limits
4. Positional Failures in Long-Context LLMs

### 第二优先级：设计训练实验

1. IN2 / FILM
2. Pos2Distill
3. Shuffle the Context
4. Randomized YaRN
5. LongCrafter
6. FocuSFT

### 第三优先级：机制与免训练修复

1. Attention Basin
2. Attention Sinks
3. Found in the Middle
4. LPES
5. Lost at Birth
6. Shortcut Before Circuit
