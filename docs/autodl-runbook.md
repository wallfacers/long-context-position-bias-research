# AutoDL 上机即跑手册

> **历史 pilot 手册，禁止直接用于当前确认性队列。** 本页保留最初四条件、2,000-step continuation 的操作记录以便审计。2026-08-28 后的论文级实验已经冻结为六条件 matched 设计、每单元在原 2,000-step scheduler horizon 上由 callback 精确停在 step 100；当前唯一入口、预算、数据矩阵和恢复规则以[顶会级完整性协议](top-tier-completion-protocol.md)及 `scripts/run_autodl_full_completion_queue.sh` 为准。不要按本页的“续跑到 2,000 steps”命令修改已冻结 adapter。

## 已经在本地完成的工作

`data/pilot_qwen25_7b/` 当前包含：

| 数据 | 数量 | 用途 |
|---|---:|---|
| raw train | 5,000 | 位置等价组母数据 |
| raw dev | 700 | 8K 开发集 |
| raw test | 4,200 | 2 个任务 × 3 个 filler × 2 个长度 × 7 个位置 × 50 组 |
| 四种 SFT JSONL | 每种 1,000 | 可读、可复现的训练源数据 |
| 四种 Arrow 预分词数据 | 每种 1,000 | AutoDL 直接训练 |

Arrow 数据已经用 `Qwen/Qwen2.5-7B-Instruct` 的 chat template 生成 completion mask。最长完整训练序列为 8,257 token，低于 8,320 上限；训练只计算 assistant completion loss。预检会同时核对源 JSONL SHA-256 和 tokenizer/chat-template 指纹。

## 租机器之前

### 1. 准备基座模型

训练和评测脚本禁止在线模型 ID，只接受完整的本地目录。可以在非计费机器上下载：

```bash
python3 scripts/stage_model.py \
  --output /path/to/Qwen2.5-7B-Instruct
```

脚本固定下载 revision `a09a35458c702b33eeacc393d103063234e8bc28`，并为模型文件生成 SHA-256 manifest。模型目录、实验 bundle 和环境应预先放在可持久化数据盘。不要租到 GPU 后才下载约 15 GB 权重。

国内节点如果连接 Hugging Face 官方端点超时，可以保持同一 revision，切换镜像后断点续传：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python3 scripts/stage_model.py \
  --output /path/to/Qwen2.5-7B-Instruct
```

遇到单个大分片的 CDN 单连接限速时，可使用 `scripts/parallel_range_download.py` 对可信下载 URL 做并行 Range 断点续传。必须同时传入上游 manifest 中的精确字节数和 SHA-256；脚本只会在最终长度与哈希都匹配后生成正式文件。

如果模型目录已经由其他方式准备，补建 manifest：

```bash
python3 scripts/stage_model.py \
  --output /path/to/Qwen2.5-7B-Instruct \
  --manifest-only
```

### 2. 生成上传 bundle

只上传训练数据：

```bash
bash scripts/package_autodl_bundle.sh \
  --output /path/to/autodl-train.tar.gz
```

评测机器还需要 dev/test：

```bash
bash scripts/package_autodl_bundle.sh \
  --include-test \
  --output /path/to/autodl-eval.tar.gz
```

上传后先运行：

```bash
sha256sum -c autodl-eval.tar.gz.sha256
```

### 3. 环境分开保存

训练环境先使用 AutoDL 中与 GPU 匹配的 PyTorch/CUDA 镜像，再安装：

```bash
python3 -m pip install -r requirements-train.txt
```

评测环境单独安装：

```bash
python3 -m pip install -r requirements-eval.txt
```

不要把 vLLM 强行装进训练环境；它可能替换 PyTorch、Transformers 或 CUDA 依赖。最好把两个可工作的环境做成可复用镜像或保存在持久盘。第一次环境准备所需时间已经计入预算中的固定开销。

训练入口使用 TRL 的 prompt-completion 与 completion-only loss；评测入口使用 vLLM 的 JSON structured outputs 和离线 LoRA 请求。对应接口以 [TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer)、[vLLM Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) 和 [vLLM LoRA](https://docs.vllm.ai/en/stable/features/lora/) 为准。

## 训练顺序

先只租一张 RTX 5090 32 GB，跑一个 100-step canary：

```bash
bash scripts/run_sft_variant.sh \
  --model /persistent/models/Qwen2.5-7B-Instruct \
  --variant paired_evidence \
  --canary
```

脚本按顺序执行：本地模型完整性检查、数据 hash、tokenizer 指纹、CUDA/bfloat16、显存和磁盘检查，然后在完整的 2,000-step 学习率计划中停在 step 100。它会生成 `checkpoint-100/`、`CANARY_COMPLETE.json` 和 `READY_TO_STOP_AUTODL_AFTER_CANARY`。

用真实吞吐重算全项目预算：

```bash
python3 scripts/estimate_autodl_budget.py \
  --train-canary outputs/paired_evidence/CANARY_COMPLETE.json
```

如果成本在上限内，从 checkpoint-100 原地续跑：

```bash
bash scripts/run_sft_variant.sh \
  --model /persistent/models/Qwen2.5-7B-Instruct \
  --variant paired_evidence
```

单机顺序完成四个变体、校验并打包结果；脚本完成后保留实例，等待主代理验收：

```bash
screen -dmS train-all bash -lc '
  source /persistent/venvs/train/bin/activate &&
  cd /persistent/position-bias-pilot &&
  bash scripts/run_autodl_training_queue.sh \
    --model /persistent/models/Qwen2.5-7B-Instruct \
    >> /persistent/train-all.log 2>&1
'
```

队列会从每个变体的最新有效 checkpoint 自动恢复。四个 `TRAINING_COMPLETE.json` 和 adapter 全部存在、最终结果包 SHA-256 复核通过后，写入 `RESULTS_READY_FOR_AGENT_REVIEW`，但绝不由脚本关机。任一训练或打包步骤失败时同样保留实例和非零状态，便于主代理检查日志。

最终打包前还会执行论文级审计：要求四个变体各有从 step 1 到 2,000 的完整逐步记录，导出 completion-only loss、gradient norm、learning rate、entropy、token accuracy 与累计 token 数，并保存原始 `trainer_state`。产物位于 `outputs/training_diagnostics/`：

```text
training_metrics.csv/jsonl        # 四个变体的可移植逐步数据
training_metrics_summary.json     # 完整性检查和过拟合预警
hardware_telemetry.csv            # 30 秒 GPU 利用率、显存、功耗和温度采样
training_loss_health.jsonl        # checkpoint 级 Loss/梯度/学习率健康判定
train_queue.log                   # 恢复、队列边界和完整训练日志快照
raw_trainer_state/                # 导出来源，供第三方交叉验证
figures/training_curves.svg       # 论文可编辑矢量图
figures/training_curves.png       # 300 DPI 预览图
reproducibility.json              # GPU、驱动、包版本、参数和代码/数据/模型哈希
publication_artifacts.sha256      # 包内非 checkpoint 文件校验清单
```

结果压缩包还会冻结当次实际执行的 `scripts/`、`src/`、配置、文档、测试、依赖锁定文件、数据 manifest 和四份预分词 metadata；大体积训练数据本体按 manifest 单独发布。这样即使工作目录没有 Git commit，第三方仍能核对实际执行代码，而不是依赖后来变化的分支状态。

曲线中的淡线是逐步原始值，实线是 span=50 的 EMA；Loss 和 gradient norm 使用对数轴。训练 Loss 只计算 assistant completion，属于优化诊断，不能替代未见 dev/test 上的位置等价组评测。answer-only 与 evidence-supervised 的 completion 目标空间不同，二者的绝对训练 Loss 不作横向效果比较。若最后 100 步 Loss 中位数低于 `1e-4` 或 token accuracy 高于 `0.999`，摘要会明确标记需要检查过拟合。

第一个变体完成并验证 adapter 可加载后，再启动 `independent_answer`、`paired_answer`、`independent_evidence`。三台并行可缩短墙钟时间，但总 GPU 小时和费用不会下降；每台机器只运行一个 variant。

## 评测顺序

评测用一张 RTX 4090 24 GB。一次加载基座，然后连续评测 base 和四个 adapter。先跑总计 100 条请求（五个 run 各 20 条、均匀覆盖条件）的 canary：

```bash
bash scripts/run_eval_suite.sh \
  --model /persistent/models/Qwen2.5-7B-Instruct \
  --run base \
  --run independent_answer=/persistent/outputs/independent_answer/final_adapter \
  --run paired_answer=/persistent/outputs/paired_answer/final_adapter \
  --run independent_evidence=/persistent/outputs/independent_evidence/final_adapter \
  --run paired_evidence=/persistent/outputs/paired_evidence/final_adapter \
  --canary
```

读取实测秒数并重算预算：

```bash
python3 scripts/estimate_autodl_budget.py \
  --train-canary outputs/paired_evidence/CANARY_COMPLETE.json \
  --eval-canary results/pilot/base.jsonl.run.json
```

确认后用完全相同的命令去掉 `--canary`。评测按 batch `fsync`，已有 `sample_id` 会跳过。最后聚合：

```bash
python3 scripts/aggregate_results.py \
  results/pilot/base.jsonl \
  results/pilot/independent_answer.jsonl \
  results/pilot/paired_answer.jsonl \
  results/pilot/independent_evidence.jsonl \
  results/pilot/paired_evidence.jsonl \
  --output results/pilot/summary.json
```

## 关机规则与预算

在本项目使用的 AutoDL 容器中，平台提供的 `/usr/bin/shutdown` 是停止实例的包装命令；不要添加普通 Linux `shutdown` 的参数，也不要用 `--help` 探测它。训练和评测脚本都不得调用关机命令。主代理必须先验收进程状态、结果完整性、聚合报告、结果包和 SHA-256，向用户汇报后，再把关机作为独立收尾任务显式执行。若更换云厂商或基础镜像，必须先按该平台文档重新确认关机与计费语义。

本次 RTX 5090 D 32 GB 的实测租价为 ¥2.78/小时：100-step canary 为 2.778 秒/step，四个 2,000-step 训练约 ¥20.22；把保守评测计入后，当前确定性 pilot 预期约 ¥85.71，15% 上限约 ¥98.57，因此先准备 ¥100 即可。租赁页价格变化时，用 `--rate rtx_5090_32gb=实际单价` 和 `--rate rtx_4090_24gb=实际单价` 覆盖配置。
