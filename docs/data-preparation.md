# 数据准备与 API 使用边界

## 首轮结论

阶段 0 和第一轮监督训练不依赖云厂商大模型 API。KV 检索、聚集式两跳、位置排列、答案、证据 ID 和精确引用都由确定性规则生成。这样可以保证：

- 同一位置等价组只改变证据位置；
- 答案和引用可以由程序完全验证；
- train/dev/test 按原始事实隔离；
- 不产生 API 费用，也不会把 teacher 的位置偏差直接写进标签；
- 数据能够从种子和配置重新生成。

单 seed pilot 使用四个等输入 Token 预算变体；正式 matched 实验扩展为六个变体：

| Variant | 原始事实组织 | 监督内容 |
|---|---|---|
| `independent_answer` | 每个事实采一个均衡位置 | 答案；证据字段为空 |
| `paired_answer` | 同一事实的四个位置 | 答案；证据字段为空 |
| `independent_evidence_id` | 每个事实固定在一个均衡位置，使用四份 filler 视图 | 答案、证据 ID；引用为空 |
| `paired_evidence_id` | 同一事实依次出现在四个位置，使用相同四份 filler 视图 | 答案、证据 ID；引用为空 |
| `independent_evidence` | 每个事实采一个均衡位置 | 答案、证据 ID、精确短引用 |
| `paired_evidence` | 同一事实的四个位置 | 答案、证据 ID、精确短引用 |

`paired` 样本保留 `group_id` 和位置元数据，后续用于跨位置 KL；普通 SFT 阶段不会把位置标签写入 prompt。

正式设计不再比较“250 个事实各四位置”和“1,000 个事实各一位置”。对于每个 seed，它先生成相同的事实集合；每个事实有四份独立 filler 视图、每份视图都有四个位置版本。`paired` 为同一事实的四份 filler 视图分别选择四个不同位置；`independent` 让同一事实的四份视图固定在一个位置，并在不同事实之间均衡位置。这样两组具有完全相同的事实、曝光次数、filler 指纹和近似输入 Token 预算，改变的只有“同一事实是否跨位置配对”。

## 本地生成

数据生成只需要 CPU。安装 tokenizer 依赖：

```bash
python3 -m pip install -e ".[data]"
```

一次生成并校验 train/dev/test、四个 SFT 变体和上传校验清单：

```bash
bash scripts/prepare_pilot_data.sh
```

默认使用 `Qwen/Qwen2.5-7B-Instruct` tokenizer，输出位于 `data/pilot_qwen25_7b/`。最终数据必须使用目标模型 tokenizer；`--tokenizer whitespace` 只允许用于单元测试和 smoke run。

CPU 预分词也已经准备为 Arrow 数据：

```bash
python3 scripts/pretokenize_sft.py \
  data/pilot_qwen25_7b/sft/paired_evidence.jsonl \
  data/pilot_qwen25_7b/tokenized/paired_evidence \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-files-only
```

四份正式预分词数据合计 32,645,447 token；answer-only completion 为 33～39 token（证据数组为空），evidence completion 为 63～121 token。最长完整序列 8,257 token，未发生截断。

正式配置为：

- train：8K，KV/两跳，neutral filler，四个训练位置；
- dev：8K，neutral filler，七个测试位置；
- test：8K/32K，三种 filler，七个位置；
- 每个测试条件 50 个位置等价组；
- 四个 SFT 变体各 1,000 条，输入 Token 预算误差不超过 2%。

生成结束后，`manifest.json` 保存每个 JSONL 的行数、字节数和 SHA-256。上传 AutoDL 后应重新计算校验和再开始计费训练。

## 正式 matched 数据

以下命令在本地 CPU 一次生成三个 seed、六个 SFT 条件、Arrow 预分词数据、设计审计和机制诊断集：

```bash
bash scripts/prepare_formal_matched_data.sh
```

默认输出为 `data/formal_matched_qwen25_7b/seed_{20260825,20260826,20260827}/`。每个 seed 包含 256 个唯一事实（KV 和两跳各 128 个）、每事实四次曝光、每个 SFT 条件 1,024 行。`matched-audit.json` 必须证明：

- 六个条件使用相同的 fact × replica 单元；
- paired 与 independent 使用相同 filler 指纹及事实指纹；
- 每个事实都是四次曝光；
- paired 的同一事实覆盖四个位置，independent 的同一事实只在一个位置；
- 两种 pairing 的输入 Token 总量差不超过 0.2%。

脚本还从未参与训练的 4,200 行 pilot test 派生 9,000 行诊断数据：4,200 行 locate-only、4,200 行 oracle-long、600 行去重后的 oracle-short。oracle-long 会把正确证据从原上下文移动到末尾 oracle 区，而不是复制一遍，因此不会无意义地挤爆 32K 窗口。

## NoLiMa 固定源数据

NoLiMa 官方源数据必须从固定版本确定性恢复，不能直接依赖可变的 `main` URL，也不能把已经用 Qwen tokenizer 生成的 JSONL 复制给第二模型家族：

```bash
git clone https://github.com/adobe-research/NoLiMa.git third_party/NoLiMa
git -C third_party/NoLiMa checkout cb14780b249fecf2851127b2101a062c1b2c6430
python3 scripts/fetch_nolima_sources.py --output-dir third_party/NoLiMa/data
```

获取器固定 Hugging Face dataset revision `378115b1f136b6ba78f90f78682bc55f70ec3ddd`，显式复现官方脚本的同名 `wget -c` 续传语义，并对 hard needle、normal/long 原文件和最终五本 book 全部做 SHA-256 门禁。Qwen 与 Mistral 随后从同一组源字节分别渲染和分词。

## 什么时候使用 DeepSeek 等 API

API 只在第二阶段自然语义数据中有价值，适合承担：

1. 把确定性事实改写为无字面重合问题；
2. 从有许可的自然文档中提出候选证据图；
3. 生成多种问题表述和干扰文档；
4. 为复杂推理生成候选解释，供 verifier 筛选。

API 不应直接决定最终答案或证据。推荐流程是：

```text
本地文档与许可检查
  -> 本地抽取候选证据
  -> API 生成问题/改写/候选证据关系
  -> 精确引用存在性检查
  -> 答案唯一性与反事实检查
  -> 去重和 train/test 污染检查
  -> 最后才生成位置等价组
```

每次 API 调用必须记录 provider、model identifier、调用日期、采样参数、prompt hash、response id 和 token usage。API 生成失败或 verifier 不通过的样本直接丢弃，不能人工静默修改后混入确定性数据。

不要向第三方 API 发送私有、敏感或授权不明确的文档。API 合成数据单独保存在新的数据版本中，不与本轮规则数据覆盖写入。

按 2026-08-27 用户提供的 DeepSeek V4 报价，若进入自然语义阶段，优先用 `deepseek-v4-flash` 批量产生候选，`deepseek-v4-pro` 只处理 verifier 难例。按 10,000 次候选、每次 4K 未缓存输入和 300 输出估算，flash 在 off-peak/peak 约为 $10.78/$21.56，pro 约为 $32.34/$64.68；这只是 API token 费，不含低通过率导致的重试。必须先测 100 条候选的通过率，再决定是否扩量，并在真正购买前再次核对厂商计费页。

## 成本控制

不要把完整 32K 文档逐条发送给 teacher。自然数据阶段先在本地检索和分块，只向 API 提交支持候选和少量困难干扰。当前正式 matched 因果矩阵不需要 teacher API；API 预算始终与 GPU 实验预算分列，GPU 的最新前瞻余额以[顶会级完整性协议](top-tier-completion-protocol.md)中的机器可复算快照为准。
