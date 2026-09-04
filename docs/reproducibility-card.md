# Reproducibility card

Frozen: 2026-08-28. This card describes the evidence needed to reproduce paper claims, not merely rerun a training command.

Implementation amendment: the 2026-08-29 realized-sampler audit demoted fixed-100 runs to historical sensitivity evidence because their consumed subsets were not block-complete. Primary claims require the strict 96-row/96-step materialized subsets, checkpoint-96, and a passing realized-subset audit. Qwen is labeled corrective because partial Qwen OOD results were already visible; Mistral is prospective under the corrected protocol.

## Claim-producing layers

| Layer | Unit | Required artifact | Integrity rule |
|---|---|---|---|
| Data | row and position-equivalent group | JSONL, generation audit, manifest | pinned source/revision, tokenizer fingerprint, SHA-256 |
| Training | independently trained seed | primary checkpoint-96, trainer state, run config, step metrics, realized-subset audit | exactly one 96-row block-complete pass; no test-set checkpoint selection; historical checkpoint-100 roots separately labeled |
| Generation | model condition × sample | resumable JSONL, `.run.json`, prompt-length audit | identical selected IDs/data hash and adapter hash; every rendered prompt plus frozen output cap fits the model window |
| Within-seed inference | rule matched group, NoLiMa semantic case, or LongBench question | analysis JSON/CSV and 5,000 bootstrap indices | pairing and preregistered stratification preserved; NoLiMa's 10 cases are clusters |
| Across-seed inference | training/data seed | seed-level JSON/CSV with `corrective` or `confirmatory` source status | pilot excluded; Qwen correction not called blindly preregistered; prediction rows never treated as training replicates |
| Qualitative audit | deterministic failure category | catalog JSON/CSV/Markdown plus manifest | all candidates counted; at most five examples/category selected by fixed sort; raw text and identifiers excluded |
| Paper | generated macro/table/figure | TeX/PDF plus generation manifest | source-analysis and output SHA-256 |
| Release | sanitized artifact tree | sanitization manifest | original and public-copy hashes both recorded; secrets fail closed |

## Frozen models

| Family | Model | Revision | Window | Training length |
|---|---|---|---:|---:|
| Qwen | Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 32,768 | 8,320 |
| Mistral | Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 32,768 | 8,320 |

Each local model snapshot gets a full SHA-256 manifest. A stat-revalidated attestation avoids rehashing 15–29GB for every condition while refusing reuse after any manifest, size, or nanosecond-mtime change. Each family must also pass a system/user/assistant preservation and completion-mask-prefix audit. Qwen uses its native protocol. Mistral v0.3 fails the native full-conversation prefix check because its training render drops the system turn, so both training and inference deterministically use the audited `merge-system-into-first-user-v1` transform. The selected protocol and audit hash are pinned with that family's model config and data manifests.

Before vLLM allocates the base model, each evaluation suite renders every selected prompt with the exact family chat protocol, tokenizes it locally, and writes `prompt-length-audit.json`. The gate fails unless `max_prompt_tokens + max_new_tokens <= max_model_len`; the audit records the data hash, selected row count, distribution summary, maximum sample ID, and remaining context headroom without copying prompt text.

Generation results retain safe input metadata (case/book/fact identifiers and hashes, never prompt text) for direct cluster lineage. The first Qwen NoLiMa generation predated that additive field, so its completed analysis performed an exact `sample_id` join against the frozen 1,050-row source, refused any missing/extra ID, recorded source SHA-256 `fafd30e7…c101`, and validated exactly 10 `metadata.case_id` clusters stratified 2/6/2 by task before all 5,000 replicates. Thus old and new result rows share the same fail-closed statistical gate without regenerating valid predictions.

The exploratory Qwen seed-1 formal process began before that strict pre-allocation gate was added. Its freeze step therefore performs and labels a **post-hoc** CPU lineage audit rather than silently claiming prospective enforcement: it stat-revalidates the pinned model attestation, re-renders all 4,200 unique prompts with the exact tokenizer/chat protocol, verifies the 32,581-token maximum plus 176 output tokens leaves 11 tokens, and binds the data/selection, historical run metadata, saved result hashes and generation-time adapter hashes. All subsequent suites use the prospective gate before model allocation.

The 2026-08-29 qualitative-analysis amendment was frozen after the exploratory Qwen diagnostics but before any strict block-96 Qwen or Mistral outcome. `analyze_failure_cases.py` exhaustively counts fixed row-, group-, and Base-versus-treatment patterns, records each category's scope-specific denominator and rate, sorts candidates deterministically, and retains at most five audit examples per category. Row errors use all generated rows as the denominator, position-pattern errors use eligible within-run groups, and Base-versus-treatment changes use matched comparisons; rates with different units are never pooled. It emits no prompt, target, generation, parsed answer, quote, benchmark answer, raw sample/group/case/book identifier, or absolute path. Private rows remain traceable through source-file and identifier SHA-256 fingerprints. These catalogs are descriptive audits, never additional hypothesis tests or a basis for selecting a winning treatment. The strict final queue freezes an explicit set of exactly 18 catalogs: Qwen three seeds across rule/NoLiMa/LongBench and Mistral three seeds across the same suites. `audit_failure_case_catalogs.py` rejects duplicates, ignores unrelated historical catalogs, and recomputes every selected output, rate, and source-JSONL row count/hash before the full evidence manifest can pass.

The exploratory Qwen rule and NoLiMa packages completed before this amendment, while its LongBench runner was already active and was deliberately not overwritten mid-execution. After all 4,200 LongBench rows and the original package gate completed, the same frozen catalog rules were applied. `retrofit_failure_catalog.sh` is the only permitted single-catalog legacy update path: it requires the suite-ready marker, validates the existing adjacent artifact checksum, performs the full source/hash/canonical-view audit, backs up and atomically updates `completion.json`, builds and inspects a replacement archive, then replaces the archive and checksum. `repack_failure_catalog_artifact.sh` provides the equivalent exact-manifest-set gate for completed multi-seed packages. A pre-artifact failure preserves the old package; an interruption after artifact replacement leaves a checksum mismatch and therefore fails closed. Active or partial suites are rejected. A later hardening pass found that the five already generated Qwen catalogs had a Python-list representation defect only in their CSV/Markdown views; the canonical safe JSON, counts, rates, source hashes, and reported statistics were unaffected. The new audit deliberately rejects all five until their views are re-rendered and their four completed packages are re-audited and replaced, so the earlier 3/18 aggregate pass is historical rather than final-release evidence. Future runners use the corrected renderer before their initial completion/package gate.

MMLU uses the same structured generation harness as the controlled suites, but valid JSON is not a knowledge metric. The first Qwen analysis revealed the confound directly: Base hit the 32-token cap on 11,738/14,042 rows and had 73.69% valid JSON, while evidence-supervised adapters were almost perfectly schema-valid. `format-robust-option-extraction-v1` therefore rescored the frozen generations with a narrow precedence rule---parsed JSON answer, truncated/embedded `answer` field, bare option, then one unambiguous explicit answer phrase---and retained JSON validity, finish reasons, extraction sources, and disagreement with the legacy score as diagnostics. Extraction succeeds on at least 99.97% of every Qwen run. The original format-constrained analysis remains in the artifact under `legacy_format_constrained_analysis`; `retrofit_mmlu_format_robust_analysis.sh` regenerates 5,000 paired subject-stratified bootstrap samples, records all seven source hashes, changes the completion pointer, validates the replacement archive, and updates its adjacent checksum. Future Qwen/Mistral MMLU runners use the robust score at their initial gate.

## Frozen public data revisions

| Dataset | Revision or immutable identity | Frozen output |
|---|---|---|
| LongBench | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` | 600 multi-document QA rows, SHA `5cd4b15b…e3fe` |
| MMLU | `c30699e8356da336a370243923dbaf21066bb9fe` | 14,042 test rows, SHA `a1d17096…19d1` |
| IFEval | `041338718b4e8151372fd63677104c65b73a0a4e` | 541 prompts/834 instructions, SHA `bb16cfea…61f` |
| NoLiMa-Hard | repo `cb14780…6430`, dataset `378115b1…3ddd`, hard needle plus five recorded book hashes | 1,050 positioned rows; each model family is retokenized separately |

NoLiMa remains under the Adobe Research License for noncommercial research. Release generation prefers scripts, source pointers, and hashes rather than relicensing benchmark text. LongBench constituent datasets retain their original terms.

The public release intentionally does not vendor benchmark text or third-party evaluator code. Reconstruct the pinned code snapshots under `third_party/` from their upstream repositories, then run the project preparation scripts, which fail unless the downloaded inputs match the frozen manifests:

```bash
git clone https://github.com/adobe-research/NoLiMa.git third_party/NoLiMa
git -C third_party/NoLiMa checkout cb14780b249fecf2851127b2101a062c1b2c6430
python3 scripts/fetch_nolima_sources.py --output-dir third_party/NoLiMa/data
git clone https://github.com/THUDM/LongBench.git third_party/LongBench
git -C third_party/LongBench checkout 2e00731f8d0bff23dc4325161044d0ed8af94c1e
git clone https://github.com/google-research/google-research.git third_party/google-research
git -C third_party/google-research checkout 041338718b4e8151372fd63677104c65b73a0a4e
```

NoLiMa's books and needle sets remain subject to its official download and license flow; their exact SHA-256 values are recorded in the NoLiMa manifest. The upstream `download_NoLiMa_data.sh` first downloads `rand_shuffle` and then uses `wget -c` on same-named `rand_shuffle_long` files. The frozen source bytes therefore consist of the normal-book prefix followed by the long-book bytes after that prefix length. `fetch_nolima_sources.py` makes this implicit continuation deterministic: it pins Hugging Face dataset revision `378115b1f136b6ba78f90f78682bc55f70ec3ddd`, verifies both raw inputs, reconstructs the five final files, requires the previously frozen Qwen SHA-256 values, and writes a machine-readable retrieval manifest. LongBench task files, the 541 official IFEval prompts, the MMLU revision, and the vendored NLTK `punkt_tab` resource are likewise accepted only when their recorded hashes match. The public package keeps checksum files and licenses but excludes the corresponding third-party payloads.

## Frozen training protocol

- Six cells per seed: independent/paired × answer/evidence-ID/exact-evidence.
- Historical Qwen fixed-100 seed 20260825 is exploratory and its post-pilot seeds are retained only as partial-dose diagnostics after the realized-sampler audit.
- Strict block-96 Qwen seeds 20260825/20260826/20260827 are all labeled corrective because the implementation repair followed partial Qwen OOD results; none is presented as blindly preregistered.
- Strict block-96 Mistral seeds 20260825/20260826/20260827 are prospective confirmatory replications under the corrected protocol.
- 4-bit NF4 double quantization, BF16 compute, LoRA rank 16, alpha 32, dropout 0.05, all linear layers.
- Batch 1, no packing, completion-only loss, paged AdamW 8-bit. The scheduler keeps the originally declared 2,000-step cosine horizon with peak learning rate `2e-4`; `warmup_ratio=0.03` therefore means 60 warmup steps. The strict callback stops every primary run at exactly step 96 after one complete materialized pass (60 warmup plus 36 post-warmup steps). Historical partial-dose runs stopped at step 100. Neither protocol has a three-step warmup, and their checkpoints cannot be mixed.
- Loss, learning rate, gradient norm, token accuracy, timing, package versions, GPU identity, and hashes are saved. Near-zero training loss is a health/overfitting warning, not a paper outcome.

The parent completion queue deliberately remains in the vLLM-capable evaluation environment. Each training queue validates and sources a separate training environment only inside its child process, including `datasets`, `trl`, `peft`, `bitsandbytes`, and `accelerate`; returning from training therefore leaves the parent environment ready for evaluation. This boundary was exercised by a fail-closed recovery on 2026-08-29: the historical Qwen handoff stopped before producing a usable adapter when the evaluation environment lacked training dependencies, the explicit train-environment gate was added and tested, and the queue resumed without accepting a partial adapter. The strict queue retains the same isolation but requires checkpoint-96, a complete 96-row trace, canary, archive, adjacent checksum, and realized-subset audit for every primary cell.

The first strict handoff also stopped before GPU training when a legacy Qwen pretokenization record encoded its implicit native chat protocol as JSON `null`. The compatibility gate now normalizes only missing/null legacy metadata to `native-system-user-assistant`, and the materializer writes that value explicitly going forward; the audited Mistral merge protocol still requires an exact non-null match. The formerly failing 96-row Qwen condition passed a real remote preflight after this change, and no partial adapter was accepted from the failed attempt.

## Determinism boundary

Seeds, data order, decoding temperature 0, model revisions, inputs, adapters, and bootstrap indices are frozen. CUDA kernels, quantization, and parallel floating-point reductions may prevent bit-for-bit retraining. Therefore the reproducible scientific object is the saved per-sample prediction plus its lineage and paired analysis; reruns should reproduce the reported direction and uncertainty, not be judged only by identical checkpoint bytes.

## Release commands

After all result gates pass, `generate_paper_results.py` builds primary tables and factorial contrasts only from seed-level JSON marked `primary_training_seed_summary=true`: corrective Qwen and confirmatory Mistral seeds are eligible, while exploratory fixed-100 runs are excluded. It separately hashes the frozen exploratory Qwen rule analysis and emits only the explicitly labeled pilot range/worst-position macros used to explain why aggregate accuracy is insufficient; that source never enters a primary mean. `aggregate_seed_level_results.py` emits per-position means and seed-level Student-t intervals, records the included status by family, and explicitly flags that the combined primary set is not confirmatory-only. The same untrained Base copied into each seed analysis is first required to be identical and then deduplicated, so it is never presented as multiple training replicates. Position-profile grids must be identical across seeds within a model family. Across families, exact audited token lengths may differ (the controlled rule long slice is 32,768 for Qwen and 32,512 for Mistral), so those numeric grids are retained rather than silently relabeled; the shared NoLiMa 32,000-token grid supports the cross-family main figure. `plot_seed_level_factorial_results.py` case-weights the three NoLiMa task strata by their frozen 2/6/2 semantic-case counts before recomputing seed-level intervals, yielding a readable model-family × context-length main figure without treating books, placements, or prompts as extra semantic cases; task-level profiles remain in the source seed-level CSV/JSON. The script writes PDF/SVG/PNG, an exact aggregate CSV alternative, alt text, and a source/output hash manifest. Plotting clips intervals to the valid accuracy range only visually while preserving the unbounded small-sample interval in CSV.

`build_public_release.py` runs only after the full evidence manifest passes. Before trusting `final_release_ready`, it resolves every evidence path inside the project tree and recomputes its byte count and SHA-256, so an audit followed by an artifact mutation cannot produce a nominally validated public package. It includes code, configuration, documentation, paper sources, data/training manifests, statistics, bootstrap indices, license-safe failure catalogs and figures, while excluding raw benchmark/training/generation JSONL, base weights, adapters, checkpoints and partial files. This prevents the generic path sanitizer from accidentally becoming a benchmark-text redistribution tool. It still applies the same fail-closed secret and absolute-path checks and records source/public-copy hashes. Submission-mode `audit_arxiv_source.py` must then pass with zero errors/pending items, including a compiled `main.bbl`, before `package_arxiv_source.py` can create a deterministic arXiv archive; the arXiv packager independently repeats the same evidence-file rehash gate.

The public-release selection and sanitizer are preflighted against the real working tree before final evidence exists. Excluded environment links such as `.venv/bin/python` do not block selection, while a symlink inside any selected source directory is rejected. Tests that require license-excluded NoLiMa or IFEval payloads explicitly skip with reconstruction instructions when run from the sanitized package; they execute normally in the complete research tree. The historical pre-correction dry run selected 410 files, applied portability rewrites to 60 files, and passed Python and shell syntax checks; its sanitized suite passed 121 tests with 3 license-data skips by design and zero warnings, while the then-complete tree passed 124 tests. After the strict block-96 lineage, semantic-evidence, derived-manifest cross-binding, one-shot queue-progress accounting and post-audit rehash gates were added, the current complete tree passes 152 tests. The final sanitized package must still rerun the entire current suite under the real complete evidence manifest. The temporary preflight evidence stub and temporary release tree were moved to trash immediately after the audit and cannot satisfy the final gate; its selection manifest explicitly records the incomplete-evidence preflight bypass.

`requirements-test.txt` is the frozen CPU-only validation environment (`pytest`, `matplotlib`, `pyparsing`, and `huggingface-hub`) and is included in the public package; the equivalent editable-install extra is `.[test]`. The explicit `pyparsing` pin prevents dependency drift from turning Matplotlib's still-supported parser aliases into release-test warnings. The environment deliberately excludes CUDA, vLLM, Transformers training, and benchmark payloads so that release integrity can be checked independently of an AutoDL image. The checked-in pytest configuration restricts default discovery to the project-owned `tests/` directory, preventing vendored upstream test modules from silently expanding the release-validation contract.

After the final experiment queue closes, active trainer/vLLM time is recomputed from the following explicit strict roots. The command intentionally does not glob all historical experiments, so abandoned diagnostics cannot silently enter the paper total; reused Base run metadata are deduplicated by immutable run identity. Strict canaries are gated at global step 96 rather than the historical default 100:

```bash
python3 scripts/summarize_compute_accounting.py \
  --project-root "$PWD" \
  --training-root outputs/qwen_block96/seed_20260825 \
  --training-root outputs/qwen_block96/seed_20260826 \
  --training-root outputs/qwen_block96/seed_20260827 \
  --training-root outputs/mistral_block96/seed_20260825 \
  --training-root outputs/mistral_block96/seed_20260826 \
  --training-root outputs/mistral_block96/seed_20260827 \
  --eval-root results/qwen_block96_rule \
  --eval-root results/qwen_block96_nolima \
  --eval-root results/qwen_block96_longbench \
  --eval-root results/qwen_block96_mmlu \
  --eval-root results/qwen_block96_ifeval \
  --eval-root results/qwen_block96_nolima_mechanisms \
  --eval-root results/mistral_block96_rule \
  --eval-root results/mistral_block96_nolima \
  --eval-root results/mistral_block96_longbench \
  --eval-root results/mistral_block96_mmlu \
  --eval-root results/mistral_block96_ifeval \
  --eval-root results/mistral_block96_nolima_mechanisms \
  --expected-training-step 96 \
  --hourly-rate 2.78 \
  --output results/compute_accounting.json
```

`compute_accounting.json` is an engine-active lower bound, not the AutoDL bill: model loading, CPU scoring/statistics, packaging, idle allocation, restarts, and provider billing granularity remain outside that sum. The paper must use this exact label and separately describe the limitation; no estimated budget may be substituted for completed-run accounting. The final evidence audit must declare this file with `--require-evidence-label compute_accounting`; only then does it emit `final_release_ready=true`, which the non-preflight public-release builder requires.

The resumable strict-primary top-level execution entry point is `scripts/run_autodl_strict_block96_full_queue.sh`. It contains only ordered experiment and audit stages, skips a stage only when its validated completion schema and recorded artifact hashes still match, and deliberately contains no timer, polling loop, cron installation, or system power action. The older `run_autodl_full_completion_queue.sh` is retained only for historical fixed-100 lineage and is not a paper-primary entry point.

Budget snapshots are computed rather than hand-timed. `estimate_autodl_budget.py` accepts completed-workload skips, exact residual units, per-workload measured throughput, and a timezone-aware `--start-time`; its JSON records total GPU-hours, single-queue wall hours, expected finish, contingency finish, expected spend, and budget ceiling. Per-suite generation uses `estimate_eval_progress.py`, which derives its measured timestamp, saved-row counts, latest run rate, ETA, and residual cost directly from resumable JSONL and the active log.

For the multi-day strict queue, `summarize_strict_queue_progress.py` is a one-shot, read-only snapshot rather than a watcher. It requires the project root, verifies completed checkpoint-96 conditions only when the canary and exact 96-row metric trace agree, identifies the active trainer from the process table, counts resumable rows for every frozen evaluation workload, samples the GPU once, and combines those observations with `autodl_strict_block96_budget.json`. The report retains both a frozen conservative ETA/cost and a family-specific measured training calibration; a faster measured Qwen condition is never silently applied to Mistral before that family has an audited timing. The script creates no daemon, loop, cron entry, `at` job, or power action.

Author names/order, affiliations, contact email, ORCID, arXiv category and publication license require author approval and are intentionally outside autonomous experimental execution.

## Paper build toolchain

`scripts/build_paper_pdf.sh` supports either a conventional `pdflatex` plus `bibtex` installation or standalone Tectonic. The independently verified fallback is the official Linux GNU asset for Tectonic 0.17.0 (`tectonic@0.17.0`): archive size 22,749,118 bytes and SHA-256 `1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606`; its extracted executable SHA-256 is `2b3a86250906c92ed0a3ae8aaa454ec55bd6cede8593b3e549640177f6aecaa3`. `scripts/install_tectonic.sh` verifies both hashes and installs outside the repository under `$HOME/.cache/long-context-position-bias/tectonic-0.17.0/` by default. Select the executable with `--tectonic-bin` or `TECTONIC_BIN`. The compiler binary is never part of the public release or arXiv source.

Both engines build from an isolated copy of `paper/`, run the bibliography pass, reject unresolved citations/references and Overfull/Underfull boxes, require a nonempty PDF and `main.bbl`, and record the selected engine/version and SHA-256 provenance in `artifacts/position-bias-paper.build.json`. A successful scaffold build proves toolchain integrity only; the submission audit still fails while experimental or author-metadata `PENDING` guards remain.
