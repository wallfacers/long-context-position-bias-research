# Responsible NLP checklist draft

This file prepares the factual answers needed for an ACL/ARR-style responsible-research checklist. It is not a substitute for the venue's current submission form, which must be checked at submission time.

| Topic | Project answer | Evidence in package |
|---|---|---|
| Limitations stated | Yes | `main.tex`, Limitations |
| Claims match evidence | Enforced before release | seed-level JSON, generated TeX manifest, arXiv audit |
| Hyperparameters reported | Yes | Appendix; per-run `run_config.json` |
| Random seeds reported | Yes | 20260825/26/27; pilot designation is explicit |
| Uncertainty/statistical tests | Yes | paired group bootstrap and seed-level Student-t intervals |
| Compute and hardware reported | Pending final queue closure | reproducibility JSON, `nvidia-smi`, cost ledger/budget, manuscript compute statement |
| Data provenance reported | Yes | pinned revisions, generators, manifests, SHA-256 |
| Data licenses respected | Yes | NoLiMa Adobe Research License/noncommercial; LongBench and constituent terms retained |
| Evaluation data used for training | No | separate frozen source manifests and hashes |
| Proprietary teacher data | No | deterministic local targets; no cloud-model distillation |
| Personal/sensitive data collected | No | synthetic training facts; public benchmark text only |
| Human subjects or annotators | No | no new human data collection or annotation study |
| Model credentials released | No | source audit rejects credentials, SSH endpoints, and private paths |
| Failure/negative results reported | Required | frozen go/no-go rules forbid deleting failed conditions; deterministic license-safe failure catalogs |
| Generalization scope bounded | Yes | two approximately 7B families; no scaling-law claim |
| Environmental/financial cost | Pending final queue closure | measured sequential GPU time and AutoDL rental cost; carbon non-estimation rationale in manuscript |

Before an actual venue submission, the authors must transfer these facts into that venue's current checklist, confirm any changed wording, and disclose any writing or coding assistance required by the venue policy in force at that time.
