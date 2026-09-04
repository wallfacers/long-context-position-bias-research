# Paper and arXiv source

`main.tex` is the full technical-report manuscript. It intentionally contains red `PENDING` guards until the corrective Qwen and prospective confirmatory Mistral strict experiments, plus author metadata, are complete; a pending manuscript is not submission-ready.

Final numerical claims must be generated from audited JSON into `generated/results.tex`; the file supplies the explicitly labeled exploratory Qwen rule-range/worst-position macros, plus strict-primary controlled-rule, NoLiMa-Hard and LongBench tables containing the fixed Base and all six factorial cells, the preregistered key factorial contrasts with seed-level intervals, and representative-seed regression/mechanism tables. The generator requires Qwen to be labeled `corrective` and Mistral `confirmatory`, and records that their combined primary summary is not confirmatory-only. The exploratory source is hashed but never enters a strict primary mean. The main position figure is generated directly from the cross-family NoLiMa seed-level analysis into `figures/`, case-weighting the three task strata by their frozen 2/6/2 semantic-case counts before recomputing seed-level intervals; task-level profiles remain in the source analysis CSV/JSON. The renderer writes PDF/SVG/PNG, an exact aggregate CSV, alt text and a source/output hash manifest. Do not hand-copy headline numbers or substitute a single-seed plot for the strict cross-family figure.

Build with an existing LaTeX/BibTeX installation, or with the pinned standalone Tectonic 0.17.0 binary:

```bash
bash ../scripts/install_tectonic.sh

bash ../scripts/build_paper_pdf.sh

# Explicit standalone build (the environment variable is equivalent).
bash ../scripts/build_paper_pdf.sh \
  --engine tectonic \
  --tectonic-bin "$HOME/.cache/long-context-position-bias/tectonic-0.17.0/tectonic"
```

`--engine auto` (the default) prefers an installed `pdflatex` plus `bibtex`, otherwise it uses `--tectonic-bin`, `TECTONIC_BIN`, or `tectonic` on `PATH`. The verified Linux GNU archive is `tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz` from the official `tectonic@0.17.0` GitHub release: 22,749,118 bytes, SHA-256 `1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606`. The binary is a build dependency only and must not be added to the arXiv or public-source package.

The build runs the full LaTeX/BibTeX sequence in an isolated temporary directory (Tectonic invokes BibTeX internally), rejects undefined citations/references and Overfull/Underfull layout boxes, requires both PDF and `main.bbl`, atomically exports them, and records the engine/version, page count, and source/PDF hashes. A missing or invalid page count in the TeX log is also a hard failure.

Before packaging:

```bash
python3 ../scripts/audit_arxiv_source.py --paper-dir . --output arxiv-audit.json
```

The submission-mode audit also requires the compiled `main.bbl`. Once it passes, create the deterministic source archive and SHA-traceable manifest with:

```bash
python3 ../scripts/package_arxiv_source.py \
  --paper-dir . \
  --evidence-manifest ../results/full_paper_evidence_manifest.json \
  --output-tar ../artifacts/arxiv-source.tar.gz \
  --output-manifest ../artifacts/arxiv-source.manifest.json
```

The final arXiv action is blocked until the audit passes and all authors approve names, order, affiliations, contact email, category, and license.

`responsible-nlp-checklist.md` records the evidence-backed draft answers for a later ACL/ARR form; the current venue form and any required tool-assistance disclosure must still be confirmed at submission time.
