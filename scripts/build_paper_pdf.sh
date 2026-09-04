#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$root_dir/paper"
output_pdf="$root_dir/artifacts/position-bias-paper.pdf"
build_manifest="$root_dir/artifacts/position-bias-paper.build.json"
engine="auto"
tectonic_bin="${TECTONIC_BIN:-}"

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --paper-dir DIR          Paper source directory (default: paper/)
  --output-pdf FILE        Validated PDF destination
  --build-manifest JSON    Build provenance destination
  --engine ENGINE          auto, latex, or tectonic (default: auto)
  --tectonic-bin FILE      Tectonic executable; also accepted via TECTONIC_BIN
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper-dir) paper_dir="$2"; shift 2 ;;
    --output-pdf) output_pdf="$2"; shift 2 ;;
    --build-manifest) build_manifest="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    --tectonic-bin) tectonic_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$engine" != "auto" && "$engine" != "latex" && "$engine" != "tectonic" ]]; then
  echo "Invalid --engine: $engine (expected auto, latex, or tectonic)" >&2
  exit 2
fi
for required in main.tex references.bib generated/results.tex; do
  if [[ ! -s "$paper_dir/$required" ]]; then
    echo "Missing paper source: $paper_dir/$required" >&2
    exit 2
  fi
done

tectonic_command=""
if [[ -n "$tectonic_bin" ]]; then
  if [[ ! -x "$tectonic_bin" ]]; then
    echo "Tectonic executable is missing or not executable: $tectonic_bin" >&2
    exit 2
  fi
  tectonic_command="$tectonic_bin"
elif command -v tectonic >/dev/null 2>&1; then
  tectonic_command="$(command -v tectonic)"
fi

selected_engine="$engine"
if [[ "$selected_engine" == "auto" ]]; then
  if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    selected_engine="latex"
  elif [[ -n "$tectonic_command" ]]; then
    selected_engine="tectonic"
  else
    echo "No supported TeX toolchain found: install pdflatex+bibtex or provide Tectonic with --tectonic-bin/TECTONIC_BIN" >&2
    exit 2
  fi
fi

if [[ "$selected_engine" == "latex" ]]; then
  for command in pdflatex bibtex; do
    if ! command -v "$command" >/dev/null 2>&1; then
      echo "Missing TeX command for --engine latex: $command" >&2
      exit 2
    fi
  done
  engine_version="$(pdflatex --version | sed -n '1p'); $(bibtex --version | sed -n '1p')"
else
  if [[ -z "$tectonic_command" ]]; then
    echo "Missing Tectonic executable; use --tectonic-bin or TECTONIC_BIN" >&2
    exit 2
  fi
  engine_version="$("$tectonic_command" --version | sed -n '1p')"
fi

build_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$build_dir"
}
trap cleanup EXIT
cp -a "$paper_dir/." "$build_dir/"
cd "$build_dir"
if [[ "$selected_engine" == "latex" ]]; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  bibtex main >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
else
  if ! "$tectonic_command" -X compile --keep-intermediates --keep-logs --outdir "$build_dir" "$build_dir/main.tex" >tectonic-output.log 2>&1; then
    echo "Tectonic build failed; final output follows:" >&2
    tail -80 tectonic-output.log >&2
    exit 1
  fi
fi
if grep -n -i -E 'undefined references|Citation .* undefined|Reference .* undefined|There were undefined references|Warning--I didn.t find a database entry' main.log main.blg >/dev/null; then
  echo "TeX build contains undefined citations or references" >&2
  exit 1
fi
if grep -n -E 'Overfull \\[hv]box|Underfull \\[hv]box' main.log >/dev/null; then
  echo "TeX build contains overfull or underfull layout boxes" >&2
  exit 1
fi
if [[ ! -s main.pdf || ! -s main.bbl ]]; then
  echo "TeX build did not produce main.pdf and main.bbl" >&2
  exit 1
fi
page_count="$(python3 - main.log <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"Output written on .*?\((\d+) pages?[,)]", text)
if not matches or int(matches[-1]) <= 0:
    raise SystemExit(1)
print(int(matches[-1]))
PY
)" || {
  echo "TeX build log does not contain a valid PDF page count" >&2
  exit 1
}

mkdir -p "$(dirname "$output_pdf")" "$(dirname "$build_manifest")"
pdf_temp="$output_pdf.tmp"
bbl_temp="$paper_dir/main.bbl.tmp"
cp main.pdf "$pdf_temp"
cp main.bbl "$bbl_temp"
mv "$pdf_temp" "$output_pdf"
mv "$bbl_temp" "$paper_dir/main.bbl"
python3 - "$paper_dir" "$output_pdf" "$build_manifest" "$selected_engine" "$engine_version" "$page_count" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
paper, pdf, output = map(Path, sys.argv[1:4])
engine_name, engine_version, page_count = sys.argv[4:]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": "paper-pdf-build-v2",
    "status": "validated",
    "built_at": datetime.now(timezone.utc).isoformat(),
    "engine": {"name": engine_name, "version": engine_version},
    "pdf": {"bytes": pdf.stat().st_size, "pages": int(page_count), "sha256": sha(pdf)},
    "bbl": {"bytes": (paper / "main.bbl").stat().st_size, "sha256": sha(paper / "main.bbl")},
    "sources": {
        name: sha(paper / name)
        for name in ("main.tex", "references.bib", "generated/results.tex")
    },
    "undefined_references": 0,
    "layout_box_warnings": 0,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "Built validated paper PDF with $selected_engine: $output_pdf"
