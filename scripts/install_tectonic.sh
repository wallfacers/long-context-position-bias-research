#!/usr/bin/env bash
set -Eeuo pipefail

version="0.17.0"
asset="tectonic-${version}-x86_64-unknown-linux-gnu.tar.gz"
archive_sha256="1a715688baf591e650c8aeb160ae934e181685eecbb38b317de30b269ac5d606"
binary_sha256="2b3a86250906c92ed0a3ae8aaa454ec55bd6cede8593b3e549640177f6aecaa3"
url="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${version}/${asset}"
install_dir="${XDG_CACHE_HOME:-$HOME/.cache}/long-context-position-bias/tectonic-${version}"

usage() {
  cat <<EOF
Usage: $0 [--install-dir DIR]

Download and verify the pinned official Tectonic ${version} Linux GNU release.
The default installation is outside the repository:
  ${install_dir}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir) install_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for command in awk curl sha256sum tar; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 2
  fi
done

binary="$install_dir/tectonic"
archive="$install_dir/$asset"
if [[ -e "$binary" ]]; then
  actual_binary_sha256="$(sha256sum "$binary" | awk '{print $1}')"
  if [[ -x "$binary" && "$actual_binary_sha256" == "$binary_sha256" ]]; then
    echo "Pinned Tectonic already validated: $binary"
    exit 0
  fi
  echo "Refusing to replace unexpected existing file: $binary" >&2
  echo "Expected executable SHA-256: $binary_sha256" >&2
  echo "Observed SHA-256: $actual_binary_sha256" >&2
  exit 1
fi

download_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$download_dir"
}
trap cleanup EXIT

curl --fail --location --retry 3 --retry-all-errors \
  --output "$download_dir/$asset" "$url"
printf '%s  %s\n' "$archive_sha256" "$download_dir/$asset" | sha256sum --check --status
mkdir -p "$download_dir/extracted"
tar -xzf "$download_dir/$asset" -C "$download_dir/extracted"
if [[ ! -f "$download_dir/extracted/tectonic" ]]; then
  echo "Verified archive did not contain the expected tectonic executable" >&2
  exit 1
fi
actual_binary_sha256="$(sha256sum "$download_dir/extracted/tectonic" | awk '{print $1}')"
if [[ "$actual_binary_sha256" != "$binary_sha256" ]]; then
  echo "Extracted binary SHA-256 mismatch" >&2
  exit 1
fi

mkdir -p "$install_dir"
cp "$download_dir/$asset" "$archive.tmp"
cp "$download_dir/extracted/tectonic" "$binary.tmp"
chmod 0755 "$binary.tmp"
mv "$archive.tmp" "$archive"
mv "$binary.tmp" "$binary"

echo "Installed and validated Tectonic ${version}: $binary"
echo "Use: TECTONIC_BIN='$binary' bash scripts/build_paper_pdf.sh"
